from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import date, datetime
from typing import Any, Dict
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from pydantic import ValidationError
import wolframalpha

from models import (
    AgentResponse,
    DashboardStatsResponse,
    ExamCountdown,
    LongTermGoal,
    QueryInput,
    TraceStep,
)


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID")

EXAM_TARGETS = {
    "IOQM": {
        "exam_date": date(2026, 9, 6),
        "target_syllabus_percent": 85,
        "target_problem_count": 600,
        "focus_area": "Proof writing, algebraic manipulation, and geometry consistency.",
    },
    "NMTC": {
        "exam_date": date(2026, 10, 17),
        "target_syllabus_percent": 90,
        "target_problem_count": 800,
        "focus_area": "Contest-style logic, number theory, and timed puzzle solving.",
    },
    "NSEJS": {
        "exam_date": date(2026, 11, 22),
        "target_syllabus_percent": 95,
        "target_problem_count": 1500,
        "focus_area": "Mixed physics numericals, chemistry recall, and speed math drills.",
    },
}
LONG_TERM_GOAL_DATE = date(2028, 1, 15)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("addix-neuro-symbolic")

client: genai.Client | None = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="ADDIX Labs Neuro-Symbolic API",
    description="Neuro-symbolic multi-agent backend for educational problem solving.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:3000",
        "http://localhost:5500",
        "https://addixlabs.in",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("A critical unhandled exception occurred", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "final_answer": "The multi-agent system encountered a critical error while processing your request.",
            "explanation_trace": [],
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error("HTTPException occurred: %s", exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "final_answer": "The multi-agent system encountered an error while processing your request.",
            "detail": exc.detail,
            "explanation_trace": [],
        },
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error("Pydantic ValidationError occurred", exc_info=exc)
    return JSONResponse(
        status_code=422,
        content={
            "final_answer": "The multi-agent system encountered a validation error while processing your request.",
            "explanation_trace": [],
        },
    )


def _days_remaining(target_date: date) -> int:
    return (target_date - datetime.now().date()).days


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned


def _clean_and_parse_json_field(result_text: str, field_name: str) -> str:
    cleaned_text = _strip_code_fences(result_text)
    try:
        parsed = json.loads(cleaned_text)
    except json.JSONDecodeError:
        field_marker = f'"{field_name}"'
        marker_index = cleaned_text.find(field_marker)
        if marker_index == -1:
            raise ValueError(f"Missing expected field '{field_name}' in model response.")

        colon_index = cleaned_text.find(":", marker_index + len(field_marker))
        opening_quote_index = cleaned_text.find('"', colon_index + 1)
        closing_brace_index = cleaned_text.rfind("}")
        search_end = closing_brace_index if closing_brace_index != -1 else len(cleaned_text)
        closing_quote_index = cleaned_text.rfind('"', opening_quote_index + 1, search_end)
        if colon_index == -1 or opening_quote_index == -1 or closing_quote_index == -1:
            raise ValueError("Could not isolate the JSON string field returned by Gemini.")

        raw_value = cleaned_text[opening_quote_index + 1:closing_quote_index]
        hardened_value = raw_value.replace("\\", "\\\\").replace('"', '\\"')
        return json.loads(f'{{"{field_name}":"{hardened_value}"}}')[field_name]

    value = parsed.get(field_name, "")
    if isinstance(value, str):
        return value
    return str(value or parsed)


def _extract_text_from_gemini_response(response: Any) -> str:
    text = getattr(response, "text", None)
    if text and text.strip():
        return text.strip()

    try:
        candidates = getattr(response, "candidates", [])
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) if content else []
            collected = [getattr(part, "text", None) for part in parts if getattr(part, "text", None)]
            joined = "\n".join(collected).strip()
            if joined:
                return joined
    except Exception:
        logger.exception("Failed while extracting text from Gemini response fallback path.")

    raise RuntimeError("Gemini returned an empty response.")


def _gemini_generate(prompt: str, *, json_mode: bool = False) -> str:
    if not client:
        raise RuntimeError("Gemini client is not configured.")

    config = types.GenerateContentConfig(
        response_mime_type="application/json" if json_mode else "text/plain"
    )
    result = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=config,
    )
    return _extract_text_from_gemini_response(result)


def _safe_format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _extract_known_values(student_query: str) -> list[str]:
    matches = re.findall(
        r"\b\d+(?:\.\d+)?\s*(?:km/h|kmph|kph|m/s|mps|seconds?|secs?|s|minutes?|min|hours?|h|meters?|metres?|m)\b",
        student_query,
        flags=re.IGNORECASE,
    )
    return matches


def _extract_unknown_target(student_query: str) -> str:
    match = re.search(r"(what is|find|determine|calculate)\s+(.+?)(?:\?|$)", student_query, flags=re.IGNORECASE)
    if match:
        return match.group(2).strip().rstrip(".")
    return "Determine the quantity requested by the student."


def _convert_speed_to_mps(speed_value: float, speed_unit: str) -> float:
    normalized = speed_unit.lower().replace(" ", "")
    if normalized in {"m/s", "mps"}:
        return speed_value
    if normalized in {"km/h", "kmph", "kph"}:
        return speed_value / 3.6
    raise ValueError(f"Unsupported speed unit: {speed_unit}")


def _solve_kinematics_locally(student_query: str) -> dict[str, Any] | None:
    match = re.search(
        r"from\s+rest\s+to\s+a\s+speed\s+of\s+(\d+(?:\.\d+)?)\s*(km/h|kmph|kph|m/s|mps)\s+in\s+(\d+(?:\.\d+)?)\s*(seconds?|secs?|s)",
        student_query,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    speed_value = float(match.group(1))
    speed_unit = match.group(2)
    time_seconds = float(match.group(3))
    final_speed_mps = _convert_speed_to_mps(speed_value, speed_unit)
    average_speed_mps = final_speed_mps / 2
    distance_meters = average_speed_mps * time_seconds
    acceleration_mps2 = final_speed_mps / time_seconds

    final_speed_text = _safe_format_number(final_speed_mps)
    average_speed_text = _safe_format_number(average_speed_mps)
    distance_text = _safe_format_number(distance_meters)
    time_text = _safe_format_number(time_seconds)
    acceleration_text = _safe_format_number(acceleration_mps2)

    return {
        "known_values": [
            "Initial speed u = 0 m/s",
            f"Final speed v = {speed_value:g} {speed_unit}",
            f"Time t = {time_text} s",
        ],
        "unknown_target": "Distance covered during the acceleration interval",
        "principle": "Uniform acceleration: average speed = (u + v) / 2 and distance = average speed x time.",
        "strategy": (
            f"1. Convert {speed_value:g} {speed_unit} to {final_speed_text} m/s.\n"
            f"2. Compute average speed = (0 + {final_speed_text}) / 2 = {average_speed_text} m/s.\n"
            f"3. Compute distance = {average_speed_text} x {time_text} = {distance_text} m."
        ),
        "wolfram_query": f"(({final_speed_text} m/s)/2) * {time_text} s",
        "calculation_text": (
            f"{speed_value:g} {speed_unit} = {final_speed_text} m/s, "
            f"so distance = ((0 + {final_speed_text}) / 2) x {time_text} = {distance_text} meters."
        ),
        "validation_text": (
            f"The acceleration is {acceleration_text} m/s^2, the average speed is {average_speed_text} m/s, "
            f"and over {time_text} s the car covers {distance_text} m. The units and magnitude are consistent."
        ),
        "final_answer_text": f"{distance_text} meters",
        "final_answer_number": distance_text,
    }


def _local_deconstruct_problem(student_query: str, local_solution: dict[str, Any] | None, reason: str) -> str:
    known_values = local_solution["known_values"] if local_solution else [
        *(f"Given quantity: {value}" for value in _extract_known_values(student_query)),
    ]
    if not known_values:
        known_values = ["Given quantities must be inferred from the student statement."]

    unknown_target = local_solution["unknown_target"] if local_solution else _extract_unknown_target(student_query)
    principle = (
        local_solution["principle"]
        if local_solution
        else "Use the governing relationship that connects the known quantities to the requested target."
    )

    return (
        f"{reason}\n"
        "Known Variables:\n"
        + "\n".join(f"- {item}" for item in known_values)
        + f"\nUnknown Target:\n- {unknown_target}\n"
        + f"Core Scientific/Mathematical Principles:\n- {principle}"
    )


def _local_strategy(local_solution: dict[str, Any] | None, reason: str) -> str:
    if local_solution:
        return f"{reason}\n{local_solution['strategy']}"
    return (
        f"{reason}\n"
        "1. Identify the known quantities and the final target.\n"
        "2. Choose the governing equation that links the known values to the unknown.\n"
        "3. Substitute values carefully with consistent units.\n"
        "4. Check whether the final answer is physically reasonable."
    )


def _local_symbolic_query(student_query: str, local_solution: dict[str, Any] | None) -> str:
    if local_solution:
        return local_solution["wolfram_query"]
    stripped = re.sub(r"\s+", " ", student_query.strip().strip("`"))
    return re.sub(r"[?]+$", "", stripped)


def _local_calculation_fallback(
    student_query: str,
    wolfram_query: str,
    local_solution: dict[str, Any] | None,
    reason: str,
) -> str:
    if local_solution:
        return f"{reason}\n{local_solution['calculation_text']}"
    return (
        f"{reason}\n"
        f"No direct symbolic answer was available for: {wolfram_query or student_query}\n"
        "Proceeding with a qualitative fallback explanation instead of failing the request."
    )


def _local_sanity_check(local_solution: dict[str, Any] | None, reason: str) -> str:
    if local_solution:
        return f"{reason}\n{local_solution['validation_text']}"
    return (
        f"{reason}\n"
        "The fallback path preserved the reasoning structure, but the numeric result should be verified once "
        "external engines are available again."
    )


def _local_final_synthesis(
    student_query: str,
    target_exam: str,
    deconstruction: str,
    strategy: str,
    calculation: str,
    validation: str,
    local_solution: dict[str, Any] | None,
    reason: str,
) -> str:
    final_answer_line = (
        local_solution["final_answer_text"]
        if local_solution
        else "A graceful fallback explanation was generated, but a verified numeric result is not available."
    )
    return (
        f"Final Answer: {final_answer_line}\n\n"
        f"Target Exam: {target_exam}\n"
        f"Problem: {student_query}\n\n"
        f"Deconstruction\n{deconstruction}\n\n"
        f"Strategy\n{strategy}\n\n"
        f"Calculation\n{calculation}\n\n"
        f"Validation\n{validation}\n\n"
        f"System Note: {reason}"
    )


def _gemini_deconstruct_problem(student_query: str) -> str:
    prompt = (
        "Return JSON only: {\"deconstruction_markdown\":\"...\"}. "
        "List known values, unknown target, and core principle in short bullets. "
        "Double-escape LaTeX backslashes.\n\n"
        f"Problem: {student_query}"
    )
    return _clean_and_parse_json_field(_gemini_generate(prompt, json_mode=True), "deconstruction_markdown")


def _gemini_formulate_strategy(student_query: str, deconstruction: str) -> str:
    prompt = (
        "Return JSON only: {\"strategy_markdown\":\"...\"}. "
        "Write a short numbered plan with only the essential equations. "
        "Double-escape LaTeX backslashes.\n\n"
        f"Problem: {student_query}\n"
        f"Deconstruction: {deconstruction}"
    )
    return _clean_and_parse_json_field(_gemini_generate(prompt, json_mode=True), "strategy_markdown")


def _gemini_translate_to_wolfram(student_query: str, strategy: str) -> str:
    prompt = (
        "Return only the single raw Wolfram query needed for the key calculation. "
        "No quotes. No markdown.\n\n"
        f"Problem: {student_query}\n"
        f"Strategy: {strategy}"
    )
    return _gemini_generate(prompt).strip().strip("`")


def _is_valid_wolfram_answer(answer_text: str) -> bool:
    if not answer_text:
        return False
    normalized = answer_text.strip().lower()
    invalid_markers = (
        "no direct result found.",
        "no short answer available",
        "did not understand your input",
        "(data not available)",
        "error",
    )
    return all(marker not in normalized for marker in invalid_markers)


def _extract_wolfram_answer(result: Any) -> str:
    try:
        answer_text = next(result.results).text
        if _is_valid_wolfram_answer(answer_text):
            return answer_text.strip()
    except StopIteration:
        pass
    except Exception as exc:
        logger.warning("Failed reading primary Wolfram result: %s", exc)

    try:
        pods = list(getattr(result, "pods", []))
        for pod in pods:
            for subpod in getattr(pod, "subpods", []):
                text = getattr(subpod, "plaintext", "")
                if _is_valid_wolfram_answer(text):
                    return text.strip()
    except Exception as exc:
        logger.warning("Failed parsing Wolfram fallback pods: %s", exc)

    return ""


def _build_wolfram_query_candidates(query: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", query.strip().strip("`").replace("\n", " "))
    normalized = re.sub(r"[?]+$", "", normalized)
    candidates: list[str] = []
    filler_pattern = re.compile(
        r"\b(?:what|is|the|find|determine|calculate|distance|covered|car|accelerates|uniformly|from|rest|to|a|speed|of|in|this|time|during|by|please)\b",
        flags=re.IGNORECASE,
    )

    def add_candidate(candidate: str) -> None:
        clean_candidate = candidate.strip(" ,;")
        if clean_candidate and clean_candidate not in candidates:
            candidates.append(clean_candidate)

    add_candidate(normalized)
    after_colon = normalized.split(":")[-1].strip()
    add_candidate(after_colon)
    add_candidate(re.sub(r"^[A-Za-z ,:;-]*?(?=(\d|\())", "", normalized).strip())
    if "=" in normalized:
        add_candidate(normalized.split("=", 1)[1].strip())
    add_candidate(" ".join(re.findall(r"\d+(?:\.\d+)?|[A-Za-z]+(?:/[A-Za-z]+)?|to|in|[()+\-*/^=]", normalized)))
    add_candidate(re.sub(r"\s+", " ", filler_pattern.sub(" ", normalized)).strip())

    conversion_match = re.search(
        r"(\d+(?:\.\d+)?)\s*([A-Za-z]+(?:/[A-Za-z]+)?)\s*(?:to|in)\s*([A-Za-z]+(?:/[A-Za-z]+)?)",
        normalized,
        flags=re.IGNORECASE,
    )
    if conversion_match:
        value, from_unit, to_unit = conversion_match.groups()
        add_candidate(f"{value} {from_unit} to {to_unit}")

    return candidates


def _wolfram_short_answer(query: str) -> str:
    if not WOLFRAM_APP_ID:
        return ""
    encoded_query = urllib_parse.urlencode({"appid": WOLFRAM_APP_ID, "i": query})
    short_answer_url = f"https://api.wolframalpha.com/v1/result?{encoded_query}"
    try:
        with urllib_request.urlopen(short_answer_url, timeout=15) as response:
            answer_text = response.read().decode("utf-8").strip()
    except urllib_error.HTTPError as exc:
        answer_text = exc.read().decode("utf-8", errors="ignore").strip()
    except Exception as exc:
        logger.warning("Wolfram short-answer request failed for '%s': %s", query, exc)
        return ""
    return answer_text if _is_valid_wolfram_answer(answer_text) else ""


def _wolfram_query_robust(query: str) -> Dict[str, str]:
    if not WOLFRAM_APP_ID:
        raise RuntimeError("Wolfram App ID is not configured.")

    client_wa = wolframalpha.Client(WOLFRAM_APP_ID)
    for candidate in _build_wolfram_query_candidates(query):
        short_answer = _wolfram_short_answer(candidate)
        if _is_valid_wolfram_answer(short_answer):
            return {"answer": short_answer, "latex": candidate}

        try:
            result = client_wa.query(candidate, params=(("format", "plaintext"),))
            answer_text = _extract_wolfram_answer(result)
            if _is_valid_wolfram_answer(answer_text):
                return {"answer": answer_text, "latex": candidate}
        except Exception as exc:
            logger.warning("Wolfram query failed for '%s': %s", candidate, exc)

    raise RuntimeError("Wolfram Alpha could not resolve the query after all fallback attempts.")


def _gemini_fallback_calculation(wolfram_query: str) -> str:
    prompt = (
        "Evaluate this expression and return only the result with units when relevant.\n\n"
        f"Expression: {wolfram_query}"
    )
    return _gemini_generate(prompt)


def _gemini_sanity_check(student_query: str, wolfram_answer: str) -> str:
    prompt = (
        "Give a brief check of whether this answer is mathematically and physically reasonable. Mention units.\n\n"
        f"Problem: {student_query}\n"
        f"Result: {wolfram_answer}"
    )
    return _gemini_generate(prompt)


def _gemini_final_synthesis(
    student_query: str,
    target_exam: str,
    deconstruction: str,
    strategy: str,
    calculation: str,
    validation: str,
) -> str:
    prompt = (
        f"Return JSON only for a {target_exam} student: "
        "{\"final_response_markdown\":\"...\"}. "
        "The markdown must begin with 'Final Answer:' and include short deconstruction, strategy, calculation, and validation sections.\n\n"
        f"Problem: {student_query}\n"
        f"Deconstruction: {deconstruction}\n"
        f"Strategy: {strategy}\n"
        f"Calculation Result: {calculation}\n"
        f"Validation: {validation}"
    )
    result_text = _gemini_generate(prompt, json_mode=True)
    clean_json = result_text.strip().replace("```json", "").replace("```", "")
    clean_json = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r"\\\\", clean_json)
    try:
        parsed = json.loads(clean_json)
    except json.JSONDecodeError:
        return _clean_and_parse_json_field(clean_json, "final_response_markdown")

    final_response_markdown = parsed.get("final_response_markdown", "")
    if isinstance(final_response_markdown, str):
        return final_response_markdown
    return str(final_response_markdown or parsed)


def _final_answer_matches_local_solution(final_answer: str, local_solution: dict[str, Any] | None) -> bool:
    if not local_solution:
        return True
    normalized = final_answer.lower()
    return (
        local_solution["final_answer_number"] in normalized
        or local_solution["final_answer_text"].lower() in normalized
    )


def _build_trace_step(
    step_number: int,
    title: str,
    description: str,
    agent_type: str,
    math_latex: str | None = None,
) -> TraceStep:
    return TraceStep(
        step_number=step_number,
        title=title,
        description=description,
        agent_type=agent_type,
        math_latex=math_latex,
    )


def _pad_trace_to_five_steps(trace: list[TraceStep]) -> list[TraceStep]:
    defaults = {
        1: ("Problem Deconstruction", "Neural fallback placeholder added to preserve trace integrity.", "Neural"),
        2: ("Strategy Formulation", "Neural fallback placeholder added to preserve trace integrity.", "Neural"),
        3: ("Symbolic Translation & Execution", "Symbolic fallback placeholder added to preserve trace integrity.", "Symbolic"),
        4: ("Sanity Check", "Neural fallback placeholder added to preserve trace integrity.", "Neural"),
        5: ("Final Synthesis", "Neural fallback placeholder added to preserve trace integrity.", "Neural"),
    }
    while len(trace) < 5:
        step_number = len(trace) + 1
        title, description, agent_type = defaults[step_number]
        trace.append(_build_trace_step(step_number, title, description, agent_type))
    return trace[:5]


def _build_dashboard_stats() -> DashboardStatsResponse:
    exam_countdowns = {
        name: ExamCountdown(
            exam_date=config["exam_date"].isoformat(),
            days_remaining=_days_remaining(config["exam_date"]),
            target_syllabus_percent=config["target_syllabus_percent"],
            target_problem_count=config["target_problem_count"],
            focus_area=config["focus_area"],
        )
        for name, config in EXAM_TARGETS.items()
    }
    long_term_goals = {
        "JEE_MAIN_2028": LongTermGoal(
            target_date=LONG_TERM_GOAL_DATE.isoformat(),
            days_remaining=_days_remaining(LONG_TERM_GOAL_DATE),
            goal_percent=10,
            milestone="Complete 10% of the JEE Main foundation syllabus by January 15, 2028.",
        )
    }
    return DashboardStatsResponse(
        student_grade=9,
        academic_year="2026-2027",
        active_tracks=["IOQM", "NMTC", "NSEJS", "JEE Main 2028"],
        exam_countdowns=exam_countdowns,
        long_term_goals=long_term_goals,
        today_sessions=3,
        weekly_growth_percent=8.5,
        recommended_focus="Prioritize IOQM proof practice now, then ramp into NMTC speed work and NSEJS mixed science revision.",
    )


@app.get("/")
def root_status() -> Dict[str, str]:
    return {"status": "ADDIX Systems Online", "tier": "Student"}


@app.post("/api/solve", response_model=AgentResponse)
async def solve_student_query(payload: QueryInput) -> AgentResponse:
    student_query = payload.student_query.strip()
    target_exam = payload.target_exam.strip()
    trace: list[TraceStep] = []
    local_solution = _solve_kinematics_locally(student_query)

    deconstruction = _local_deconstruct_problem(
        student_query,
        local_solution,
        "Neural fallback activated because the structured deconstruction service was unavailable.",
    )
    strategy = _local_strategy(
        local_solution,
        "Neural fallback activated because the strategy service was unavailable.",
    )
    wolfram_query = _local_symbolic_query(student_query, local_solution)
    calculation = _local_calculation_fallback(
        student_query,
        wolfram_query,
        local_solution,
        "Neural fallback activated because the symbolic service was unavailable.",
    )
    validation = _local_sanity_check(
        local_solution,
        "Neural fallback activated because the validation service was unavailable.",
    )
    final_explanation = _local_final_synthesis(
        student_query,
        target_exam,
        deconstruction,
        strategy,
        calculation,
        validation,
        local_solution,
        "Local deterministic synthesis was used.",
    )

    try:
        try:
            deconstruction = _gemini_deconstruct_problem(student_query)
            step1_description = deconstruction
        except Exception as exc:
            logger.warning("Deconstruction fallback engaged: %s", exc)
            step1_description = _local_deconstruct_problem(
                student_query,
                local_solution,
                f"Neural fallback engaged for deconstruction: {exc}",
            )
            deconstruction = step1_description
        trace.append(_build_trace_step(1, "Problem Deconstruction", step1_description, "Neural"))

        try:
            strategy = _gemini_formulate_strategy(student_query, deconstruction)
            step2_description = strategy
        except Exception as exc:
            logger.warning("Strategy fallback engaged: %s", exc)
            step2_description = _local_strategy(
                local_solution,
                f"Neural fallback engaged for strategy: {exc}",
            )
            strategy = step2_description
        trace.append(_build_trace_step(2, "Strategy Formulation", step2_description, "Neural"))

        await asyncio.sleep(2)

        try:
            wolfram_query = _gemini_translate_to_wolfram(student_query, strategy)
        except Exception as exc:
            logger.warning("Wolfram translation fallback engaged: %s", exc)
            wolfram_query = _local_symbolic_query(student_query, local_solution)

        symbolic_description = f"Wolfram Alpha triggered with query: {wolfram_query}"
        try:
            symbolic_result = _wolfram_query_robust(wolfram_query)
            wolfram_query = symbolic_result["latex"]
            calculation = symbolic_result["answer"]
            symbolic_description = f"Wolfram Alpha Request: {wolfram_query}\nResult: {calculation}"
        except Exception as symbolic_exc:
            logger.warning("Symbolic fallback engaged: %s", symbolic_exc)
            try:
                calculation = _gemini_fallback_calculation(wolfram_query)
                symbolic_description = (
                    f"Wolfram attempt failed for query: {wolfram_query}\n"
                    f"Neural Fallback Result: {calculation}"
                )
            except Exception as neural_exc:
                logger.warning("Neural calculation fallback engaged: %s", neural_exc)
                calculation = _local_calculation_fallback(
                    student_query,
                    wolfram_query,
                    local_solution,
                    f"Symbolic and neural engines were unavailable: {neural_exc}",
                )
                symbolic_description = (
                    f"Wolfram attempt failed for query: {wolfram_query}\n"
                    f"Neural Fallback: {calculation}"
                )
        trace.append(
            _build_trace_step(
                3,
                "Symbolic Translation & Execution",
                symbolic_description,
                "Symbolic",
                wolfram_query,
            )
        )

        try:
            validation = _gemini_sanity_check(student_query, calculation)
            step4_description = validation
        except Exception as exc:
            logger.warning("Sanity-check fallback engaged: %s", exc)
            step4_description = _local_sanity_check(
                local_solution,
                f"Neural fallback engaged for validation: {exc}",
            )
            validation = step4_description
        trace.append(_build_trace_step(4, "Sanity Check", step4_description, "Neural"))

        try:
            final_explanation = _gemini_final_synthesis(
                student_query,
                target_exam,
                deconstruction,
                strategy,
                calculation,
                validation,
            )
            if not _final_answer_matches_local_solution(final_explanation, local_solution):
                raise ValueError("Final synthesis omitted the deterministic local answer.")
            step5_description = (
                "Synthesized structured markdown response using the neural tutor with cleaned JSON parsing."
            )
        except Exception as exc:
            logger.warning("Final-synthesis fallback engaged: %s", exc)
            final_explanation = _local_final_synthesis(
                student_query,
                target_exam,
                deconstruction,
                strategy,
                calculation,
                validation,
                local_solution,
                f"Neural fallback engaged for final synthesis: {exc}",
            )
            step5_description = "Local deterministic synthesis generated the final markdown response."
        trace.append(_build_trace_step(5, "Final Synthesis", step5_description, "Neural"))
    except Exception as exc:
        logger.error("Unexpected failure during solve flow.", exc_info=exc)
        trace = _pad_trace_to_five_steps(trace)
        final_explanation = _local_final_synthesis(
            student_query,
            target_exam,
            deconstruction,
            strategy,
            calculation,
            validation,
            local_solution,
            f"Emergency fallback was used after an unexpected backend error: {exc}",
        )
        return AgentResponse(final_answer=final_explanation, explanation_trace=trace)

    return AgentResponse(final_answer=final_explanation, explanation_trace=_pad_trace_to_five_steps(trace))


@app.get("/api/dashboard-stats", response_model=DashboardStatsResponse)
def dashboard_stats() -> DashboardStatsResponse:
    return _build_dashboard_stats()


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}
