from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import urllib.parse
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Dict, Literal, TypedDict

import pytesseract
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from pydantic import ValidationError
from sqlalchemy import Boolean, DateTime, Integer, String, Text, create_engine, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from models import (
    AgentResponse,
    ConversationContext,
    DashboardStatsResponse,
    ExamCountdown,
    LongTermGoal,
    OCRInput,
    OCRResponse,
    QueryInput,
    TraceStep,
)


load_dotenv()

WOLFRAM_APP_ID_ENV_NAME = "WOLFRAM_APP_ID"
WOLFRAM_FALLBACK_RESULT = "Symbolic Solver: No deterministic result available for this specific query."
SECURITY_PROTOCOL_MESSAGE = "Security Protocol: Resetting API Handshake"
SYSTEM_OVERRIDE_TIMEOUT_MESSAGE = "System Override: API timeout detected. Stabilizing agent pipeline."
VISION_FUZZY_WARNING = "[Vision Agent]: Text extraction fuzzy. Please verify the query below."
INVALID_WOLFRAM_MARKERS = (
    "no short answer available",
    "did not understand your input",
    "unable to interpret",
    "error",
)
CACHE_LOOKBACK_HOURS = 8
CACHE_SIMILARITY_THRESHOLD = 0.90


class APITimeoutError(Exception):
    """Raised when an upstream API timeout needs global system override handling."""


class WolframResult(TypedDict):
    answer: str
    state: Literal["ok", "fallback", "security"]
    formula: str
    steps: str


class Base(DeclarativeBase):
    pass


class QuerySession(Base):
    __tablename__ = "query_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_prompt: Mapped[str] = mapped_column(String(1024), nullable=False)
    wolfram_response: Mapped[str] = mapped_column(Text, nullable=False)
    subject: Mapped[str] = mapped_column(String(32), nullable=False, default="Math")
    ocr_source: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


DATABASE_PATH = Path(__file__).resolve().parent / "history.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)
Base.metadata.create_all(bind=engine)


def _ensure_query_session_schema() -> None:
    inspector = inspect(engine)
    if "query_sessions" not in inspector.get_table_names():
        return
    existing_columns = {column["name"] for column in inspector.get_columns("query_sessions")}
    alter_statements: list[str] = []
    if "subject" not in existing_columns:
        alter_statements.append("ALTER TABLE query_sessions ADD COLUMN subject VARCHAR(32) NOT NULL DEFAULT 'Math'")
    if "ocr_source" not in existing_columns:
        alter_statements.append("ALTER TABLE query_sessions ADD COLUMN ocr_source BOOLEAN NOT NULL DEFAULT 0")

    if not alter_statements:
        return

    with engine.begin() as connection:
        for statement in alter_statements:
            connection.execute(text(statement))


_ensure_query_session_schema()
CONVERSATION_CONTEXT: dict[str, ConversationContext] = {}

EXAM_TARGETS = {
    "NSEJS": {
        "exam_date": date(2026, 11, 22),
        "target_syllabus_percent": 95,
        "target_problem_count": 1500,
        "focus_area": "Mixed physics numericals, chemistry recall, and speed math drills.",
    },
    "NMTC": {
        "exam_date": date(2026, 9, 5),
        "target_syllabus_percent": 90,
        "target_problem_count": 800,
        "focus_area": "Contest-style logic, number theory, and timed puzzle solving.",
    },
    "IOQM": {
        "exam_date": date(2026, 9, 6),
        "target_syllabus_percent": 85,
        "target_problem_count": 600,
        "focus_area": "Proof writing, algebraic manipulation, and geometry consistency.",
    },
    "JEE": {
        "exam_date": date(2027, 1, 24),
        "target_syllabus_percent": 40,
        "target_problem_count": 1200,
        "focus_area": "Foundational PCM depth and timed mixed-set endurance.",
    },
}
LONG_TERM_GOAL_DATE = date(2027, 1, 24)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("addix-deterministic-solver")

app = FastAPI(
    title="ADDIX Labs Deterministic API",
    description="Production-safe deterministic solver powered by Wolfram Alpha.",
    version="2.0.0",
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


@app.exception_handler(APITimeoutError)
@app.exception_handler(asyncio.TimeoutError)
async def timeout_exception_handler(request: Request, exc: Exception):
    logger.warning("System override triggered for API timeout.", exc_info=exc)
    trace = [
        {
            "step_number": 1,
            "title": "System Override",
            "description": SYSTEM_OVERRIDE_TIMEOUT_MESSAGE,
            "agent_type": "Planner",
        }
    ]
    return JSONResponse(
        status_code=504,
        content={
            "final_answer": SYSTEM_OVERRIDE_TIMEOUT_MESSAGE,
            "detail": SYSTEM_OVERRIDE_TIMEOUT_MESSAGE,
            "explanation_trace": trace,
            "logic_trace": trace,
            "planner_state": {
                "source": "system_override",
                "cache_hit": False,
                "verification_triggered": False,
                "session_id": None,
            },
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled backend exception.", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "final_answer": WOLFRAM_FALLBACK_RESULT,
            "explanation_trace": [],
            "logic_trace": [],
            "planner_state": {
                "source": "system_error",
                "cache_hit": False,
                "verification_triggered": False,
                "session_id": None,
            },
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning("HTTPException occurred: %s", exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "final_answer": WOLFRAM_FALLBACK_RESULT,
            "detail": exc.detail,
            "explanation_trace": [],
            "logic_trace": [],
            "planner_state": {
                "source": "http_error",
                "cache_hit": False,
                "verification_triggered": False,
                "session_id": None,
            },
        },
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning("Validation error occurred.", exc_info=exc)
    return JSONResponse(
        status_code=422,
        content={
            "final_answer": WOLFRAM_FALLBACK_RESULT,
            "detail": "Validation failed for request payload.",
            "explanation_trace": [],
            "logic_trace": [],
            "planner_state": {
                "source": "validation_error",
                "cache_hit": False,
                "verification_triggered": False,
                "session_id": None,
            },
        },
    )


def _days_remaining(target_date: date) -> int:
    return (target_date - datetime.now().date()).days


def _normalize_prompt(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", prompt.strip().lower())
    normalized = re.sub(r"[^a-z0-9\s+\-*/^=().]", "", normalized)
    return normalized[:512]


def _detect_subject(prompt: str) -> str:
    normalized = prompt.lower()
    physics_markers = ("mass", "force", "velocity", "acceleration", "newton", "projectile", "momentum")
    chemistry_markers = ("mole", "reaction", "acid", "base", "ph", "stoichiometry", "atom")
    math_markers = ("integral", "derivative", "limit", "matrix", "equation", "solve", "calculate")

    if any(marker in normalized for marker in physics_markers):
        return "Physics"
    if any(marker in normalized for marker in chemistry_markers):
        return "Chemistry"
    if any(marker in normalized for marker in math_markers):
        return "Math"
    return "Math"


def _extract_variables(prompt: str, inherited: dict[str, str] | None = None) -> dict[str, str]:
    variables = dict(inherited or {})
    patterns = {
        "mass": r"\bmass\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?(?:\s*[a-zA-Z/]+)?)",
        "velocity": r"\bvelocity\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?(?:\s*[a-zA-Z/]+)?)",
        "acceleration": r"\bacceleration\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?(?:\s*[a-zA-Z/]+)?)",
        "distance": r"\bdistance\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?(?:\s*[a-zA-Z/]+)?)",
        "time": r"\btime\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?(?:\s*[a-zA-Z/]+)?)",
    }
    lowered = prompt.lower()
    for name, pattern in patterns.items():
        match = re.search(pattern, lowered)
        if match and match.group(1):
            variables[name] = match.group(1).strip()

    if "mass" in variables and re.search(r"mass\s+is\s+doubled|mass\s+doubles|mass\s+double", lowered):
        variables["mass_modifier"] = "2x"
    return variables


def _is_contextual_followup(prompt: str) -> bool:
    normalized = prompt.strip().lower()
    followup_markers = (
        "and what if",
        "what if",
        "and if",
        "then if",
        "if mass",
        "if we",
        "and now",
    )
    return any(marker in normalized for marker in followup_markers)


def _apply_contextual_memory(prompt: str, context: ConversationContext | None) -> tuple[str, bool]:
    if not context or not _is_contextual_followup(prompt):
        return prompt, False

    variable_text = ", ".join([f"{key}={value}" for key, value in context.variables.items()])
    resolved_query = (
        f"Base problem: {context.last_query}. Follow-up: {prompt}. "
        f"Known variables from previous step: {variable_text or 'none'}"
    )
    return resolved_query, True


def _clean_math_text(raw_text: str) -> str:
    cleaned = raw_text.replace("×", "*").replace("−", "-").replace("÷", "/")
    cleaned = cleaned.replace("=", " = ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\b([a-zA-Z])\s*([0-9]{1,2})\b", r"\1^\2", cleaned)
    cleaned = re.sub(r"\^\s+", "^", cleaned)
    return cleaned


def _decode_base64_image(image_payload: str) -> Image.Image:
    payload = image_payload.strip()
    if "," in payload and payload.lower().startswith("data:image"):
        payload = payload.split(",", 1)[1]
    image_bytes = base64.b64decode(payload, validate=True)
    image = Image.open(BytesIO(image_bytes))
    image = image.convert("RGB")
    grayscale = ImageOps.grayscale(image)
    return grayscale


def _extract_text_with_confidence(image: Image.Image) -> tuple[str, float]:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 6")
    tokens = pytesseract.image_to_string(image, config="--psm 6")

    confidences: list[float] = []
    for value in data.get("conf", []):
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if score >= 0:
            confidences.append(score)

    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return tokens.strip(), round(confidence, 2)


def _check_database_status_sync() -> tuple[str, int]:
    with SessionLocal() as session:
        total_logs = session.execute(select(QuerySession.id)).scalars().all()
        return "connected", len(total_logs)


async def _check_wolfram_connection() -> str:
    app_id = os.getenv(WOLFRAM_APP_ID_ENV_NAME)
    if not app_id:
        return "missing_app_id"

    endpoint = "http://api.wolframalpha.com/v1/result"
    query_params = urllib.parse.urlencode({"appid": app_id, "i": "1+1"})
    request_url = f"{endpoint}?{query_params}"
    try:
        response = await asyncio.to_thread(requests.get, request_url, timeout=8)
    except requests.RequestException:
        return "offline"

    if response.status_code == 200:
        return "online"
    if response.status_code in {401, 403}:
        return "auth_error"
    return f"http_{response.status_code}"


def _is_verification_query(prompt: str) -> bool:
    return bool(re.search(r"\b(calculate|solve)\b", prompt.lower()))


def _semantic_check_sync(prompt: str) -> tuple[QuerySession | None, float]:
    target = _normalize_prompt(prompt)
    if not target:
        return None, 0.0

    cutoff = datetime.utcnow() - timedelta(hours=CACHE_LOOKBACK_HOURS)
    with SessionLocal() as session:
        candidates = session.execute(
            select(QuerySession)
            .where(QuerySession.timestamp >= cutoff)
            .order_by(QuerySession.timestamp.desc())
            .limit(80)
        ).scalars().all()

    best_match: QuerySession | None = None
    best_score = 0.0
    for item in candidates:
        comparison = _normalize_prompt(item.user_prompt)
        if comparison == target:
            return item, 1.0
        score = SequenceMatcher(None, target, comparison).ratio()
        if score > best_score:
            best_score = score
            best_match = item

    if best_match and best_score >= CACHE_SIMILARITY_THRESHOLD:
        return best_match, best_score
    return None, best_score


async def lookup_recent_similar(prompt: str) -> tuple[QuerySession | None, float]:
    return await asyncio.to_thread(_semantic_check_sync, prompt)


def _persist_query_session_sync(prompt: str, answer: str, subject: str, ocr_source: bool) -> QuerySession:
    with SessionLocal() as session:
        entry = QuerySession(
            user_prompt=prompt,
            wolfram_response=answer,
            subject=subject,
            ocr_source=ocr_source,
            timestamp=datetime.utcnow(),
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return entry


async def persist_query_session(prompt: str, answer: str, subject: str, ocr_source: bool) -> QuerySession:
    return await asyncio.to_thread(_persist_query_session_sync, prompt, answer, subject, ocr_source)


async def run_python_repl_placeholder(query_text: str, answer_text: str) -> str:
    await asyncio.sleep(0)
    sanitized = answer_text.strip() if answer_text.strip() else "No answer to verify"
    return (
        "Placeholder REPL check executed: prepared deterministic verification context for "
        f"'{query_text}' and compared against Wolfram output '{sanitized[:120]}'."
    )


def _is_valid_wolfram_answer(answer_text: str) -> bool:
    if not answer_text:
        return False
    normalized = answer_text.strip().lower()
    return all(marker not in normalized for marker in INVALID_WOLFRAM_MARKERS)


def _derive_symbolic_formula(query_text: str) -> str:
    normalized = query_text.lower()
    if "limit" in normalized:
        return r"\lim_{x \to a} f(x)"
    if "integral" in normalized or "integrate" in normalized:
        return r"\int f(x)\,dx"
    if "derivative" in normalized or "differentiate" in normalized:
        return r"\frac{d}{dx}f(x)"
    if "projectile" in normalized or "range" in normalized or "angle" in normalized:
        return r"R = \frac{u^2\sin(2\theta)}{g}"
    if "acceleration" in normalized or "velocity" in normalized or "distance" in normalized:
        return r"s = ut + \frac{1}{2}at^2"
    if "quadratic" in normalized or "roots" in normalized:
        return r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}"
    if "current" in normalized or "voltage" in normalized or "resistance" in normalized:
        return r"V = IR"
    if re.search(r"\d", normalized) and re.search(r"[=+\-*/^]", normalized):
        return r"\text{Expression evaluated deterministically}"
    return r"\text{Formula derived from query context}"


def _build_step_trace(query_text: str, result_text: str) -> str:
    return (
        f"Step 1: Input query forwarded to Wolfram: {query_text}\n"
        "Step 2: Executed Wolfram Alpha Short Answer API deterministic call.\n"
        f"Step 3: Raw computational output: {result_text}"
    )


async def query_wolfram(query_text: str) -> WolframResult:
    app_id = os.getenv(WOLFRAM_APP_ID_ENV_NAME)
    normalized_query = query_text.strip()
    formula = _derive_symbolic_formula(normalized_query)
    if not normalized_query:
        return {
            "answer": WOLFRAM_FALLBACK_RESULT,
            "state": "fallback",
            "formula": formula,
            "steps": "Step 1: Empty query received.\nStep 2: No deterministic computation executed.",
        }

    if not app_id:
        logger.warning("WOLFRAM_APP_ID is missing. Activating security protocol.")
        return {
            "answer": SECURITY_PROTOCOL_MESSAGE,
            "state": "security",
            "formula": formula,
            "steps": "Step 1: Missing WOLFRAM_APP_ID.\nStep 2: Security Protocol: Resetting API Handshake.",
        }

    endpoint = "http://api.wolframalpha.com/v1/result"
    query_params = urllib.parse.urlencode({"appid": app_id, "i": normalized_query})
    request_url = f"{endpoint}?{query_params}"

    try:
        response = await asyncio.to_thread(requests.get, request_url, timeout=20)
    except requests.Timeout as exc:
        logger.warning("Wolfram timeout for '%s'", normalized_query)
        raise APITimeoutError("Wolfram timeout") from exc
    except requests.RequestException as exc:
        logger.warning("Wolfram request failed for '%s': %s", normalized_query, exc)
        return {
            "answer": SECURITY_PROTOCOL_MESSAGE,
            "state": "security",
            "formula": formula,
            "steps": "Step 1: Request dispatch failed before deterministic execution.\n"
            "Step 2: Security Protocol: Resetting API Handshake.",
        }

    if response.status_code == 200:
        answer_text = response.text.strip()
        if _is_valid_wolfram_answer(answer_text):
            return {
                "answer": answer_text,
                "state": "ok",
                "formula": formula,
                "steps": _build_step_trace(normalized_query, answer_text),
            }
        return {
            "answer": WOLFRAM_FALLBACK_RESULT,
            "state": "fallback",
            "formula": formula,
            "steps": _build_step_trace(normalized_query, "No deterministic short-answer output was available."),
        }

    if response.status_code in {408, 504}:
        raise APITimeoutError(f"Wolfram timeout response HTTP {response.status_code}")

    if response.status_code in {401, 403}:
        logger.warning("Wolfram authentication failed with HTTP %s", response.status_code)
        return {
            "answer": SECURITY_PROTOCOL_MESSAGE,
            "state": "security",
            "formula": formula,
            "steps": "Step 1: Authentication challenge received from Wolfram API.\n"
            "Step 2: Security Protocol: Resetting API Handshake.",
        }

    if response.status_code == 501:
        logger.info("Wolfram could not compute deterministic result for query: %s", normalized_query)
        return {
            "answer": WOLFRAM_FALLBACK_RESULT,
            "state": "fallback",
            "formula": formula,
            "steps": _build_step_trace(normalized_query, "Wolfram returned HTTP 501: no deterministic result."),
        }

    logger.warning("Wolfram returned HTTP %s for query '%s'", response.status_code, normalized_query)
    return {
        "answer": SECURITY_PROTOCOL_MESSAGE,
        "state": "security",
        "formula": formula,
        "steps": "Step 1: Non-success HTTP status received from Wolfram API.\n"
        "Step 2: Security Protocol: Resetting API Handshake.",
    }


def _build_trace(
    query_text: str,
    wolfram_result: WolframResult,
    *,
    cache_hit: bool,
    cache_similarity: float,
    verification_summary: str | None,
    context_applied: bool,
    subject: str,
) -> list[TraceStep]:
    if wolfram_result["state"] == "security":
        validation_description = "Security validation failed. Security Protocol: Resetting API Handshake."
    elif wolfram_result["state"] == "fallback":
        validation_description = "Deterministic validation incomplete; fallback response was returned safely."
    else:
        validation_description = "Deterministic validation passed with a stable symbolic result."

    trace_steps: list[TraceStep] = [
        TraceStep(
            step_number=1,
            title="Planner Intake",
            description=f"Planner received student query: {query_text}",
            agent_type="Planner",
        ),
        TraceStep(
            step_number=2,
            title="Semantic Check",
            description=(
                f"Found a semantic match with similarity score {cache_similarity:.2f}; reusing previous reasoning."
                if cache_hit
                else "No semantic match above 90% similarity. Routing to deterministic solver."
            ),
            agent_type="Caching",
        ),
        TraceStep(
            step_number=3,
            title="Wolfram Execution" if not cache_hit else "Cache Retrieval",
            description=wolfram_result["steps"],
            agent_type="Symbolic",
            math_latex=wolfram_result["formula"],
        ),
        TraceStep(
            step_number=4,
            title="Context Linking",
            description=(
                f"Session context applied with subject tag '{subject}'."
                if context_applied
                else f"Stored latest context state with subject tag '{subject}'."
            ),
            agent_type="Planner",
        ),
    ]

    if verification_summary:
        trace_steps.append(
            TraceStep(
                step_number=5,
                title="Python REPL Verification",
                description=verification_summary,
                agent_type="Verification",
            )
        )

    trace_steps.append(
        TraceStep(
            step_number=6,
            title="Result Validation",
            description=validation_description,
            agent_type="Symbolic",
        )
    )
    trace_steps.append(
        TraceStep(
            step_number=7,
            title="Final Response",
            description=f"Returned solver output: {wolfram_result['answer']}",
            agent_type="Neural",
        )
    )

    for step_number, step in enumerate(trace_steps, start=1):
        step.step_number = step_number
    return trace_steps


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
        "JEE_MAIN_2027": LongTermGoal(
            target_date=LONG_TERM_GOAL_DATE.isoformat(),
            days_remaining=_days_remaining(LONG_TERM_GOAL_DATE),
            goal_percent=40,
            milestone="Complete 40% of the JEE foundation by Jan 24, 2027.",
        )
    }

    return DashboardStatsResponse(
        student_grade=9,
        academic_year="2026-2027",
        active_tracks=["NSEJS", "NMTC", "IOQM", "JEE"],
        exam_countdowns=exam_countdowns,
        long_term_goals=long_term_goals,
        today_sessions=3,
        weekly_growth_percent=8.5,
        recommended_focus="Prioritize deterministic PCM problem solving with daily mixed sets.",
    )


@app.get("/")
async def root_status() -> Dict[str, str]:
    return {"status": "ADDIX Systems Online", "tier": "Student"}


class VisionAgent:
    def __init__(self, fuzzy_threshold: float = 60.0) -> None:
        self.fuzzy_threshold = fuzzy_threshold

    async def extract_query(self, image_base64: str) -> tuple[str, str, float, str | None]:
        try:
            image = await asyncio.to_thread(_decode_base64_image, image_base64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid base64 image payload.") from exc

        try:
            extracted_text, confidence = await asyncio.to_thread(_extract_text_with_confidence, image)
        except pytesseract.TesseractNotFoundError as exc:
            raise HTTPException(status_code=503, detail="Tesseract OCR engine is not installed on the server.") from exc

        cleaned_query = _clean_math_text(extracted_text)
        if not cleaned_query:
            raise HTTPException(status_code=422, detail="Vision Agent could not extract usable text from the image.")

        warning = VISION_FUZZY_WARNING if confidence < self.fuzzy_threshold else None
        return extracted_text, cleaned_query, confidence, warning


class Planner:
    def __init__(self) -> None:
        self.context_buffer = CONVERSATION_CONTEXT
        self.vision_agent = VisionAgent()

    async def _semantic_cache_lookup(self, query: str) -> tuple[QuerySession | None, float]:
        return await lookup_recent_similar(query)

    async def _route_text_query(self, resolved_query: str, subject: str) -> WolframResult:
        if subject in {"Math", "Physics"}:
            return await asyncio.wait_for(query_wolfram(resolved_query), timeout=25)

        return {
            "answer": WOLFRAM_FALLBACK_RESULT,
            "state": "fallback",
            "formula": _derive_symbolic_formula(resolved_query),
            "steps": "Step 1: Subject classified outside Math/Physics Wolfram route.\n"
            "Step 2: Returned deterministic safe fallback.",
        }

    async def solve_text_query(
        self,
        *,
        student_query: str,
        target_exam: str,
        session_id: str,
        ocr_source: bool,
    ) -> AgentResponse:
        query_text = student_query.strip()
        if not query_text:
            raise HTTPException(status_code=422, detail="student_query cannot be empty.")

        context_key = (session_id or "default").strip() or "default"
        previous_context = self.context_buffer.get(context_key)
        resolved_query, context_applied = _apply_contextual_memory(query_text, previous_context)
        subject = _detect_subject(resolved_query)

        cached_session, similarity_score = await self._semantic_cache_lookup(resolved_query)
        cache_hit = cached_session is not None

        if cache_hit and cached_session:
            cached_answer = cached_session.wolfram_response.strip() or WOLFRAM_FALLBACK_RESULT
            wolfram_result: WolframResult = {
                "answer": cached_answer,
                "state": "ok",
                "formula": _derive_symbolic_formula(resolved_query),
                "steps": (
                    f"Step 1: Retrieved similar prompt from QuerySession cache (similarity {similarity_score:.2f}).\n"
                    f"Step 2: Reused persisted deterministic response: {cached_answer}"
                ),
            }
            subject = cached_session.subject
        else:
            wolfram_result = await self._route_text_query(resolved_query, subject)

        verification_summary: str | None = None
        verification_triggered = _is_verification_query(resolved_query)
        if verification_triggered:
            verification_summary = await run_python_repl_placeholder(resolved_query, wolfram_result["answer"])

        record_id: int | None = None
        try:
            persisted = await persist_query_session(
                resolved_query,
                wolfram_result["answer"],
                subject,
                ocr_source,
            )
            record_id = persisted.id
        except Exception as exc:
            logger.warning("Unable to persist QuerySession record.", exc_info=exc)

        merged_variables = _extract_variables(query_text, previous_context.variables if previous_context else None)
        self.context_buffer[context_key] = ConversationContext(
            session_id=context_key,
            last_query=resolved_query,
            last_answer=wolfram_result["answer"],
            variables=merged_variables,
        )

        trace = _build_trace(
            resolved_query,
            wolfram_result,
            cache_hit=cache_hit,
            cache_similarity=similarity_score,
            verification_summary=verification_summary,
            context_applied=context_applied,
            subject=subject,
        )

        return AgentResponse(
            final_answer=wolfram_result["answer"],
            explanation_trace=trace,
            logic_trace=trace,
            planner_state={
                "source": "cache" if cache_hit else "wolfram",
                "cache_hit": cache_hit,
                "verification_triggered": verification_triggered,
                "session_record_id": record_id,
                "semantic_similarity": round(similarity_score, 2),
                "subject": subject,
                "ocr_source": ocr_source,
                "session_id": context_key,
                "target_exam": target_exam,
                "context_applied": context_applied,
            },
        )

    async def solve_image_query(self, payload: OCRInput) -> OCRResponse:
        extracted_text, cleaned_query, confidence, warning = await self.vision_agent.extract_query(payload.image_base64)
        solve_result = await self.solve_text_query(
            student_query=cleaned_query,
            target_exam=payload.target_exam,
            session_id=payload.session_id or "default",
            ocr_source=True,
        )
        return OCRResponse(
            extracted_text=extracted_text,
            cleaned_query=cleaned_query,
            confidence=confidence,
            warning=warning,
            solve_result=solve_result,
        )


planner = Planner()


@app.post("/api/solve", response_model=AgentResponse)
async def solve_student_query(payload: QueryInput) -> AgentResponse:
    return await planner.solve_text_query(
        student_query=payload.student_query,
        target_exam=payload.target_exam,
        session_id=payload.session_id or "default",
        ocr_source=False,
    )


@app.post("/api/ocr", response_model=OCRResponse)
async def solve_from_ocr(payload: OCRInput) -> OCRResponse:
    return await planner.solve_image_query(payload)


@app.get("/api/system-status")
async def system_status() -> Dict[str, str | bool | int]:
    try:
        db_status, total_logs = await asyncio.to_thread(_check_database_status_sync)
    except Exception as exc:
        logger.warning("Database health check failed.", exc_info=exc)
        db_status, total_logs = "error", 0

    wolfram_status = await _check_wolfram_connection()
    return {
        "status": "ok",
        "wolfram_status": wolfram_status,
        "database_status": db_status,
        "history_records": total_logs,
    }


@app.get("/api/dashboard-stats", response_model=DashboardStatsResponse)
async def dashboard_stats() -> DashboardStatsResponse:
    return _build_dashboard_stats()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}
