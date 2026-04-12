from __future__ import annotations

import asyncio
import logging
import os
import urllib.parse
from datetime import date, datetime
from typing import Dict

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from models import (
    AgentResponse,
    DashboardStatsResponse,
    ExamCountdown,
    LongTermGoal,
    QueryInput,
    TraceStep,
)


load_dotenv()

WOLFRAM_APP_ID_ENV_NAME = "WOLFRAM_APP_ID"
WOLFRAM_FALLBACK_RESULT = "Symbolic Solver: No deterministic result available for this specific query."
INVALID_WOLFRAM_MARKERS = (
    "no short answer available",
    "did not understand your input",
    "unable to interpret",
    "error",
)

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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled backend exception.", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "final_answer": WOLFRAM_FALLBACK_RESULT,
            "explanation_trace": [],
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
        },
    )


def _days_remaining(target_date: date) -> int:
    return (target_date - datetime.now().date()).days


def _is_valid_wolfram_answer(answer_text: str) -> bool:
    if not answer_text:
        return False
    normalized = answer_text.strip().lower()
    return all(marker not in normalized for marker in INVALID_WOLFRAM_MARKERS)


async def query_wolfram(query_text: str) -> str:
    app_id = os.getenv(WOLFRAM_APP_ID_ENV_NAME)
    normalized_query = query_text.strip()
    if not app_id or not normalized_query:
        return WOLFRAM_FALLBACK_RESULT

    endpoint = "http://api.wolframalpha.com/v1/result"
    query_params = urllib.parse.urlencode({"appid": app_id, "i": normalized_query})
    request_url = f"{endpoint}?{query_params}"

    try:
        response = await asyncio.to_thread(requests.get, request_url, timeout=20)
    except requests.RequestException as exc:
        logger.warning("Wolfram request failed for '%s': %s", normalized_query, exc)
        return WOLFRAM_FALLBACK_RESULT

    if response.status_code == 200:
        answer_text = response.text.strip()
        return answer_text if _is_valid_wolfram_answer(answer_text) else WOLFRAM_FALLBACK_RESULT

    if response.status_code == 501:
        logger.info("Wolfram could not compute deterministic result for query: %s", normalized_query)
    else:
        logger.warning(
            "Wolfram returned HTTP %s for query '%s': %s",
            response.status_code,
            normalized_query,
            response.text.strip(),
        )
    return WOLFRAM_FALLBACK_RESULT


def _build_trace(query_text: str, deterministic_result: str) -> list[TraceStep]:
    return [
        TraceStep(
            step_number=1,
            title="Problem Intake",
            description=f"Received student query: {query_text}",
            agent_type="Neural",
        ),
        TraceStep(
            step_number=2,
            title="Deterministic Routing",
            description="Routed request directly to Wolfram Alpha Short Answer API.",
            agent_type="Symbolic",
        ),
        TraceStep(
            step_number=3,
            title="Wolfram Execution",
            description=f"Executed deterministic query: {query_text}",
            agent_type="Symbolic",
            math_latex=query_text,
        ),
        TraceStep(
            step_number=4,
            title="Result Validation",
            description="Checked deterministic response status and fallback conditions.",
            agent_type="Symbolic",
        ),
        TraceStep(
            step_number=5,
            title="Final Response",
            description=f"Returned solver output: {deterministic_result}",
            agent_type="Neural",
        ),
    ]


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
def root_status() -> Dict[str, str]:
    return {"status": "ADDIX Systems Online", "tier": "Student"}


@app.post("/api/solve", response_model=AgentResponse)
async def solve_student_query(payload: QueryInput) -> AgentResponse:
    student_query = payload.student_query.strip()
    if not student_query:
        raise HTTPException(status_code=422, detail="student_query cannot be empty.")

    deterministic_result = await query_wolfram(student_query)
    trace = _build_trace(student_query, deterministic_result)
    return AgentResponse(final_answer=deterministic_result, explanation_trace=trace)


@app.get("/api/dashboard-stats", response_model=DashboardStatsResponse)
def dashboard_stats() -> DashboardStatsResponse:
    return _build_dashboard_stats()


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}
