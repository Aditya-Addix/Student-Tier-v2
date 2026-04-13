from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    step_number: int = Field(..., ge=1, description="1-based position in the reasoning workflow.")
    title: str = Field(..., min_length=1, description="Frontend-ready title for this reasoning step.")
    description: str = Field(..., min_length=1, description="Human-readable summary of what happened in this step.")
    agent_type: Literal["Neural", "Symbolic", "Planner", "Caching", "Verification", "Vision"] = Field(
        ...,
        description="Type of reasoning engine responsible for this step.",
    )
    math_latex: Optional[str] = Field(
        default=None,
        description="Optional LaTeX or raw math expression associated with the step.",
    )


class AgentResponse(BaseModel):
    final_answer: str = Field(
        ...,
        min_length=1,
        description="Markdown-ready final answer for the frontend renderer.",
    )
    explanation_trace: List[TraceStep] = Field(
        ...,
        min_length=1,
        description="Primary trace steps used by the frontend terminal UI.",
    )
    logic_trace: List[TraceStep] = Field(
        ...,
        min_length=1,
        description="Ordered planner steps that can be replayed one-by-one in the frontend.",
    )
    planner_state: Dict[str, str | bool | int | float | None] = Field(
        ...,
        description="State metadata for cache status, verification branch, and persisted session linkage.",
    )


class FinalAnswerResponse(BaseModel):
    final_answer: str = Field(
        ...,
        min_length=1,
        description="Deterministic final answer returned by the low-latency solve endpoint.",
    )


class QueryInput(BaseModel):
    student_query: str = Field(..., min_length=1, description="Student's full natural-language question.")
    target_exam: str = Field(..., min_length=1, description="Exam context to tune the explanation style.")
    session_id: Optional[str] = Field(
        default="default",
        min_length=1,
        description="Conversation session identifier for contextual follow-up reasoning.",
    )


class OCRInput(BaseModel):
    image_base64: str = Field(..., min_length=20, description="Base64-encoded image payload.")
    target_exam: str = Field(default="NSEJS", min_length=1, description="Exam context for planner routing.")
    session_id: Optional[str] = Field(
        default="default",
        min_length=1,
        description="Conversation session identifier for contextual follow-up reasoning.",
    )


class ConversationContext(BaseModel):
    session_id: str = Field(..., min_length=1, description="Session key associated with this context state.")
    last_query: str = Field(..., min_length=1, description="Most recent resolved query text used for reasoning.")
    last_answer: str = Field(..., min_length=1, description="Most recent final answer emitted to the user.")
    variables: Dict[str, str] = Field(
        default_factory=dict,
        description="Extracted symbolic variables retained for follow-up prompts.",
    )


class OCRResponse(BaseModel):
    extracted_text: str = Field(..., min_length=1, description="Raw OCR text extracted from the image.")
    cleaned_query: str = Field(..., min_length=1, description="Math-normalized query string for planner execution.")
    confidence: float = Field(..., ge=0, le=100, description="Average OCR confidence score.")
    warning: Optional[str] = Field(default=None, description="Low-confidence warning from Vision Agent.")
    solve_result: AgentResponse = Field(..., description="Planner response auto-generated from OCR query.")


class ExamCountdown(BaseModel):
    exam_date: str = Field(..., description="Exam date in ISO format (YYYY-MM-DD).")
    days_remaining: int = Field(..., description="Days remaining from today until the exam date.")
    target_syllabus_percent: int = Field(..., ge=0, le=100, description="Target syllabus completion percentage.")
    target_problem_count: int = Field(..., ge=0, description="Target number of problems to solve before the exam.")
    focus_area: str = Field(..., min_length=1, description="Suggested focus area for the current prep window.")


class LongTermGoal(BaseModel):
    target_date: str = Field(..., description="Milestone date in ISO format (YYYY-MM-DD).")
    days_remaining: int = Field(..., description="Days remaining from today until the milestone date.")
    goal_percent: int = Field(..., ge=0, le=100, description="Target progress percentage for the long-term milestone.")
    milestone: str = Field(..., min_length=1, description="Readable milestone label for the frontend.")


class DashboardStatsResponse(BaseModel):
    student_grade: int = Field(..., ge=1, description="Current student grade level.")
    academic_year: str = Field(..., min_length=1, description="Academic context for the student dashboard.")
    active_tracks: List[str] = Field(..., min_length=1, description="Active olympiad and long-term goal tracks.")
    exam_countdowns: Dict[str, ExamCountdown] = Field(
        ...,
        description="Dynamic countdowns and targets for near-term exams.",
    )
    long_term_goals: Dict[str, LongTermGoal] = Field(
        ...,
        description="Long-range preparation milestones such as JEE/NEET.",
    )
    today_sessions: int = Field(..., ge=0, description="Recommended focused study sessions for today.")
    weekly_growth_percent: float = Field(..., description="Target week-over-week learning growth percentage.")
    recommended_focus: str = Field(..., min_length=1, description="Primary study recommendation for the UI.")
