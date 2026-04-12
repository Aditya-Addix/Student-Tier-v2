from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    step_number: int = Field(..., ge=1, le=5, description="1-based position in the reasoning workflow.")
    title: str = Field(..., min_length=1, description="Frontend-ready title for this reasoning step.")
    description: str = Field(..., min_length=1, description="Human-readable summary of what happened in this step.")
    agent_type: Literal["Neural", "Symbolic"] = Field(
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
        min_length=5,
        max_length=5,
        description="Exactly five reasoning steps used by the frontend trace UI.",
    )


class QueryInput(BaseModel):
    student_query: str = Field(..., min_length=1, description="Student's full natural-language question.")
    target_exam: str = Field(..., min_length=1, description="Exam context to tune the explanation style.")


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
