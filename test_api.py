import sys
from datetime import date
from typing import Any

import requests


BASE_URL = "http://127.0.0.1:8000"
SOLVE_URL = f"{BASE_URL}/api/solve"
DASHBOARD_URL = f"{BASE_URL}/api/dashboard-stats"
TIMEOUT_SECONDS = 120
SOLVE_PAYLOAD = {
    "student_query": (
        "A car accelerates uniformly from rest to a speed of 72 km/h in 10 seconds. "
        "What is the distance covered by the car in this time?"
    ),
    "target_exam": "NSEJS",
}
EXPECTED_EXAMS = {
    "NSEJS": {"exam_date": date(2026, 11, 22), "target_syllabus_percent": 95, "target_problem_count": 1500},
    "NMTC": {"exam_date": date(2026, 10, 17), "target_syllabus_percent": 90, "target_problem_count": 800},
    "IOQM": {"exam_date": date(2026, 9, 6), "target_syllabus_percent": 85, "target_problem_count": 600},
}
EXPECTED_LONG_TERM_DATE = date(2028, 1, 15)


def fail(message: str, *, response_text: str | None = None) -> None:
    print("\n" + "=" * 88)
    print("ADDIX LABS QA TEST FAILED")
    print("=" * 88)
    print(message)
    if response_text:
        print("\nServer response:")
        print(response_text)
    sys.exit(1)


def print_trace(explanation_trace: list[dict[str, Any]]) -> None:
    print("\nReasoning Trace")
    print("-" * 88)
    for step in explanation_trace:
        print(
            f"Step {step.get('step_number', '?')}: "
            f"{step.get('title', '<missing title>')} "
            f"[{step.get('agent_type', '<missing agent_type>')}]"
        )


def test_solve_endpoint() -> None:
    print("=" * 88)
    print("Testing POST /api/solve")
    print("=" * 88)

    try:
        response = requests.post(SOLVE_URL, json=SOLVE_PAYLOAD, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        fail(f"Request error while calling /api/solve: {exc}")

    if response.status_code != 200:
        fail(f"Expected HTTP 200 from /api/solve, received {response.status_code}.", response_text=response.text)

    try:
        data = response.json()
    except ValueError as exc:
        fail(f"/api/solve returned invalid JSON: {exc}", response_text=response.text)

    explanation_trace = data.get("explanation_trace")
    final_answer = str(data.get("final_answer", ""))

    if not isinstance(explanation_trace, list):
        fail("/api/solve did not return a valid explanation_trace array.", response_text=response.text)
    if len(explanation_trace) != 5:
        fail(f"/api/solve returned {len(explanation_trace)} trace steps instead of 5.", response_text=response.text)

    symbolic_steps = [
        step for step in explanation_trace if isinstance(step, dict) and step.get("agent_type") == "Symbolic"
    ]
    if not symbolic_steps:
        print_trace(explanation_trace)
        fail("/api/solve did not record a Symbolic trace step.", response_text=response.text)

    lowered_answer = final_answer.lower()
    if "100" not in lowered_answer and "100 meters" not in lowered_answer:
        print_trace(explanation_trace)
        fail("The final_answer did not contain the expected result '100' or '100 meters'.", response_text=final_answer)

    print("PASS: /api/solve returned HTTP 200")
    print("PASS: explanation_trace contains exactly 5 steps")
    print("PASS: explanation_trace includes a Symbolic step")
    print("PASS: final_answer contains the expected distance of 100 meters")
    print_trace(explanation_trace)
    print("\nReturned final_answer:")
    print(final_answer)


def test_dashboard_endpoint() -> None:
    print("\n" + "=" * 88)
    print("Testing GET /api/dashboard-stats")
    print("=" * 88)

    try:
        response = requests.get(DASHBOARD_URL, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        fail(f"Request error while calling /api/dashboard-stats: {exc}")

    if response.status_code != 200:
        fail(
            f"Expected HTTP 200 from /api/dashboard-stats, received {response.status_code}.",
            response_text=response.text,
        )

    try:
        data = response.json()
    except ValueError as exc:
        fail(f"/api/dashboard-stats returned invalid JSON: {exc}", response_text=response.text)

    exam_countdowns = data.get("exam_countdowns")
    long_term_goals = data.get("long_term_goals")
    if not isinstance(exam_countdowns, dict):
        fail("/api/dashboard-stats did not return a valid exam_countdowns object.", response_text=response.text)
    if not isinstance(long_term_goals, dict):
        fail("/api/dashboard-stats did not return a valid long_term_goals object.", response_text=response.text)

    today = date.today()
    for exam_name, expected in EXPECTED_EXAMS.items():
        exam_data = exam_countdowns.get(exam_name)
        if not isinstance(exam_data, dict):
            fail(f"Missing countdown data for {exam_name}.", response_text=response.text)

        expected_days = (expected["exam_date"] - today).days
        if exam_data.get("exam_date") != expected["exam_date"].isoformat():
            fail(f"{exam_name} exam_date mismatch.", response_text=response.text)
        if exam_data.get("days_remaining") != expected_days:
            fail(f"{exam_name} days_remaining mismatch.", response_text=response.text)
        if exam_data.get("target_syllabus_percent") != expected["target_syllabus_percent"]:
            fail(f"{exam_name} target_syllabus_percent mismatch.", response_text=response.text)
        if exam_data.get("target_problem_count") != expected["target_problem_count"]:
            fail(f"{exam_name} target_problem_count mismatch.", response_text=response.text)

    jee_neet_goal = long_term_goals.get("JEE_MAIN_2028")
    if not isinstance(jee_neet_goal, dict):
        fail("Missing JEE_MAIN_2028 long-term goal.", response_text=response.text)
    expected_long_term_days = (EXPECTED_LONG_TERM_DATE - today).days
    if jee_neet_goal.get("target_date") != EXPECTED_LONG_TERM_DATE.isoformat():
        fail("JEE_MAIN_2028 target_date mismatch.", response_text=response.text)
    if jee_neet_goal.get("days_remaining") != expected_long_term_days:
        fail("JEE_MAIN_2028 days_remaining mismatch.", response_text=response.text)
    if jee_neet_goal.get("goal_percent") != 10:
        fail("JEE_MAIN_2028 goal_percent mismatch.", response_text=response.text)

    print("PASS: /api/dashboard-stats returned HTTP 200")
    print("PASS: exam countdown dates and dynamic day counts match expectations")
    print("PASS: syllabus and problem targets match the 2026 prep plan")
    print("PASS: JEE Main 2028 long-term goal is present")
    print("\nDashboard Countdown Summary")
    print("-" * 88)
    for exam_name, exam_data in exam_countdowns.items():
        print(
            f"{exam_name}: {exam_data.get('days_remaining')} days remaining | "
            f"Target {exam_data.get('target_syllabus_percent')}% syllabus | "
            f"{exam_data.get('target_problem_count')} problems"
        )
    print(
        f"JEE Main 2028: {jee_neet_goal.get('days_remaining')} days remaining "
        f"(target date {jee_neet_goal.get('target_date')})"
    )


def main() -> None:
    test_solve_endpoint()
    test_dashboard_endpoint()
    print("\n" + "=" * 88)
    print("ADDIX LABS QA TEST SUITE PASSED")
    print("=" * 88)


if __name__ == "__main__":
    main()
