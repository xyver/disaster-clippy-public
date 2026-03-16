"""
Minimal QA harness for Disaster Clippy API regression checks.

Runs:
- API contract checks
- Retrieval-style smoke checks against representative questions
- Basic session continuity checks

This is intentionally lightweight and dependency-free so it can run
against local or deployed environments with standard Python only.
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request


ROOT = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = ROOT / "datasets" / "questions_v1.json"
DEFAULT_CONTRACT_CASES = ROOT / "datasets" / "api_contract_cases_v1.json"
DEFAULT_REPORTS_DIR = ROOT / "reports"


@dataclass
class CheckResult:
    id: str
    status: str
    message: str
    duration_ms: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def http_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
    verify_ssl: bool = True,
) -> Tuple[int, Dict[str, str], Any, str]:
    body_bytes = None
    headers = {
        "Accept": "application/json",
    }
    if payload is not None:
        body_bytes = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=body_bytes, headers=headers, method=method.upper())
    context = None
    if not verify_ssl and url.startswith("https://"):
        context = ssl._create_unverified_context()

    try:
        with request.urlopen(req, timeout=timeout, context=context) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            content_type = resp.headers.get("Content-Type", "")
            data: Any = text
            if "application/json" in content_type:
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    data = text
            return resp.status, dict(resp.headers.items()), data, text
    except error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return exc.code, dict(exc.headers.items()), text, text


def run_contract_case(
    base_url: str,
    case: Dict[str, Any],
    timeout: int,
    verify_ssl: bool,
) -> CheckResult:
    started = time.time()
    url = build_url(base_url, case["path"])
    status, headers, data, text = http_json(
        case["method"],
        url,
        payload=case.get("payload"),
        timeout=timeout,
        verify_ssl=verify_ssl,
    )
    duration_ms = int((time.time() - started) * 1000)
    failures: List[str] = []

    if status != case["expect_status"]:
        failures.append(f"Expected status {case['expect_status']}, got {status}")

    if "expect_json_keys" in case:
        if not isinstance(data, dict):
            failures.append("Expected JSON object response")
        else:
            for key in case["expect_json_keys"]:
                if key not in data:
                    failures.append(f"Missing JSON key: {key}")

    response_text = ""
    if isinstance(data, dict):
        response_text = str(data.get("response", ""))
    elif isinstance(text, str):
        response_text = text

    if "expect_response_contains_all" in case:
        for needle in case["expect_response_contains_all"]:
            if needle not in response_text:
                failures.append(f"Missing expected text: {needle}")

    if "expect_response_contains_any" in case:
        needles = case["expect_response_contains_any"]
        if not any(needle in response_text for needle in needles):
            failures.append(f"Response missing all expected hints: {needles}")

    if "expect_stream_contains" in case:
        for needle in case["expect_stream_contains"]:
            if needle not in text:
                failures.append(f"Stream missing expected marker: {needle}")

    return CheckResult(
        id=case["id"],
        status="pass" if not failures else "fail",
        message="OK" if not failures else "; ".join(failures),
        duration_ms=duration_ms,
        details={
            "path": case["path"],
            "http_status": status,
            "response_preview": response_text[:280] if response_text else text[:280],
            "headers": headers,
        },
    )


def fetch_sources(base_url: str, timeout: int, verify_ssl: bool) -> Tuple[CheckResult, List[str]]:
    started = time.time()
    status, _, data, text = http_json(
        "GET",
        build_url(base_url, "/api/v1/sources"),
        timeout=timeout,
        verify_ssl=verify_ssl,
    )
    duration_ms = int((time.time() - started) * 1000)

    if status != 200:
        return (
            CheckResult(
                id="sources_list",
                status="fail",
                message=f"Expected 200, got {status}",
                duration_ms=duration_ms,
                details={"response_preview": text[:280]},
            ),
            [],
        )

    if not isinstance(data, dict) or "sources" not in data:
        return (
            CheckResult(
                id="sources_list",
                status="fail",
                message="Invalid /api/v1/sources payload",
                duration_ms=duration_ms,
                details={"response_preview": text[:280]},
            ),
            [],
        )

    source_ids = []
    for item in data.get("sources", []):
        if isinstance(item, dict) and "id" in item:
            source_ids.append(str(item["id"]))

    return (
        CheckResult(
            id="sources_list",
            status="pass",
            message=f"Loaded {len(source_ids)} sources",
            duration_ms=duration_ms,
            details={"source_count": len(source_ids), "total_documents": data.get("total_documents")},
        ),
        source_ids,
    )


def run_retrieval_question(
    base_url: str,
    question_case: Dict[str, Any],
    available_sources: List[str],
    timeout: int,
    verify_ssl: bool,
) -> CheckResult:
    started = time.time()
    payload = {"message": question_case["question"]}
    status, _, data, text = http_json(
        "POST",
        build_url(base_url, "/api/v1/chat"),
        payload=payload,
        timeout=timeout,
        verify_ssl=verify_ssl,
    )
    duration_ms = int((time.time() - started) * 1000)

    failures: List[str] = []
    response_text = ""
    session_id = None

    if status != 200:
        failures.append(f"Expected 200, got {status}")

    if not isinstance(data, dict):
        failures.append("Expected JSON object response")
    else:
        response_text = str(data.get("response", ""))
        session_id = data.get("session_id")
        if not response_text.strip():
            failures.append("Empty response text")
        if not session_id:
            failures.append("Missing session_id")

    lower_response = response_text.lower()
    topic_hits = [
        topic for topic in question_case.get("expected_topics", [])
        if topic.lower() in lower_response
    ]
    if question_case.get("expected_topics") and not topic_hits:
        failures.append("Response did not mention any expected topic hints")

    missing_sources = [
        source_id for source_id in question_case.get("expected_sources", [])
        if source_id not in available_sources
    ]

    return CheckResult(
        id=question_case["id"],
        status="pass" if not failures else "fail",
        message="OK" if not failures else "; ".join(failures),
        duration_ms=duration_ms,
        details={
            "question": question_case["question"],
            "session_id": session_id,
            "topic_hits": topic_hits,
            "expected_sources_missing_from_deployment": missing_sources,
            "response_preview": response_text[:280] if response_text else text[:280],
        },
    )


def run_session_continuity_check(base_url: str, timeout: int, verify_ssl: bool) -> CheckResult:
    session_id = f"qa-session-{int(time.time())}"
    started = time.time()
    first_status, _, first_data, _ = http_json(
        "POST",
        build_url(base_url, "/api/v1/chat"),
        payload={"message": "Answer in one short sentence: what topic are you helping with?", "session_id": session_id},
        timeout=timeout,
        verify_ssl=verify_ssl,
    )
    second_status, _, second_data, second_text = http_json(
        "POST",
        build_url(base_url, "/api/v1/chat"),
        payload={"message": "Now answer in one short sentence: what did I just ask you?", "session_id": session_id},
        timeout=timeout,
        verify_ssl=verify_ssl,
    )
    duration_ms = int((time.time() - started) * 1000)

    failures: List[str] = []
    if first_status != 200 or second_status != 200:
        failures.append(f"Unexpected statuses: first={first_status}, second={second_status}")

    if not isinstance(first_data, dict) or not isinstance(second_data, dict):
        failures.append("Expected JSON objects for both session checks")
    else:
        if first_data.get("session_id") != session_id:
            failures.append("First call did not preserve requested session_id")
        if second_data.get("session_id") != session_id:
            failures.append("Second call did not preserve requested session_id")
        second_response = str(second_data.get("response", "")).lower()
        if "topic" not in second_response and "helping" not in second_response and "asked" not in second_response:
            failures.append("Second response did not look session-aware")

    return CheckResult(
        id="session_continuity",
        status="pass" if not failures else "fail",
        message="OK" if not failures else "; ".join(failures),
        duration_ms=duration_ms,
        details={
            "session_id": session_id,
            "first_response": first_data.get("response", "")[:180] if isinstance(first_data, dict) else "",
            "second_response": second_data.get("response", "")[:180] if isinstance(second_data, dict) else second_text[:180],
        },
    )


def write_reports(report: Dict[str, Any], report_dir: Path, label: str) -> Tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    json_path = report_dir / f"{timestamp}_{label}.json"
    md_path = report_dir / f"{timestamp}_{label}.md"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)

    lines = [
        f"# QA Report: {label}",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Base URL: `{report['base_url']}`",
        f"- Overall status: `{report['summary']['overall_status']}`",
        f"- Passed: {report['summary']['passed']}",
        f"- Failed: {report['summary']['failed']}",
        "",
        "## Checks",
        "",
    ]

    for section_name in ["setup", "contract", "retrieval", "session"]:
        results = report["results"].get(section_name, [])
        if not results:
            continue
        lines.append(f"### {section_name.title()}")
        lines.append("")
        for result in results:
            lines.append(f"- `{result['status']}` `{result['id']}`: {result['message']}")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")

    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Disaster Clippy API QA checks.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL to test")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="Question dataset JSON")
    parser.add_argument("--contract-cases", type=Path, default=DEFAULT_CONTRACT_CASES, help="Contract cases JSON")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR, help="Output report directory")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout in seconds")
    parser.add_argument("--skip-contract", action="store_true", help="Skip API contract cases")
    parser.add_argument("--skip-retrieval", action="store_true", help="Skip representative question checks")
    parser.add_argument("--skip-session", action="store_true", help="Skip session continuity check")
    parser.add_argument("--verify-ssl", action="store_true", help="Verify HTTPS certificates")
    args = parser.parse_args()

    questions = load_json(args.questions)
    contract_cases = load_json(args.contract_cases)

    setup_results: List[CheckResult] = []
    contract_results: List[CheckResult] = []
    retrieval_results: List[CheckResult] = []
    session_results: List[CheckResult] = []

    sources_result, available_sources = fetch_sources(args.base_url, args.timeout, args.verify_ssl)
    setup_results.append(sources_result)

    if not args.skip_contract:
        for case in contract_cases:
            contract_results.append(run_contract_case(args.base_url, case, args.timeout, args.verify_ssl))

    if not args.skip_retrieval:
        for question_case in questions:
            retrieval_results.append(
                run_retrieval_question(
                    args.base_url,
                    question_case,
                    available_sources,
                    args.timeout,
                    args.verify_ssl,
                )
            )

    if not args.skip_session:
        session_results.append(run_session_continuity_check(args.base_url, args.timeout, args.verify_ssl))

    all_results = setup_results + contract_results + retrieval_results + session_results
    passed = sum(1 for result in all_results if result.status == "pass")
    failed = sum(1 for result in all_results if result.status == "fail")
    overall_status = "pass" if failed == 0 else "fail"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "summary": {
            "overall_status": overall_status,
            "passed": passed,
            "failed": failed,
            "total": len(all_results),
        },
        "results": {
            "setup": [asdict(result) for result in setup_results],
            "contract": [asdict(result) for result in contract_results],
            "retrieval": [asdict(result) for result in retrieval_results],
            "session": [asdict(result) for result in session_results],
        },
    }

    json_path, md_path = write_reports(report, args.reports_dir, "api_qa")
    print(f"Overall status: {overall_status}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
