from __future__ import annotations

import json
import sys

from online_softmax_common import cases, check_stream, load_submission


def main(argv: list[str]) -> int:
    results = []
    status = "passed"
    try:
        if len(argv) != 3 or argv[1] != "--mode" or argv[2] not in {"public", "full"}:
            raise ValueError("expected --mode public or --mode full")
        update = load_submission()
        for name, history, chunks in cases(argv[2] == "full"):
            try:
                check_stream(update, history, chunks)
                results.append({"name": name, "passed": True})
            except AssertionError as error:
                status = "wrong_answer"
                results.append({"name": name, "passed": False, "message": str(error)})
    except Exception as error:
        status = "compile_error" if isinstance(error, SyntaxError) else "runtime_error"
        results.append({"name": "execution", "passed": False, "message": str(error)[:300]})
    passed = sum(case["passed"] for case in results)
    print(
        "MYLEETGPU_RESULT="
        + json.dumps(
            {
                "status": status,
                "cases": results,
                "summary": {
                    "total": len(results),
                    "passed": passed,
                    "failed": len(results) - passed,
                },
            },
            separators=(",", ":"),
        )
    )
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
