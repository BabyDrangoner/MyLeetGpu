"""Pure judge-result policy, independent of scheduling, storage, and execution adapters."""

from __future__ import annotations

from typing import Any

from myleetgpu.domain.benchmark import Measurement
from myleetgpu.domain.jobs import ErrorCode, JobError
from myleetgpu.domain.problems import KernelLanguage, Problem
from myleetgpu.runner.models import CompileResult, ExecutionResult


class JobFailed(RuntimeError):
    def __init__(self, error: JobError, diagnostics: str | None = None):
        super().__init__(error.message)
        self.error = error
        self.diagnostics = diagnostics


def require_compile(result: CompileResult, language: str = "cuda_cpp") -> None:
    if result.succeeded:
        return
    toolchain = {
        KernelLanguage.CUDA_CPP.value: "NVCC 编译",
        KernelLanguage.TRITON_PYTHON.value: "Triton/Python 提交策略预检",
        KernelLanguage.TORCH_PYTHON.value: "PyTorch/Python 提交策略预检",
        KernelLanguage.CPP.value: "C++ 编译",
        KernelLanguage.PYTHON.value: "Python 语法检查",
    }.get(language, "提交预检")
    if result.timed_out:
        code, message = ErrorCode.TIMEOUT, f"{toolchain}超时"
    elif result.output_limited:
        code, message = ErrorCode.OUTPUT_LIMIT, f"{toolchain}诊断输出超过限制"
    else:
        code, message = ErrorCode.COMPILE_ERROR, f"{toolchain}失败"
    raise JobFailed(JobError(code=code, message=message, stage="compiling"), result.diagnostics)


def require_execution(
    result: ExecutionResult,
    stage: str,
    safe_correctness: dict[str, Any] | None = None,
) -> None:
    if result.succeeded:
        return
    if result.timed_out:
        error = JobError(code=ErrorCode.TIMEOUT, message=f"{stage} 超时", stage=stage)
    elif result.output_limited:
        error = JobError(
            code=ErrorCode.OUTPUT_LIMIT,
            message=f"{stage} 输出超过限制",
            stage=stage,
        )
    elif result.parsed and result.parsed.get("status") == "wrong_answer":
        error = JobError(
            code=ErrorCode.WRONG_ANSWER,
            message="结果与参考实现不一致",
            stage=stage,
            details={"correctness": safe_correctness or {"status": "wrong_answer"}},
        )
    elif result.parsed and result.parsed.get("status") == "compile_error":
        error = JobError(
            code=ErrorCode.COMPILE_ERROR,
            message="运行时编译或提交策略检查失败",
            stage=stage,
            details={"result": {"status": "compile_error"}},
        )
    else:
        error = JobError(
            code=ErrorCode.RUNTIME_ERROR,
            message=f"{stage} 运行失败",
            stage=stage,
            details={
                "returncode": result.returncode,
                "result": safe_correctness or {"status": result.parsed.get("status")}
                if result.parsed
                else None,
            },
        )
    diagnostics = result.output if stage == "public" else None
    raise JobFailed(error, diagnostics)


def safe_correctness(
    problem: Problem,
    mode: str,
    parsed: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if parsed is None:
        return None
    if mode == "public":
        return parsed

    allowed_statuses = {"passed", "wrong_answer", "runtime_error"}
    status = str(parsed.get("status", "runtime_error"))
    if status not in allowed_statuses:
        status = "runtime_error"
    raw_cases = parsed.get("cases")
    safe_cases: list[dict[str, Any]] = []
    public_cases = problem.manifest.public.cases
    if isinstance(raw_cases, list):
        for index, raw_case in enumerate(raw_cases):
            if not isinstance(raw_case, dict):
                continue
            is_public = index < len(public_cases)
            name = (
                str(public_cases[index].get("name", f"sample_{index + 1}"))
                if is_public
                else f"internal_case_{index - len(public_cases) + 1}"
            )
            passed = raw_case.get("passed") is True
            safe_case: dict[str, Any] = {
                "name": name,
                "scope": "public" if is_public else "internal",
                "passed": passed,
            }
            if not passed:
                safe_case["message"] = "用例未通过"
            safe_cases.append(safe_case)

    if safe_cases:
        passed_count = sum(item["passed"] is True for item in safe_cases)
        summary = {
            "total": len(safe_cases),
            "passed": passed_count,
            "failed": len(safe_cases) - passed_count,
        }
    else:
        raw_summary = parsed.get("summary")
        source = raw_summary if isinstance(raw_summary, dict) else parsed
        total = source.get("total", 0)
        passed_count = source.get("passed", 0)
        total = total if isinstance(total, int) and total >= 0 else 0
        passed_count = (
            passed_count if isinstance(passed_count, int) and 0 <= passed_count <= total else 0
        )
        summary = {
            "total": total,
            "passed": passed_count,
            "failed": total - passed_count,
        }
    return {"status": status, "cases": safe_cases, "summary": summary}


def normalize_measurements(
    problem: Problem, result: ExecutionResult
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parsed = result.parsed or {}
    protocol = str(parsed.get("protocol_version", ""))
    expected_protocol = problem.manifest.benchmark.protocol_version
    if protocol != expected_protocol:
        raise JobFailed(
            JobError(
                code=ErrorCode.INTERNAL_ERROR,
                message="benchmark protocol version mismatch",
                stage="benchmarking",
            )
        )
    raw = parsed.get("measurements")
    if not isinstance(raw, list) or len(raw) != len(problem.manifest.benchmark.sizes):
        raise JobFailed(
            JobError(
                code=ErrorCode.INTERNAL_ERROR,
                message="benchmark returned an invalid measurement set",
                stage="benchmarking",
            )
        )
    normalized: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise JobFailed(
                JobError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="benchmark returned a malformed measurement",
                    stage="benchmarking",
                )
            )
        label = str(item.get("label", ""))
        if label in seen:
            raise JobFailed(
                JobError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="benchmark returned duplicate input sizes",
                    stage="benchmarking",
                )
            )
        seen.add(label)
        measurement = Measurement(
            size=label,
            samples_ms=item.get("samples_ms", []),
            inner_repetitions=item.get("inner_repetitions", 1),
        ).with_statistics()
        if len(measurement.samples_ms) != problem.manifest.benchmark.iterations:
            raise JobFailed(
                JobError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="benchmark sample count does not match the manifest",
                    stage="benchmarking",
                )
            )
        normalized.append(measurement.model_dump(mode="json"))
        samples.append({"size": label, "samples_ms": measurement.samples_ms[:200]})
    expected_labels = [item.label for item in problem.manifest.benchmark.sizes]
    if [item["size"] for item in normalized] != expected_labels:
        raise JobFailed(
            JobError(
                code=ErrorCode.INTERNAL_ERROR,
                message="benchmark input sizes do not match the manifest",
                stage="benchmarking",
            )
        )
    return normalized, samples
