"""Read-only runner circuit status shared by readiness and environment routes."""

from __future__ import annotations

import json

from myleetgpu.config import Settings


def runner_circuit_error(
    settings: Settings, target: str = "local", language: str = "cuda_cpp"
) -> str | None:
    if target == "cpu":
        if language not in {"cpp", "python"}:
            return "无效的 CPU 实现语言"
        name = f"cpu-{language}-runner-unhealthy.json"
    else:
        name = "colab-runner-unhealthy.json" if target == "colab" else "runner-unhealthy.json"
    path = settings.data_dir / name
    reason = "CPU Runner 已熔断" if target == "cpu" else "GPU Runner 已熔断"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            recorded_reason = payload.get("reason")
            if isinstance(recorded_reason, str) and recorded_reason.strip():
                reason = recorded_reason
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError):
        # A present but unreadable/malformed marker must fail closed. Reading
        # directly also avoids treating a failed stat as a healthy runner.
        pass
    if target == "colab":
        return f"{reason}；请在运行环境页测试 Colab 连接，健康检查通过后解除熔断"
    if target == "cpu":
        return f"{reason}；请重新测试该 CPU 语言的运行环境"
    return f"{reason}；运行 make doctor 并执行 make recover-runner"
