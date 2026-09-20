from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query

from myleetgpu.api.dependencies import RepositoryDependency, SettingsDependency
from myleetgpu.api.presenters import environment_response, unavailable_telemetry
from myleetgpu.api.runtime_status import runner_circuit_error
from myleetgpu.api.schemas import ExecutionProbeRequest, ExecutionSettingsUpdate, KernelLanguage
from myleetgpu.application.execution import wait_for_probe
from myleetgpu.domain.jobs import GPU_RESOURCE

router = APIRouter()


@router.get("/execution-settings")
def get_execution_settings(
    repository: RepositoryDependency, settings: SettingsDependency
) -> dict[str, Any]:
    return {
        **repository.execution_settings(),
        "colab": {
            "ssh_host": settings.colab_ssh_host,
            "remote_root": settings.colab_remote_root,
            "isolation": "trusted-native",
        },
    }


@router.put("/execution-settings")
def put_execution_settings(
    body: ExecutionSettingsUpdate,
    repository: RepositoryDependency,
    settings: SettingsDependency,
) -> dict[str, Any]:
    repository.update_execution_settings(body.target, body.colab_acknowledged)
    return get_execution_settings(repository, settings)


@router.post("/execution-settings/probe")
async def test_execution_connection(
    body: ExecutionProbeRequest,
    repository: RepositoryDependency,
    settings: SettingsDependency,
) -> dict[str, Any]:
    if not repository.has_active_lease(GPU_RESOURCE):
        raise HTTPException(status_code=503, detail="Worker 未运行，请先启动 Worker 再测试连接")
    try:
        return await wait_for_probe(
            repository,
            body.target,
            body.language,
            settings.execution_probe_timeout_seconds,
        )
    except TimeoutError as error:
        raise HTTPException(status_code=504, detail=str(error)) from error


@router.get("/environment")
def environment(
    repository: RepositoryDependency,
    settings: SettingsDependency,
    language: Annotated[KernelLanguage, Query()] = "cuda_cpp",
) -> dict[str, Any]:
    target = "cpu" if language in {"cpp", "python"} else repository.execution_settings()["target"]
    record = repository.latest_environment(language, execution_target=target)
    if record is None:
        return {
            "status": "unknown",
            "healthy": False,
            "backend": language,
            "execution_target": target,
            "error": "Worker 尚未完成该语言运行环境探测",
            "telemetry": unavailable_telemetry(),
        }
    payload = environment_response(record)
    worker_active = repository.has_active_lease(GPU_RESOURCE)
    circuit_error = runner_circuit_error(settings, target, language)
    payload["worker_active"] = worker_active
    if not worker_active or circuit_error:
        payload["healthy"] = False
        payload["status"] = "unavailable"
        payload["error"] = circuit_error or "Worker 未运行或租约已过期"
    return payload
