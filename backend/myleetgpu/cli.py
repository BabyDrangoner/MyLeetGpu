from __future__ import annotations

import argparse
import json

from myleetgpu.application.jobs import JobService
from myleetgpu.config import get_settings
from myleetgpu.domain.jobs import GPU_RESOURCE
from myleetgpu.domain.problems import ProblemCatalog
from myleetgpu.infrastructure.database import Base, build_engine, build_session_factory
from myleetgpu.infrastructure.repository import Repository
from myleetgpu.runner.router import ExecutionRouter


def build_components() -> tuple[Repository, JobService, ExecutionRouter]:
    settings = get_settings()
    settings.ensure_directories()
    engine = build_engine(settings)
    Base.metadata.create_all(engine)
    repository = Repository(build_session_factory(engine))
    catalog = ProblemCatalog(settings.problems_dir).load()
    runner = ExecutionRouter(settings)
    runner.select_target(repository.execution_settings()["target"])
    return repository, JobService(settings, catalog, repository), runner


def main() -> None:
    parser = argparse.ArgumentParser(prog="myleetgpu")
    parser.add_argument("command", choices=["clean-jobs", "recover-runner", "environment"])
    parser.add_argument(
        "--target", choices=["local", "colab"], help="Override the saved execution target"
    )
    args = parser.parse_args()
    repository, jobs, runner = build_components()
    if args.target:
        runner.select_target(args.target)
    if args.command != "clean-jobs" and repository.has_active_lease(GPU_RESOURCE):
        parser.error("Worker 正在运行；请使用网页连接测试，或先停止 Worker 再执行命令行恢复")
    if args.command == "clean-jobs":
        removed = jobs.cleanup_stale_spool()
        print(json.dumps({"removed": removed, "count": len(removed)}, ensure_ascii=False))
    elif args.command == "recover-runner":
        probe = runner.recover()
        repository.save_environment(probe)
        print(json.dumps(probe.__dict__, ensure_ascii=False, indent=2))
    else:
        probe = runner.probe_environment(force=True, ignore_circuit_breaker=True)
        repository.save_environment(probe)
        print(json.dumps(probe.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
