from __future__ import annotations

import argparse
import json

from myleetgpu.application.jobs import JobService
from myleetgpu.config import get_settings
from myleetgpu.domain.problems import ProblemCatalog
from myleetgpu.infrastructure.database import Base, build_engine, build_session_factory
from myleetgpu.infrastructure.repository import Repository
from myleetgpu.runner.docker import DockerRunner


def build_components() -> tuple[Repository, JobService, DockerRunner]:
    settings = get_settings()
    settings.ensure_directories()
    engine = build_engine(settings)
    Base.metadata.create_all(engine)
    repository = Repository(build_session_factory(engine))
    catalog = ProblemCatalog(settings.problems_dir).load()
    return repository, JobService(settings, catalog, repository), DockerRunner(settings)


def main() -> None:
    parser = argparse.ArgumentParser(prog="myleetgpu")
    parser.add_argument("command", choices=["clean-jobs", "recover-runner", "environment"])
    args = parser.parse_args()
    repository, jobs, runner = build_components()
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
