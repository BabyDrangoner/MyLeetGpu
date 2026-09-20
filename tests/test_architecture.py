"""Executable module boundaries; no dependency-inspection framework is required."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "backend" / "myleetgpu"


def imported_modules(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def forbidden_imports(path: Path, prefixes: tuple[str, ...]) -> set[str]:
    return {
        module
        for module in imported_modules(path)
        if any(module == prefix or module.startswith(prefix + ".") for prefix in prefixes)
    }


@pytest.mark.parametrize("path", sorted((PACKAGE / "domain").glob("*.py")), ids=lambda p: p.name)
def test_domain_does_not_depend_on_processes_storage_or_delivery(path):
    assert not forbidden_imports(
        path,
        (
            "fastapi",
            "sqlalchemy",
            "subprocess",
            "myleetgpu.application",
            "myleetgpu.api",
            "myleetgpu.infrastructure",
            "myleetgpu.runner",
            "myleetgpu.worker",
        ),
    )


@pytest.mark.parametrize(
    "path", sorted((PACKAGE / "application").glob("*.py")), ids=lambda p: p.name
)
def test_application_uses_runner_contracts_not_concrete_execution_or_http(path):
    assert not forbidden_imports(
        path,
        (
            "fastapi",
            "subprocess",
            "sqlalchemy",
            "myleetgpu.api",
            "myleetgpu.worker",
            "myleetgpu.runner.docker",
            "myleetgpu.runner.colab",
            "myleetgpu.runner.router",
        ),
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "session_factory"
        for node in ast.walk(tree)
    ), "SQL and transaction scope belong to Repository, not application services"


def test_result_policy_has_no_storage_or_execution_adapter_dependencies():
    assert not forbidden_imports(
        PACKAGE / "application" / "results.py",
        ("myleetgpu.infrastructure", "myleetgpu.runner.docker", "myleetgpu.runner.colab"),
    )
