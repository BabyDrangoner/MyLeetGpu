from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from myleetgpu.domain.problems import ProblemCatalog, ProblemManifest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBLEMS_ROOT = PROJECT_ROOT / "problems"
EXPECTED_SLUGS = {"vector-addition", "matrix-transpose", "reduction"}


@pytest.fixture(scope="module")
def catalog() -> ProblemCatalog:
    return ProblemCatalog(PROBLEMS_ROOT).load()


def load_raw_manifest(slug: str = "vector-addition") -> dict[str, object]:
    return yaml.safe_load((PROBLEMS_ROOT / slug / "problem.yaml").read_text(encoding="utf-8"))


def test_catalog_loads_exactly_the_three_builtin_original_problems(catalog: ProblemCatalog) -> None:
    assert len(catalog) == 3
    assert {problem.manifest.slug for problem in catalog.list()} == EXPECTED_SLUGS


@pytest.mark.parametrize("slug", sorted(EXPECTED_SLUGS))
def test_each_builtin_problem_loads_all_required_assets(catalog: ProblemCatalog, slug: str) -> None:
    problem = catalog.get(slug)

    assert problem.root == (PROBLEMS_ROOT / slug).resolve()
    assert problem.statement_markdown.strip()
    assert problem.starter_code.strip()
    assert problem.header_path.is_file()
    assert problem.validator_path.is_file()
    assert problem.benchmark_path.is_file()
    assert problem.manifest.signature.symbol == "solve"
    assert problem.manifest.benchmark.warmup > 0
    assert problem.manifest.benchmark.iterations >= 3
    assert problem.suite_hash and len(problem.suite_hash) == 64


def test_catalog_order_is_stable(catalog: ProblemCatalog) -> None:
    assert [item.manifest.slug for item in catalog.list()] == sorted(EXPECTED_SLUGS)


def test_manifest_integer_revision_is_normalized_to_string() -> None:
    raw = load_raw_manifest()
    assert raw["revision"] == 1

    manifest = ProblemManifest.model_validate(raw)

    assert manifest.revision == "1"


def test_public_problem_payload_never_leaks_internal_tests_or_harness_paths(
    catalog: ProblemCatalog,
) -> None:
    problem = catalog.get("reduction")

    detail = problem.public_detail()
    serialized = repr(detail)

    assert "internal" not in detail
    assert "public" not in detail
    assert "harness" not in detail
    assert "suite_seed" not in serialized
    assert "harness/validator.cu" not in serialized
    assert "signed_random" not in serialized
    assert detail["starter_code"] == problem.starter_code


def test_public_summary_does_not_include_source_or_test_configuration(
    catalog: ProblemCatalog,
) -> None:
    summary = catalog.get("vector-addition").public_summary()

    assert set(summary) == {"slug", "title", "difficulty", "revision", "summary"}
    assert summary["summary"]


def test_unknown_problem_has_a_clear_error(catalog: ProblemCatalog) -> None:
    with pytest.raises(KeyError, match="unknown problem: not-installed"):
        catalog.get("not-installed")


def test_catalog_rejects_absent_and_empty_problem_directories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        ProblemCatalog(tmp_path / "missing").load()

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no problem manifests"):
        ProblemCatalog(empty).load()


def test_catalog_requires_directory_name_to_match_manifest_slug(tmp_path: Path) -> None:
    source = PROBLEMS_ROOT / "vector-addition"
    destination = tmp_path / "wrong-directory"
    shutil.copytree(source, destination)

    with pytest.raises(ValueError, match="directory must equal slug"):
        ProblemCatalog(tmp_path).load()


@pytest.mark.parametrize(
    ("field_path", "invalid_value", "message"),
    [
        (("slug",), "../escape", "slug"),
        (("schema_version",), 2, "schema_version"),
        (("compiler", "executable"), "g++", "executable"),
        (("compiler", "architecture"), "sm_89", "architecture"),
        (("compiler", "allowed_flags"), ["-O3", "--run-user-shell"], "not allowed"),
        (("benchmark", "iterations"), 2, "iterations"),
        (("benchmark", "sizes"), [], "sizes"),
        (("public", "cases"), [], "cases"),
    ],
)
def test_manifest_rejects_unsafe_or_incomplete_protocols(
    field_path: tuple[str, ...], invalid_value: object, message: str
) -> None:
    raw = load_raw_manifest()
    target: dict[str, object] = raw
    for part in field_path[:-1]:
        target = target[part]  # type: ignore[assignment,index]
    target[field_path[-1]] = invalid_value

    with pytest.raises(ValidationError, match=message):
        ProblemManifest.model_validate(raw)


@pytest.mark.parametrize(
    ("field", "configured", "limit"),
    [
        ("public", 5001, 5000),
        ("internal", 10001, 10000),
        ("benchmark", 60001, 60000),
    ],
)
def test_nested_timeout_cannot_exceed_top_level_limit(
    field: str, configured: int, limit: int
) -> None:
    raw = load_raw_manifest()
    raw[field]["timeout_ms"] = configured  # type: ignore[index]
    timeout_fields = {
        "public": "public_ms",
        "internal": "validation_ms",
        "benchmark": "benchmark_ms",
    }
    top_level_field = timeout_fields[field]
    raw["timeouts"][top_level_field] = limit  # type: ignore[index]

    with pytest.raises(ValidationError, match="timeout exceeds"):
        ProblemManifest.model_validate(raw)


@pytest.mark.parametrize("unsafe_path", ["../secret.cu", "/etc/passwd", "C:\\secret.cu"])
def test_problem_assets_cannot_escape_problem_directory(tmp_path: Path, unsafe_path: str) -> None:
    destination = tmp_path / "vector-addition"
    shutil.copytree(PROBLEMS_ROOT / "vector-addition", destination)
    manifest_path = destination / "problem.yaml"
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    raw["starter"] = unsafe_path
    manifest_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="missing or escaped|must stay relative"):
        ProblemCatalog(tmp_path).load()


def test_suite_hash_is_stable_across_catalog_reloads() -> None:
    first = ProblemCatalog(PROBLEMS_ROOT).load()
    second = ProblemCatalog(PROBLEMS_ROOT).load()

    assert {problem.manifest.slug: problem.suite_hash for problem in first.list()} == {
        problem.manifest.slug: problem.suite_hash for problem in second.list()
    }
