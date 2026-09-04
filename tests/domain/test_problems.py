from __future__ import annotations

import ast
import copy
import shutil
from contextlib import suppress
from pathlib import Path

import pytest
import yaml
from myleetgpu.domain.problems import KernelLanguage, ProblemCatalog, ProblemManifest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBLEMS_ROOT = PROJECT_ROOT / "problems"
KERNEL_SLUGS = {
    "matrix-multiplication",
    "matrix-transpose",
    "max-reduction",
    "reduction",
    "softmax",
    "top-k",
    "top-p",
    "vector-addition",
}
TORCH_SLUGS = {"multi-head-attention", "grouped-query-attention"}
TORCH_ENTRYPOINTS = {
    "multi-head-attention": "MultiHeadAttention",
    "grouped-query-attention": "GroupedQueryAttention",
}
EXPECTED_SLUGS = KERNEL_SLUGS | TORCH_SLUGS


@pytest.fixture(scope="module")
def catalog() -> ProblemCatalog:
    return ProblemCatalog(PROBLEMS_ROOT).load()


def load_raw_manifest(slug: str = "vector-addition") -> dict[str, object]:
    return yaml.safe_load((PROBLEMS_ROOT / slug / "problem.yaml").read_text(encoding="utf-8"))


def test_catalog_loads_all_ten_builtin_original_problems(catalog: ProblemCatalog) -> None:
    assert len(catalog) == 10
    assert {problem.manifest.slug for problem in catalog.list()} == EXPECTED_SLUGS


def test_new_kernel_protocols_cover_their_defining_operations_and_boundaries(
    catalog: ProblemCatalog,
) -> None:
    reduction = catalog.get("reduction")
    maximum = catalog.get("max-reduction")
    softmax = catalog.get("softmax")
    matmul = catalog.get("matrix-multiplication")
    top_k = catalog.get("top-k")
    top_p = catalog.get("top-p")

    assert reduction.manifest.revision == "2"
    assert "tl.sum" in reduction.get_implementation("triton_python").starter_code

    assert maximum.manifest.tolerance.atol == 0.0
    assert maximum.manifest.tolerance.rtol == 0.0
    assert {case["pattern"] for case in maximum.manifest.internal.cases} >= {
        "all_negative",
        "extremes",
        "signed_zero",
    }
    assert "tl.atomic_max" in maximum.get_implementation("triton_python").starter_code

    softmax_cases = softmax.manifest.internal.cases
    assert any(case["rows"] == 65536 and case["cols"] == 1 for case in softmax_cases)
    assert any(case["rows"] * case["cols"] >= 16_000_000 for case in softmax_cases)
    assert "triton.next_power_of_2" in softmax.get_implementation("triton_python").starter_code

    matmul_cases = matmul.manifest.internal.cases
    assert all(any(case[axis] == 4096 for case in matmul_cases) for axis in ("m", "k", "n"))
    assert "tl.dot" in matmul.get_implementation("triton_python").starter_code

    top_k_starter = top_k.get_implementation("triton_python").starter_code
    assert "tl.argmax" in top_k_starter
    assert "k: tl.constexpr" not in top_k_starter
    assert any(case["k"] == case["cols"] for case in top_k.manifest.internal.cases)

    top_p_starter = top_p.get_implementation("triton_python").starter_code
    assert "tl.sort" in top_p_starter
    assert "tl.cumsum" in top_p_starter
    assert "below_threshold.to(tl.int32)" in top_p_starter
    assert "tl.argsort" not in top_p_starter
    assert any(case["p"] == 1.0 for case in top_p.manifest.internal.cases)
    assert any(
        case["rows"] == 65536 and case["cols"] == 1 for case in top_p.manifest.internal.cases
    )


@pytest.mark.parametrize("slug", sorted(KERNEL_SLUGS))
def test_each_kernel_problem_loads_cuda_and_triton_assets(
    catalog: ProblemCatalog, slug: str
) -> None:
    problem = catalog.get(slug)

    assert problem.root == (PROBLEMS_ROOT / slug).resolve()
    assert problem.manifest.schema_version == 2
    assert problem.statement_markdown.strip()
    assert problem.default_language is KernelLanguage.CUDA_CPP
    assert problem.supported_languages == (
        KernelLanguage.CUDA_CPP,
        KernelLanguage.TRITON_PYTHON,
    )

    cuda = problem.get_implementation(KernelLanguage.CUDA_CPP)
    triton = problem.get_implementation("triton_python")
    for implementation in (cuda, triton):
        assert implementation.starter_code.strip()
        assert implementation.statement_appendix
        assert implementation.validator_path.is_file()
        assert implementation.benchmark_path.is_file()
        assert implementation.signature.symbol == "solve"
        assert implementation.suite_hash and len(implementation.suite_hash) == 64

    assert cuda.source_suffix == ".cu"
    assert cuda.editor_language == "cpp"
    assert cuda.display_name == "CUDA C++"
    assert cuda.header_path is not None and cuda.header_path.is_file()
    assert cuda.compile_flags == ["--std=c++17", "-O3"]
    assert triton.source_suffix == ".py"
    assert triton.editor_language == "python"
    assert triton.display_name == "Triton Python"
    assert triton.header_path is None
    assert triton.compile_flags == []
    assert "@triton.jit" in triton.starter_code
    assert "cudaStream_t" in cuda.statement_appendix
    assert "torch.Tensor" in triton.statement_appendix
    assert "@triton.jit" in triton.statement_appendix
    assert "cudaStream_t" not in triton.statement_appendix
    assert "cudaStream_t" not in problem.statement_markdown
    assert "void solve(" not in problem.statement_markdown

    # Existing callers continue to see the default CUDA implementation.
    assert problem.starter_code.strip()
    assert problem.starter_code == cuda.starter_code
    assert problem.header_path == cuda.header_path
    assert problem.validator_path.is_file()
    assert problem.benchmark_path.is_file()
    assert problem.validator_path == cuda.validator_path
    assert problem.benchmark_path == cuda.benchmark_path
    assert problem.compile_flags == cuda.compile_flags
    assert problem.suite_hash == cuda.suite_hash
    assert problem.manifest.benchmark.warmup > 0
    assert problem.manifest.benchmark.iterations >= 3
    assert problem.suite_hash and len(problem.suite_hash) == 64


@pytest.mark.parametrize("slug", sorted(TORCH_SLUGS))
def test_each_attention_problem_is_torch_only_and_uses_its_default_assets(
    catalog: ProblemCatalog, slug: str
) -> None:
    problem = catalog.get(slug)

    assert problem.root == (PROBLEMS_ROOT / slug).resolve()
    assert problem.manifest.schema_version == 2
    assert problem.statement_markdown.strip()
    assert problem.default_language is KernelLanguage.TORCH_PYTHON
    assert problem.supported_languages == (KernelLanguage.TORCH_PYTHON,)

    implementation = problem.get_implementation()
    assert implementation.language is KernelLanguage.TORCH_PYTHON
    assert implementation.source_suffix == ".py"
    assert implementation.editor_language == "python"
    assert implementation.display_name == "PyTorch Python"
    assert implementation.header_path is None
    assert implementation.compile_flags == []
    assert implementation.toolchain_profile == "torch_cuda_v1"
    assert implementation.starter_code.strip()
    assert problem.manifest.revision == "2"
    starter_tree = ast.parse(implementation.starter_code)
    entry_class = next(
        node
        for node in starter_tree.body
        if isinstance(node, ast.ClassDef) and node.name == TORCH_ENTRYPOINTS[slug]
    )
    forward = next(
        node
        for node in entry_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    assert [argument.arg for argument in forward.args.args] == ["self", "X", "isCasual"]
    assert implementation.statement_appendix
    assert "torch.Tensor" in implementation.statement_appendix
    assert implementation.validator_path.is_file()
    assert implementation.benchmark_path.is_file()
    assert implementation.signature.symbol == TORCH_ENTRYPOINTS[slug]
    assert len(implementation.suite_hash) == 64
    assert [item["name"] for item in problem.manifest.types["inputs"]] == ["X", "isCasual"]
    assert problem.manifest.types["inputs"][1]["type"] == "bool"

    assert problem.starter_code == implementation.starter_code
    assert problem.header_path is None
    assert problem.validator_path == implementation.validator_path
    assert problem.benchmark_path == implementation.benchmark_path
    assert problem.suite_hash == implementation.suite_hash
    with pytest.raises(KeyError, match="does not support language: cuda_cpp"):
        problem.get_implementation(KernelLanguage.CUDA_CPP)


def test_catalog_order_is_stable(catalog: ProblemCatalog) -> None:
    assert [item.manifest.slug for item in catalog.list()] == sorted(EXPECTED_SLUGS)


def test_manifest_integer_revision_is_normalized_to_string() -> None:
    raw = load_raw_manifest()
    assert raw["revision"] == 1

    manifest = ProblemManifest.model_validate(raw)

    assert manifest.revision == "1"


def test_schema_v1_is_normalized_to_one_cuda_implementation() -> None:
    raw = load_raw_manifest()
    cuda = raw["implementations"]["cuda_cpp"]  # type: ignore[index]
    raw["schema_version"] = 1
    raw.pop("default_language")
    raw.pop("implementations")
    raw.update(
        {
            "language": "cuda_cpp",
            "signature": cuda["signature"],
            "starter": cuda["starter"],
            "harness": cuda["harness"],
            "compiler": cuda["compiler"],
        }
    )

    manifest = ProblemManifest.model_validate(raw)

    assert manifest.schema_version == 1
    assert manifest.default_language is KernelLanguage.CUDA_CPP
    assert set(manifest.implementations) == {KernelLanguage.CUDA_CPP}
    assert manifest.implementations[KernelLanguage.CUDA_CPP].starter == "starter.cu"


def test_schema_v1_problem_package_keeps_legacy_problem_attributes(tmp_path: Path) -> None:
    destination = tmp_path / "vector-addition"
    shutil.copytree(PROBLEMS_ROOT / "vector-addition", destination)
    manifest_path = destination / "problem.yaml"
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    cuda = raw["implementations"]["cuda_cpp"]
    raw["schema_version"] = 1
    raw.pop("default_language")
    raw.pop("implementations")
    raw.update(
        {
            "language": "cuda_cpp",
            "signature": cuda["signature"],
            "starter": cuda["starter"],
            "harness": cuda["harness"],
            "compiler": cuda["compiler"],
        }
    )
    manifest_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    problem = ProblemCatalog(tmp_path).load().get("vector-addition")

    assert problem.default_language is KernelLanguage.CUDA_CPP
    assert problem.supported_languages == (KernelLanguage.CUDA_CPP,)
    assert problem.starter_code == problem.get_implementation().starter_code
    assert problem.header_path is not None and problem.header_path.is_file()
    assert problem.validator_path.is_file()
    assert problem.benchmark_path.is_file()
    assert problem.compile_flags == ["--std=c++17", "-O3"]


def test_schema_v2_rejects_legacy_fields_and_unknown_keys() -> None:
    with_legacy = load_raw_manifest()
    with_legacy["starter"] = "starter.cu"
    with pytest.raises(ValidationError, match="schema v2"):
        ProblemManifest.model_validate(with_legacy)

    with_unknown = load_raw_manifest()
    with_unknown["surprise_runner_command"] = "curl example.invalid"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProblemManifest.model_validate(with_unknown)


def test_implementation_mapping_key_must_match_language() -> None:
    raw = load_raw_manifest()
    cuda = copy.deepcopy(raw["implementations"]["cuda_cpp"])  # type: ignore[index]
    raw["implementations"]["triton_python"] = cuda  # type: ignore[index]

    with pytest.raises(ValidationError, match="implementation key must match"):
        ProblemManifest.model_validate(raw)


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
    assert "triton_torch_cuda_v1" not in serialized
    assert detail["starter_code"] == problem.starter_code
    assert detail["default_language"] == "cuda_cpp"
    assert detail["supported_languages"] == ["cuda_cpp", "triton_python"]
    assert set(detail["implementations"]) == {"cuda_cpp", "triton_python"}
    assert detail["implementations"]["triton_python"]["editor_language"] == "python"
    assert "@triton.jit" in detail["implementations"]["triton_python"]["starter_code"]
    assert "@triton.jit" in detail["implementations"]["triton_python"]["statement_appendix"]
    assert "cudaStream_t" in detail["implementations"]["cuda_cpp"]["statement_appendix"]
    assert all(item["name"] != "stream" for item in detail["types"]["inputs"])


def test_public_torch_problem_payload_uses_default_language_without_leaking_harness(
    catalog: ProblemCatalog,
) -> None:
    problem = catalog.get("multi-head-attention")

    detail = problem.public_detail()
    serialized = repr(detail)

    assert detail["language"] == "torch_python"
    assert detail["default_language"] == "torch_python"
    assert detail["supported_languages"] == ["torch_python"]
    assert set(detail["implementations"]) == {"torch_python"}
    assert detail["starter_code"] == problem.get_implementation().starter_code
    assert detail["implementations"]["torch_python"]["display_name"] == "PyTorch Python"
    assert detail["implementations"]["torch_python"]["source_suffix"] == ".py"
    assert "internal" not in detail
    assert "public" not in detail
    assert "harness" not in detail
    assert "suite_seed" not in serialized
    assert "torch_cuda_v1" not in serialized
    assert "restricted_torch_" not in serialized


def test_public_summary_does_not_include_source_or_test_configuration(
    catalog: ProblemCatalog,
) -> None:
    summary = catalog.get("vector-addition").public_summary()

    assert set(summary) == {
        "slug",
        "title",
        "difficulty",
        "revision",
        "summary",
        "languages",
    }
    assert summary["summary"]
    assert summary["languages"] == ["cuda_cpp", "triton_python"]


def test_unknown_problem_has_a_clear_error(catalog: ProblemCatalog) -> None:
    with pytest.raises(KeyError, match="unknown problem: not-installed"):
        catalog.get("not-installed")


def test_unknown_problem_language_has_a_clear_error(catalog: ProblemCatalog) -> None:
    with pytest.raises(KeyError, match="does not support language: mojo"):
        catalog.get("vector-addition").get_implementation("mojo")


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
        (("schema_version",), 3, "schema_version"),
        (("implementations", "cuda_cpp", "compiler", "executable"), "g++", "executable"),
        (
            ("implementations", "cuda_cpp", "compiler", "architecture"),
            "sm_89",
            "architecture",
        ),
        (
            ("implementations", "cuda_cpp", "compiler", "allowed_flags"),
            ["-O3", "--run-user-shell"],
            "not allowed",
        ),
        (
            ("implementations", "triton_python", "runtime", "profile"),
            "download-at-runtime",
            "profile",
        ),
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


def test_torch_runtime_profile_is_fixed_by_the_manifest_schema() -> None:
    raw = load_raw_manifest("multi-head-attention")
    raw["implementations"]["torch_python"]["runtime"]["profile"] = "install-at-runtime"  # type: ignore[index]

    with pytest.raises(ValidationError, match="profile"):
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
    raw["implementations"]["triton_python"]["starter"] = unsafe_path
    manifest_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="missing or escaped|must stay relative"):
        ProblemCatalog(tmp_path).load()


def test_suite_hash_is_stable_across_catalog_reloads() -> None:
    first = ProblemCatalog(PROBLEMS_ROOT).load()
    second = ProblemCatalog(PROBLEMS_ROOT).load()

    assert {problem.manifest.slug: problem.suite_hash for problem in first.list()} == {
        problem.manifest.slug: problem.suite_hash for problem in second.list()
    }


def _literal_assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                with suppress(TypeError, ValueError):
                    values[target.id] = ast.literal_eval(node.value)
    return values


@pytest.mark.parametrize("slug", sorted(KERNEL_SLUGS))
def test_triton_assets_are_valid_python_and_mirror_benchmark_manifest(
    catalog: ProblemCatalog, slug: str
) -> None:
    problem = catalog.get(slug)
    implementation = problem.get_implementation(KernelLanguage.TRITON_PYTHON)

    for path in (
        implementation.starter_path,
        implementation.validator_path,
        implementation.benchmark_path,
    ):
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")

    starter = implementation.starter_code
    validator = implementation.validator_path.read_text(encoding="utf-8")
    benchmark = implementation.benchmark_path.read_text(encoding="utf-8")
    assignments = _literal_assignments(implementation.benchmark_path)
    configured_sizes = problem.manifest.benchmark.sizes

    assert "@triton.jit" in starter
    assert "def solve(" in starter
    assert "torch.cuda.stream(stream)" in validator
    assert "torch.cuda.stream(stream)" in benchmark
    assert "torch.cuda.Event(enable_timing=True)" in benchmark
    assert "triton.testing.do_bench" not in benchmark
    assert '"compile_error"' in validator
    assert '"compile_error"' in benchmark
    assert benchmark.count("returned = module.solve") == 2
    assert 'f"benchmark failed: {error}"' not in benchmark
    assert validator.count("MYLEETGPU_RESULT=") == 1
    assert benchmark.count("MYLEETGPU_RESULT=") == 1
    assert assignments["PROTOCOL_VERSION"] == problem.manifest.benchmark.protocol_version
    assert assignments["WARMUP"] == problem.manifest.benchmark.warmup
    assert assignments["ITERATIONS"] == problem.manifest.benchmark.iterations

    cases = assignments["CASES"]
    assert isinstance(cases, tuple)
    assert [case[0] for case in cases] == [size.label for size in configured_sizes]
    assert [case[-1] for case in cases] == [size.inner_repetitions for size in configured_sizes]


@pytest.mark.parametrize("slug", sorted(TORCH_SLUGS))
def test_torch_assets_are_valid_python_and_mirror_benchmark_manifest(
    catalog: ProblemCatalog, slug: str
) -> None:
    problem = catalog.get(slug)
    implementation = problem.get_implementation(KernelLanguage.TORCH_PYTHON)

    for path in (
        implementation.starter_path,
        implementation.validator_path,
        implementation.benchmark_path,
    ):
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")

    validator = implementation.validator_path.read_text(encoding="utf-8")
    benchmark = implementation.benchmark_path.read_text(encoding="utf-8")
    assignments = _literal_assignments(implementation.benchmark_path)

    assert f"class {TORCH_ENTRYPOINTS[slug]}:" in implementation.starter_code
    assert "isCasual" in implementation.starter_code
    assert "scaled_dot_product_attention" not in implementation.starter_code
    assert "torch.cuda.stream(stream)" in validator
    assert "torch.cuda.stream(stream)" in benchmark
    assert "torch.cuda.Event(enable_timing=True)" in benchmark
    for harness in (validator, benchmark):
        assert "isinstance(output, torch.Tensor)" in harness
        assert 'output.device.type != "cuda"' in harness
        assert "output.dtype != torch.float32" in harness
        assert "tuple(output.shape)" in harness
        assert "torch.isfinite" in harness
        assert "tensor.clone() for tensor" in harness
        assert "untyped_storage().data_ptr()" in harness
        assert "torch.backends.cuda.matmul.allow_tf32 = False" in harness
        assert "torch.backends.cudnn.allow_tf32 = False" in harness
        assert 'torch.set_float32_matmul_precision("highest")' in harness
        assert "torch.use_deterministic_algorithms(True)" in harness
    assert validator.count("MYLEETGPU_RESULT=") == 1
    assert benchmark.count("MYLEETGPU_RESULT=") == 1
    assert ".forward(" in validator
    assert ".forward(" in benchmark
    assert assignments["PROTOCOL_VERSION"] == problem.manifest.benchmark.protocol_version
    assert assignments["WARMUP"] == problem.manifest.benchmark.warmup
    assert assignments["ITERATIONS"] == problem.manifest.benchmark.iterations

    cases = assignments["CASES"]
    assert isinstance(cases, tuple)
    assert [case[0] for case in cases] == [size.label for size in problem.manifest.benchmark.sizes]
    assert [case[-1] for case in cases] == [
        size.inner_repetitions for size in problem.manifest.benchmark.sizes
    ]
