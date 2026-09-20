from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROBLEM = ROOT / "problems" / "online-softmax"
PREFIX = "MYLEETGPU_RESULT="


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def common() -> ModuleType:
    return load_module(PROBLEM / "python/harness/online_softmax_common.py", "online_test_common")


@pytest.fixture(scope="module")
def starter() -> ModuleType:
    return load_module(PROBLEM / "python/starter.py", "online_test_starter")


def result(command: list[str], source: Path | None = None) -> tuple[int, dict]:
    environment = dict(os.environ)
    if source is not None:
        environment["MYLEETGPU_SOURCE_PATH"] = str(source)
    completed = subprocess.run(
        command, capture_output=True, text=True, env=environment, timeout=30, check=False
    )
    records = [
        line.removeprefix(PREFIX)
        for line in completed.stdout.splitlines()
        if line.startswith(PREFIX)
    ]
    assert len(records) == 1, completed.stderr or completed.stdout
    return completed.returncode, json.loads(records[0])


@pytest.fixture(scope="module")
def cpp_binaries(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++17 compiler is not installed")
    directory = tmp_path_factory.mktemp("online-softmax-cpp")
    binaries = {}
    for kind in ("validator", "benchmark"):
        binary = directory / kind
        completed = subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O3",
                "-I",
                str(PROBLEM / "cpp"),
                str(PROBLEM / "cpp/starter.cpp"),
                str(PROBLEM / f"cpp/harness/{kind}.cpp"),
                "-o",
                str(binary),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        binaries[kind] = binary
    return binaries


def test_manifest_matches_harness_cases_and_benchmark(common: ModuleType) -> None:
    manifest = yaml.safe_load((PROBLEM / "problem.yaml").read_text())
    assert manifest["default_language"] == "python"
    assert set(manifest["implementations"]) == {"cpp", "python"}
    assert [case[0] for case in common.cases(False)] == [
        case["name"] for case in manifest["public"]["cases"]
    ]
    assert [case[0] for case in common.cases(True)] == [
        case["name"] for case in manifest["public"]["cases"] + manifest["internal"]["cases"]
    ]
    assert manifest["benchmark"]["iterations"] == common.ITERATIONS
    assert manifest["benchmark"]["warmup"] == common.WARMUP
    assert [case[0] for case in common.BENCHMARK_CASES] == [
        size["label"] for size in manifest["benchmark"]["sizes"]
    ]


@pytest.mark.parametrize("mode,count", [("public", 3), ("full", 11)])
def test_python_starter_validator(mode: str, count: int) -> None:
    code, payload = result(
        [sys.executable, str(PROBLEM / "python/harness/validator.py"), "--mode", mode],
        PROBLEM / "python/starter.py",
    )
    assert code == 0
    assert payload["status"] == "passed"
    assert payload["summary"] == {"total": count, "passed": count, "failed": 0}


@pytest.mark.parametrize("mode,count", [("public", 3), ("full", 11)])
def test_cpp_starter_validator(cpp_binaries: dict[str, Path], mode: str, count: int) -> None:
    code, payload = result([str(cpp_binaries["validator"]), "--mode", mode])
    assert code == 0
    assert payload["status"] == "passed"
    assert payload["summary"] == {"total": count, "passed": count, "failed": 0}


@pytest.mark.parametrize("language", ["cpp", "python"])
def test_cpu_benchmark_protocol(
    language: str, cpp_binaries: dict[str, Path], common: ModuleType
) -> None:
    if language == "cpp":
        command = [str(cpp_binaries["benchmark"]), "--mode", "benchmark"]
    else:
        command = [
            sys.executable,
            str(PROBLEM / "python/harness/benchmark.py"),
            "--mode",
            "benchmark",
        ]
    code, payload = result(command, PROBLEM / "python/starter.py")
    assert code == 0
    assert payload["status"] == "passed"
    assert payload["protocol_version"] == "1"
    assert [item["label"] for item in payload["measurements"]] == [
        case[0] for case in common.BENCHMARK_CASES
    ]
    for item, case in zip(payload["measurements"], common.BENCHMARK_CASES, strict=True):
        assert item["inner_repetitions"] == case[3]
        assert len(item["samples_ms"]) == common.ITERATIONS
        assert all(math.isfinite(sample) and sample > 0 for sample in item["samples_ms"])


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [0.0, 1.0],
        (0.0,),
        (False, 1.0),
        (0.0, 0.0),
        (math.nan, 1.0),
        (0.0, math.inf),
        (0.0, math.nan),
    ],
)
def test_validator_rejects_invalid_statistics(common: ModuleType, invalid: object) -> None:
    with pytest.raises(AssertionError):
        common.check_state(invalid, (0.0, 1.0))


def test_validator_rejects_input_mutation(common: ModuleType, starter: ModuleType) -> None:
    def mutating_update(maximum: float, denominator: float, chunk: list[float]) -> tuple:
        state = starter.online_softmax_update(maximum, denominator, chunk)
        chunk.append(0.0)
        return state

    with pytest.raises(AssertionError, match="must not be modified"):
        common.check_stream(mutating_update, [], [[1.0, 2.0]])


def test_validator_rejects_missing_rescale(common: ModuleType) -> None:
    def no_rescale(maximum: float, denominator: float, chunk: list[float]) -> tuple:
        for value in chunk:
            maximum = max(maximum, value)
            denominator += math.exp(value - maximum)
        return maximum, denominator

    with pytest.raises(AssertionError, match="rescaled"):
        common.check_stream(no_rescale, [], [[1.0], [2.0]])


def test_validator_rejects_ignoring_carried_history(common: ModuleType) -> None:
    def stateless(maximum: float, denominator: float, chunk: list[float]) -> tuple:
        return common.reference(chunk) if chunk else (maximum, denominator)

    _, history, chunks = next(
        case for case in common.cases(True) if case[0] == "internal_carried_state"
    )
    with pytest.raises(AssertionError, match="rescaled"):
        common.check_stream(stateless, history, chunks)


def test_chunk_boundaries_preserve_final_statistics(
    common: ModuleType, starter: ModuleType
) -> None:
    values = common.random_values(4099)
    for chunk_size in (1, 7, 64, 4099):
        chunks = common.partition(values, chunk_size)
        state = common.run_stream(starter.online_softmax_update, chunks)
        common.check_probabilities(values, state)
    assert starter.online_softmax_update(-math.inf, 0.0, []) == (-math.inf, 0.0)
    assert starter.online_softmax_update(9.0, 3.0, []) == (9.0, 3.0)


@pytest.mark.parametrize(
    "source",
    [
        "def online_softmax_update(m, d, chunk): return (0.0, 0.0)\n",
        "def online_softmax_update(m, d, chunk): return [m, d]\n",
    ],
)
def test_python_wrong_answers_fail_with_protocol_status(tmp_path: Path, source: str) -> None:
    path = tmp_path / "wrong.py"
    path.write_text(source)
    code, payload = result(
        [sys.executable, str(PROBLEM / "python/harness/validator.py"), "--mode", "full"], path
    )
    assert code == 1
    assert payload["status"] == "wrong_answer"
    assert payload["summary"]["failed"] > 0


@pytest.mark.parametrize(
    "body",
    [
        "for (double x : chunk) { m = std::max(m, x); d += std::exp(x - m); } return {m, d};",
        "const_cast<std::vector<double>&>(chunk).push_back(1.0); return {m, d};",
    ],
)
def test_cpp_wrong_answers_fail_with_protocol_status(tmp_path: Path, body: str) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++17 compiler is not installed")
    source = tmp_path / "wrong.cpp"
    source.write_text(
        '#include "solve.h"\n#include <algorithm>\n#include <cmath>\n'
        "std::pair<double, double> online_softmax_update("
        "double m, double d, const std::vector<double>& chunk) {" + body + "}\n"
    )
    binary = tmp_path / "wrong"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O3",
            "-I",
            str(PROBLEM / "cpp"),
            str(source),
            str(PROBLEM / "cpp/harness/validator.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    code, payload = result([str(binary), "--mode", "full"])
    assert code == 1
    assert payload["status"] == "wrong_answer"
    assert payload["summary"]["failed"] > 0
