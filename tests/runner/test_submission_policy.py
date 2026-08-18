from __future__ import annotations

from pathlib import Path

import pytest
from myleetgpu.runner.submission_policy import (
    POLICY_VERSION,
    SubmissionPolicyError,
    validate_source,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("relative_path", "parameters"),
    [
        ("problems/vector-addition/triton/starter.py", ("a", "b", "output", "n")),
        ("problems/reduction/triton/starter.py", ("input", "output", "n")),
        (
            "problems/matrix-transpose/triton/starter.py",
            ("input", "output", "rows", "cols"),
        ),
    ],
)
def test_builtin_triton_starters_are_accepted(
    relative_path: str, parameters: tuple[str, ...]
) -> None:
    source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")

    validate_source(source, expected_parameters=parameters)


@pytest.mark.parametrize(
    "source",
    [
        """
import torch
import triton
import triton.language as tl
import __main__
@triton.jit
def kernel(output, n: tl.constexpr):
    tl.store(output, 1.0)
def solve(output: torch.Tensor, n: int) -> None:
    kernel[(1,)](output, n)
""",
        """
import torch
import triton
import triton.language as tl
open('/work/platform.py').read()
@triton.jit
def kernel(output, n: tl.constexpr):
    tl.store(output, 1.0)
def solve(output: torch.Tensor, n: int) -> None:
    kernel[(1,)](output, n)
""",
        """
import torch
import triton
import triton.language as tl
print('MYLEETGPU_RESULT={"status":"passed","measurements":[]}')
@triton.jit
def kernel(output, n: tl.constexpr):
    tl.store(output, 1.0)
def solve(output: torch.Tensor, n: int) -> None:
    kernel[(1,)](output, n)
""",
        """
import torch
import triton
import triton.language as tl
torch.isclose = lambda *args: True
@triton.jit
def kernel(output, n: tl.constexpr):
    tl.store(output, 1.0)
def solve(output: torch.Tensor, n: int) -> None:
    kernel[(1,)](output, n)
""",
        """
import torch
import triton
import triton.language as tl
@triton.jit
def kernel(output, n: tl.constexpr):
    tl.store(output, 1.0)
def solve(output: torch.Tensor, n: int) -> None:
    value = output.__class__
    kernel[(1,)](output, n)
""",
        """
import torch
import triton
import triton.language as tl
@triton.jit
def kernel(output, n: tl.constexpr):
    tl.store(output, 1.0)
def solve(output: torch.Tensor, n: int) -> None:
    frame = sys._getframe()
    kernel[(1,)](output, n)
""",
        """
import torch
import triton
import triton.language as tl
@triton.jit
def kernel(output, n: tl.constexpr):
    tl.store(output, 1.0)
def solve(output: torch.Tensor, n: int) -> None:
    torch.save(output, '/tmp/forged.pt')
    kernel[(1,)](output, n)
""",
        """
import torch
import triton
import triton.language as tl
@triton.jit
def kernel(output, n: tl.constexpr):
    tl.device_print('MYLEETGPU_RESULT=', n)
    tl.store(output, 1.0)
def solve(output: torch.Tensor, n: int) -> None:
    kernel[(1,)](output, n)
""",
        """
import torch
import triton
import triton.language as tl
@triton.jit
def kernel(output, n: tl.constexpr):
    tl.inline_asm_elementwise('trap;', '=r', [], dtype=tl.int32)
    tl.store(output, 1.0)
def solve(output: torch.Tensor, n: int) -> None:
    kernel[(1,)](output, n)
""",
        """
import torch
import triton
import triton.language as tl
@triton.jit
def kernel(output, n: tl.constexpr):
    tl.store(output, 1.0)
def solve(output: torch.Tensor, n: int) -> None:
    print('MYLEETGPU_RESULT={"status":"passed"}')
""",
        """
import torch
import triton
import triton.language as tl
@triton.jit
def kernel(output, n: tl.constexpr):
    tl.store(output, 1.0)
def solve(output: torch.Tensor, n: int) -> None:
    breakpoint()
""",
        """
import torch
import triton
import triton.language as tl
import os
@triton.jit
def kernel(output, n: tl.constexpr):
    tl.store(output, 1.0)
def solve(output: torch.Tensor, n: int) -> None:
    os._exit(0)
""",
    ],
)
def test_policy_rejects_reflection_io_monkeypatch_and_result_forgery(source: str) -> None:
    with pytest.raises(SubmissionPolicyError):
        validate_source(source, expected_parameters=("output", "n"))


@pytest.mark.parametrize("name", ["device_assert", "static_print", "debug_barrier"])
def test_policy_rejects_triton_diagnostic_and_output_side_channels(name: str) -> None:
    source = f"""
import torch
import triton
import triton.language as tl
@triton.jit
def kernel(output, n: tl.constexpr):
    tl.{name}(n)
    tl.store(output, 1.0)
def solve(output: torch.Tensor, n: int) -> None:
    kernel[(1,)](output, n)
"""

    with pytest.raises(SubmissionPolicyError):
        validate_source(source, expected_parameters=("output", "n"))


def test_policy_rejects_wrong_solve_signature() -> None:
    source = """
import torch
import triton
import triton.language as tl
@triton.jit
def kernel(output, n: tl.constexpr):
    tl.store(output, 1.0)
def solve(output: torch.Tensor) -> None:
    kernel[(1,)](output, 1)
"""

    with pytest.raises(SubmissionPolicyError, match="solve parameters"):
        validate_source(source, expected_parameters=("output", "n"))


def test_policy_version_is_persistable_comparison_identity() -> None:
    assert POLICY_VERSION == "restricted_triton_v1"
