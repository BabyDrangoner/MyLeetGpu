from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
from myleetgpu.runner.torch_submission_policy import (
    POLICY_VERSION,
    TorchSubmissionPolicyError,
    load_submission,
    validate_source,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = ("query", "key", "value", "attention_mask")


@pytest.mark.parametrize(
    "relative_path",
    [
        "problems/multi-head-attention/torch/starter.py",
        "problems/grouped-query-attention/torch/starter.py",
    ],
)
def test_builtin_torch_starters_are_accepted(relative_path: str) -> None:
    source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")

    validate_source(source, expected_parameters=PARAMETERS)


@pytest.mark.parametrize(
    "source",
    [
        """
import os
import torch
def solve(query, key, value, attention_mask):
    return query
""",
        """
from torch import matmul
def solve(query, key, value, attention_mask):
    return matmul(query, value)
""",
        """
import torch
def helper(query):
    return query
def solve(query, key, value, attention_mask):
    return helper(query)
""",
        """
import torch
def solve(query, key, value, attention_mask):
    print('MYLEETGPU_RESULT={"status":"passed"}')
    return query
""",
        """
import torch
def solve(query, key, value, attention_mask):
    payload = open('/work/platform.py').read()
    return query
""",
        """
import torch
def solve(query, key, value, attention_mask):
    fn = getattr(torch, 'matmul')
    return fn(query, value)
""",
        """
import torch
def solve(query, key, value, attention_mask):
    torch.matmul = lambda left, right: query
    return query
""",
        """
import torch
def solve(query, key, value, attention_mask):
    return torch.nn.functional.scaled_dot_product_attention(query, key, value)
""",
        """
import torch
def solve(query, key, value, attention_mask):
    torch.cuda.synchronize()
    return query
""",
        """
import torch
def solve(query, key, value, attention_mask):
    query.add_(1)
    return query
""",
        """
import torch
def solve(query, key, value, attention_mask):
    return torch.matmul(query, key.transpose(-2, -1), out=query)
""",
        """
import torch
def solve(query, key, value, attention_mask):
    return query.__class__
""",
        """
import torch
def solve(query, key, value, attention_mask):
    module = __import__('os')
    return query
""",
        """
import torch
def solve(query, key, value, attention_mask):
    return torch.ops.aten.matmul.default(query, key)
""",
        """
import torch
def solve(query, key, value, attention_mask):
    mutated = torch.matmul(query, key, out=query)
    return query
""",
        """
import torch
def solve(query, key, value, attention_mask):
    query[0] = value[0]
    return query
""",
        """
import torch
CALL_COUNT = [0]
def solve(query, key, value, attention_mask):
    if True:
        CALL_COUNT[0] += 1
    return query
""",
        """
import torch
def solve(query, key, value, attention_mask):
    if query.shape[0] > 0:
        query *= 2
        query /= 2
    return query
""",
        """
import torch
def solve(query, key, value, attention_mask):
    if False:
        import os
    return query
""",
        """
import torch
def solve(query, key, value, attention_mask):
    if True:
        query.add_(1)
    return query
""",
        """
import torch
def solve(query, key, value, attention_mask):
    float = torch.Tensor
    return float(100)
""",
        """
import torch
def solve(query, key, value, attention_mask):
    scale = float(x=1)
    return query * scale
""",
    ],
)
def test_torch_policy_rejects_escape_hatches_and_attention_shortcuts(source: str) -> None:
    with pytest.raises(TorchSubmissionPolicyError):
        validate_source(source, expected_parameters=PARAMETERS)


def test_torch_policy_accepts_primitive_attention_composition() -> None:
    source = """
import torch

def solve(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    groups = query.shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(groups, dim=1)
    expanded_value = value.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query, expanded_key.transpose(-2, -1))
    scores = scores * (query.shape[-1] ** -0.5)
    scores = scores.masked_fill(~attention_mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, expanded_value)
"""

    validate_source(source, expected_parameters=PARAMETERS)


def test_torch_policy_accepts_functional_repeat_interleave_for_gqa() -> None:
    source = """
import torch

def solve(query, key, value, attention_mask):
    groups = query.shape[1] // key.shape[1]
    expanded_key = torch.repeat_interleave(key, groups, dim=1)
    return torch.matmul(query, expanded_key.transpose(-2, -1))
"""

    validate_source(source, expected_parameters=PARAMETERS)


def test_torch_policy_recursively_accepts_only_safe_conditional_statements() -> None:
    source = """
import torch

def solve(query, key, value, attention_mask):
    if query.shape[1] == key.shape[1]:
        grouped_key = key
    else:
        groups = query.shape[1] // key.shape[1]
        grouped_key = key.repeat_interleave(groups, dim=1)
    return torch.matmul(query, grouped_key.transpose(-2, -1))
"""

    validate_source(source, expected_parameters=PARAMETERS)


def test_torch_policy_loads_submission_with_isolated_globals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_torch = ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    source_path = tmp_path / "source.py"
    source_path.write_text(
        "import torch\n"
        "SCALE = 0.5\n"
        "def solve(query, key, value, attention_mask):\n"
        "    return query\n",
        encoding="utf-8",
    )

    module = load_submission(source_path, expected_parameters=PARAMETERS)
    sentinel = object()

    assert module.solve(sentinel, None, None, None) is sentinel  # type: ignore[attr-defined]
    assert module.__dict__["torch"] is fake_torch
    assert module.__dict__["SCALE"] == 0.5
    assert set(module.__dict__["__builtins__"]) == {"bool", "float", "int"}


def test_torch_policy_rejects_wrong_solve_signature() -> None:
    source = """
import torch
def solve(query, key, value):
    return query
"""

    with pytest.raises(TorchSubmissionPolicyError, match="solve parameters"):
        validate_source(source, expected_parameters=PARAMETERS)


def test_torch_policy_version_is_persistable_comparison_identity() -> None:
    assert POLICY_VERSION == "restricted_torch_v1"
