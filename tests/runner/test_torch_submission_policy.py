from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from myleetgpu.runner.torch_submission_policy import (
    POLICY_VERSION,
    TorchSubmissionPolicyError,
    _main,
    load_submission,
    submission_contract_from_declaration,
    validate_source,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = ("query", "key", "value", "attention_mask")
MHA_CLASS_NAME = "MultiHeadAttention"
MHA_INIT_PARAMETERS = ("numHeads", "qWeight", "kWeight", "vWeight", "outputWeight")
GQA_CLASS_NAME = "GroupedQueryAttention"
GQA_INIT_PARAMETERS = (
    "numQueryHeads",
    "numKeyValueHeads",
    "qWeight",
    "kWeight",
    "vWeight",
    "outputWeight",
)
ATTENTION_FORWARD_PARAMETERS = ("X", "isCasual")


@pytest.mark.parametrize(
    ("relative_path", "class_name", "init_parameters"),
    [
        (
            "problems/multi-head-attention/torch/starter.py",
            MHA_CLASS_NAME,
            MHA_INIT_PARAMETERS,
        ),
        (
            "problems/grouped-query-attention/torch/starter.py",
            GQA_CLASS_NAME,
            GQA_INIT_PARAMETERS,
        ),
    ],
)
def test_builtin_torch_starters_are_accepted(
    relative_path: str,
    class_name: str,
    init_parameters: tuple[str, ...],
) -> None:
    source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")

    validate_source(
        source,
        expected_class_name=class_name,
        expected_init_parameters=init_parameters,
        expected_forward_parameters=ATTENTION_FORWARD_PARAMETERS,
    )


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


VALID_MHA_CLASS_SOURCE = """
import torch

class MultiHeadAttention:
    def __init__(
        self,
        numHeads: int,
        qWeight: torch.Tensor,
        kWeight: torch.Tensor,
        vWeight: torch.Tensor,
        outputWeight: torch.Tensor,
    ) -> None:
        self.numHeads = numHeads
        self.qWeight = qWeight
        self.kWeight = kWeight
        self.vWeight = vWeight
        self.outputWeight = outputWeight
        self.scale = 0.5

    def forward(self, X: torch.Tensor, isCasual: bool) -> torch.Tensor:
        batch, sequence, hidden = X.shape
        headSize = hidden // self.numHeads
        query = X.matmul(self.qWeight)
        query = query.reshape(batch, sequence, self.numHeads, headSize).transpose(1, 2)
        key = X.matmul(self.kWeight)
        key = key.reshape(batch, sequence, self.numHeads, headSize).transpose(1, 2)
        value = X.matmul(self.vWeight)
        value = value.reshape(batch, sequence, self.numHeads, headSize).transpose(1, 2)
        scores = query.matmul(key.transpose(-2, -1)) * self.scale
        if isCasual:
            positions = torch.arange(sequence, device=X.device)
            causalMask = positions.unsqueeze(1) >= positions.unsqueeze(0)
            scores = scores.masked_fill(~causalMask, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        context = probabilities.matmul(value)
        context = context.transpose(1, 2).contiguous().reshape(batch, sequence, hidden)
        return context.matmul(self.outputWeight)
"""


def validate_mha_class(source: str) -> None:
    validate_source(
        source,
        expected_class_name=MHA_CLASS_NAME,
        expected_init_parameters=MHA_INIT_PARAMETERS,
        expected_forward_parameters=ATTENTION_FORWARD_PARAMETERS,
    )


def test_torch_policy_accepts_restricted_class_attention_composition() -> None:
    validate_mha_class(VALID_MHA_CLASS_SOURCE)


def test_torch_policy_accepts_exact_gqa_class_contract() -> None:
    source = """
import torch

class GroupedQueryAttention:
    def __init__(
        self,
        numQueryHeads,
        numKeyValueHeads,
        qWeight,
        kWeight,
        vWeight,
        outputWeight,
    ):
        self.numQueryHeads = numQueryHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.qWeight = qWeight
        self.kWeight = kWeight
        self.vWeight = vWeight
        self.outputWeight = outputWeight

    def forward(self, X, isCasual):
        if isCasual:
            output = X.matmul(self.qWeight)
        else:
            output = X.matmul(self.outputWeight)
        return output
"""

    validate_source(
        source,
        expected_class_name=GQA_CLASS_NAME,
        expected_init_parameters=GQA_INIT_PARAMETERS,
        expected_forward_parameters=ATTENTION_FORWARD_PARAMETERS,
    )


def test_torch_policy_auto_detects_a_single_class_for_compile_checks() -> None:
    validate_source(VALID_MHA_CLASS_SOURCE)


@pytest.mark.parametrize(
    "source",
    [
        VALID_MHA_CLASS_SOURCE.replace(
            "class MultiHeadAttention:",
            "class MultiHeadAttention(object):",
        ),
        VALID_MHA_CLASS_SOURCE.replace(
            "class MultiHeadAttention:",
            "@torch.compile\nclass MultiHeadAttention:",
        ),
        VALID_MHA_CLASS_SOURCE.replace(
            "class MultiHeadAttention:",
            "class MultiHeadAttention(metaclass=type):",
        ),
        VALID_MHA_CLASS_SOURCE.replace(
            "class MultiHeadAttention:\n    def __init__",
            "class MultiHeadAttention:\n    cache = 0\n    def __init__",
        ),
        VALID_MHA_CLASS_SOURCE.replace(
            "    def forward(self, X: torch.Tensor, isCasual: bool) -> torch.Tensor:",
            "    def helper(self, X):\n        return X\n\n"
            "    def forward(self, X: torch.Tensor, isCasual: bool) -> torch.Tensor:",
        ),
        VALID_MHA_CLASS_SOURCE.replace(
            "        self.numHeads = numHeads",
            "        localHeads = numHeads\n        self.numHeads = localHeads",
        ),
        VALID_MHA_CLASS_SOURCE.replace(
            "        self.qWeight = qWeight",
            "        self.qWeight = qWeight.contiguous()",
        ),
        VALID_MHA_CLASS_SOURCE.replace(
            "        self.qWeight = qWeight",
            "        self._qWeight = qWeight",
        ),
        VALID_MHA_CLASS_SOURCE.replace(
            "        self.qWeight = qWeight",
            "        self.qWeight = qWeight\n        self.qWeight = kWeight",
        ),
        VALID_MHA_CLASS_SOURCE.replace(
            "        batch, sequence, hidden = X.shape",
            "        self.qWeight = X\n        batch, sequence, hidden = X.shape",
        ),
        VALID_MHA_CLASS_SOURCE.replace(
            "        batch, sequence, hidden = X.shape",
            "        cached = self.cache\n        batch, sequence, hidden = X.shape",
        ),
        VALID_MHA_CLASS_SOURCE.replace(
            "        batch, sequence, hidden = X.shape",
            "        self = X\n        batch, sequence, hidden = X.shape",
        ),
        VALID_MHA_CLASS_SOURCE.replace(
            "        batch, sequence, hidden = X.shape",
            "        metadata = self.__dict__\n        batch, sequence, hidden = X.shape",
        ),
    ],
)
def test_torch_policy_rejects_unsafe_class_shapes_and_state(source: str) -> None:
    with pytest.raises(TorchSubmissionPolicyError):
        validate_mha_class(source)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "class MultiHeadAttention:",
            "class OtherAttention:",
            "class entrypoint",
        ),
        (
            "        numHeads: int,\n        qWeight: torch.Tensor,",
            "        qWeight: torch.Tensor,\n        numHeads: int,",
            "__init__ parameters",
        ),
        (
            "def forward(self, X: torch.Tensor, isCasual: bool)",
            "def forward(self, X: torch.Tensor, causal: bool)",
            "forward parameters",
        ),
    ],
)
def test_torch_policy_enforces_exact_class_contract(
    old: str,
    new: str,
    message: str,
) -> None:
    with pytest.raises(TorchSubmissionPolicyError, match=message):
        validate_mha_class(VALID_MHA_CLASS_SOURCE.replace(old, new))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("numHeads: int", "numHeads: 2 ** 1000000000"),
        (
            "def forward(self, X: torch.Tensor, isCasual: bool) -> torch.Tensor:",
            "def forward(self, X: [torch.Tensor] * 1000000000, isCasual: bool) -> torch.Tensor:",
        ),
        (
            "def forward(self, X: torch.Tensor, isCasual: bool) -> torch.Tensor:",
            "def forward(self, X: torch.Tensor, isCasual: bool) -> (torch.Tensor,) * 1000000000:",
        ),
        (
            "        batch, sequence, hidden = X.shape",
            "        batch: 2 ** 1000000000 = X.shape[0]\n"
            "        sequence = X.shape[1]\n"
            "        hidden = X.shape[2]",
        ),
    ],
)
def test_torch_policy_rejects_executable_annotation_expressions(
    old: str,
    new: str,
) -> None:
    with pytest.raises(TorchSubmissionPolicyError, match="type annotations"):
        validate_mha_class(VALID_MHA_CLASS_SOURCE.replace(old, new))


def test_torch_policy_rejects_executable_annotations_on_solve() -> None:
    source = """
import torch
def solve(query: 2 ** 1000000000, key, value, attention_mask):
    return query
"""

    with pytest.raises(TorchSubmissionPolicyError, match="type annotations"):
        validate_source(source, expected_parameters=PARAMETERS)


def test_torch_policy_rejects_mixed_or_partial_entrypoint_contracts() -> None:
    with pytest.raises(TorchSubmissionPolicyError, match="mutually exclusive"):
        validate_source(
            VALID_MHA_CLASS_SOURCE,
            expected_parameters=PARAMETERS,
            expected_class_name=MHA_CLASS_NAME,
            expected_init_parameters=MHA_INIT_PARAMETERS,
            expected_forward_parameters=ATTENTION_FORWARD_PARAMETERS,
        )
    with pytest.raises(TorchSubmissionPolicyError, match="must be supplied together"):
        validate_source(
            VALID_MHA_CLASS_SOURCE,
            expected_class_name=MHA_CLASS_NAME,
        )


def test_torch_policy_rejects_class_when_function_contract_is_requested() -> None:
    with pytest.raises(TorchSubmissionPolicyError):
        validate_source(VALID_MHA_CLASS_SOURCE, expected_parameters=PARAMETERS)


def test_torch_policy_rejects_ambiguous_auto_detected_entrypoints() -> None:
    source = (
        VALID_MHA_CLASS_SOURCE
        + "\ndef solve(query, key, value, attention_mask):\n    return query\n"
    )

    with pytest.raises(TorchSubmissionPolicyError, match="exactly one"):
        validate_source(source)


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
    assert set(module.__dict__["__builtins__"]) == {
        "__build_class__",
        "__import__",
        "bool",
        "float",
        "int",
    }


def test_torch_policy_loads_class_with_injected_constructor_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_torch = ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    source_path = tmp_path / "source.py"
    source_path.write_text(
        "import torch\n"
        "class MultiHeadAttention:\n"
        "    def __init__(self, numHeads, qWeight, kWeight, vWeight, outputWeight):\n"
        "        self.numHeads = numHeads\n"
        "        self.qWeight = qWeight\n"
        "        self.kWeight = kWeight\n"
        "        self.vWeight = vWeight\n"
        "        self.outputWeight = outputWeight\n"
        "    def forward(self, X, isCasual):\n"
        "        if isCasual:\n"
        "            result = X\n"
        "        else:\n"
        "            result = X\n"
        "        return result\n",
        encoding="utf-8",
    )

    module = load_submission(
        source_path,
        expected_class_name=MHA_CLASS_NAME,
        expected_init_parameters=MHA_INIT_PARAMETERS,
        expected_forward_parameters=ATTENTION_FORWARD_PARAMETERS,
    )
    attention_type = module.__dict__[MHA_CLASS_NAME]
    weights = [object() for _ in range(4)]
    attention = attention_type(8, *weights)
    sentinel = object()

    assert attention.forward(sentinel, True) is sentinel
    assert attention.numHeads == 8
    assert attention.qWeight is weights[0]
    assert attention.outputWeight is weights[3]
    assert module.__dict__["torch"] is fake_torch
    assert module.__dict__["__name__"] == "_myleetgpu_torch_submission"
    assert set(module.__dict__["__builtins__"]) == {
        "__build_class__",
        "__import__",
        "bool",
        "float",
        "int",
    }


def test_torch_policy_class_can_run_causal_masked_fill_with_real_torch() -> None:
    torch = pytest.importorskip("torch")
    source_path = PROJECT_ROOT / "problems/multi-head-attention/torch/starter.py"
    module = load_submission(
        source_path,
        expected_class_name=MHA_CLASS_NAME,
        expected_init_parameters=MHA_INIT_PARAMETERS,
        expected_forward_parameters=ATTENTION_FORWARD_PARAMETERS,
    )
    attention_type = module.__dict__[MHA_CLASS_NAME]
    identity = torch.eye(4, dtype=torch.float32)
    attention = attention_type(2, identity, identity, identity, identity)
    inputs = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) / 12.0

    output = attention.forward(inputs, True)

    assert tuple(output.shape) == (1, 3, 4)
    assert bool(torch.all(torch.isfinite(output)))


@pytest.mark.parametrize("dunder_builtin", ["__build_class__", "__import__"])
def test_runtime_only_dunder_builtins_remain_inaccessible_to_source(
    dunder_builtin: str,
) -> None:
    source = (
        "import torch\n"
        "def solve(query, key, value, attention_mask):\n"
        f"    leaked = {dunder_builtin}('os')\n"
        "    return query\n"
    )

    with pytest.raises(TorchSubmissionPolicyError, match="dunder"):
        validate_source(source, expected_parameters=PARAMETERS)


def test_torch_policy_rejects_wrong_solve_signature() -> None:
    source = """
import torch
def solve(query, key, value):
    return query
"""

    with pytest.raises(TorchSubmissionPolicyError, match="solve parameters"):
        validate_source(source, expected_parameters=PARAMETERS)


def test_manifest_signature_contract_supports_classes_and_legacy_solve() -> None:
    class_contract = submission_contract_from_declaration(
        MHA_CLASS_NAME,
        """
class MultiHeadAttention:
    def __init__(self, numHeads: int, qWeight: torch.Tensor): ...
    def forward(self, X: torch.Tensor, isCasual: bool) -> torch.Tensor: ...
""",
    )
    solve_contract = submission_contract_from_declaration(
        "solve",
        "def solve(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, "
        "attention_mask: torch.Tensor) -> torch.Tensor: ...",
    )

    assert class_contract == {
        "kind": "class",
        "symbol": MHA_CLASS_NAME,
        "init_parameters": ("numHeads", "qWeight"),
        "forward_parameters": ATTENTION_FORWARD_PARAMETERS,
    }
    assert solve_contract == {
        "kind": "function",
        "symbol": "solve",
        "parameters": PARAMETERS,
    }


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("class MultiHeadAttention:", "class WrongAttention:"),
        ("numHeads: int,", "wrongHeads: int,"),
        ("isCasual: bool", "causal: bool"),
    ],
)
def test_compile_cli_rejects_source_that_mismatches_manifest_contract(
    tmp_path: Path,
    old: str,
    new: str,
) -> None:
    source_path = tmp_path / "source.py"
    source_path.write_text(VALID_MHA_CLASS_SOURCE.replace(old, new), encoding="utf-8")
    declaration = """
class MultiHeadAttention:
    def __init__(
        self,
        numHeads: int,
        qWeight: torch.Tensor,
        kWeight: torch.Tensor,
        vWeight: torch.Tensor,
        outputWeight: torch.Tensor,
    ): ...
    def forward(self, X: torch.Tensor, isCasual: bool) -> torch.Tensor: ...
"""
    contract = submission_contract_from_declaration(MHA_CLASS_NAME, declaration)

    assert _main(["torch_submission_policy.py", str(source_path), json.dumps(contract)]) == 1


def test_torch_policy_version_is_persistable_comparison_identity() -> None:
    assert POLICY_VERSION == "restricted_torch_v2"
