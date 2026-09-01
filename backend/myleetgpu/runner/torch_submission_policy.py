from __future__ import annotations

import ast
import builtins
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final

POLICY_VERSION: Final = "restricted_torch_v2"

_EXACT_IMPORTS = {("torch", None)}
_RESERVED_NAMES = {"bool", "float", "int", "solve", "torch"}
_SAFE_BUILTINS = {
    "__build_class__": builtins.__build_class__,
    "__import__": builtins.__import__,
    "bool": bool,
    "float": float,
    "int": int,
}
_FORBIDDEN_NAMES = {
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "print",
    "setattr",
    "super",
    "type",
    "vars",
}
_TORCH_CALLS = {
    "amax",
    "arange",
    "bmm",
    "einsum",
    "exp",
    "logsumexp",
    "matmul",
    "maximum",
    "minimum",
    "repeat_interleave",
    "softmax",
    "sum",
    "where",
}
_TORCH_VALUES = _TORCH_CALLS | {
    "Tensor",
    "bfloat16",
    "bool",
    "float16",
    "float32",
    "float64",
}
_TENSOR_METHODS = {
    "amax",
    "contiguous",
    "expand",
    "exp",
    "flatten",
    "logsumexp",
    "masked_fill",
    "matmul",
    "permute",
    "repeat_interleave",
    "reshape",
    "softmax",
    "squeeze",
    "sum",
    "transpose",
    "unsqueeze",
    "view",
}
_TENSOR_PROPERTIES = {"device", "dtype", "ndim", "shape"}
_SIMPLE_ANNOTATION_NAMES = {"bool", "float", "int"}
_FORBIDDEN_NODES = (
    ast.Assert,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.Delete,
    ast.DictComp,
    ast.For,
    ast.GeneratorExp,
    ast.Global,
    ast.ImportFrom,
    ast.Lambda,
    ast.ListComp,
    ast.Match,
    ast.NamedExpr,
    ast.Nonlocal,
    ast.Raise,
    ast.SetComp,
    ast.Try,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Yield,
    ast.YieldFrom,
)


class TorchSubmissionPolicyError(SyntaxError):
    pass


def _error(node: ast.AST, message: str) -> TorchSubmissionPolicyError:
    line = getattr(node, "lineno", None)
    suffix = f" (line {line})" if line is not None else ""
    return TorchSubmissionPolicyError(f"PyTorch submission policy: {message}{suffix}")


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def _is_literal_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str | int | float | bool | type(None))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        return _is_literal_expression(node.operand)
    if isinstance(node, ast.Tuple | ast.List):
        return all(_is_literal_expression(item) for item in node.elts)
    return False


def _is_immutable_literal_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str | int | float | bool | type(None))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        return _is_immutable_literal_expression(node.operand)
    if isinstance(node, ast.Tuple):
        return all(_is_immutable_literal_expression(item) for item in node.elts)
    return False


class _TorchPolicyValidator(ast.NodeVisitor):
    def __init__(self, expected_parameters: tuple[str, ...] | None):
        self.expected_parameters = expected_parameters
        self.solve: ast.FunctionDef | None = None
        self._inside_solve = False

    def validate(self, tree: ast.Module) -> ast.Module:
        for node in tree.body:
            if isinstance(node, ast.Import):
                self._validate_import(node)
            elif isinstance(node, ast.Assign | ast.AnnAssign):
                self._validate_module_assignment(node)
            elif isinstance(node, ast.FunctionDef) and node.name == "solve":
                if self.solve is not None:
                    raise _error(node, "exactly one solve function is required")
                self.solve = node
            else:
                raise _error(
                    node,
                    "module scope permits only `import torch`, literal constants, and solve",
                )
        if self.solve is None:
            raise TorchSubmissionPolicyError(
                "PyTorch submission policy: solve function is required"
            )
        for node in tree.body:
            if not isinstance(node, ast.Import):
                self.visit(node)
        return tree

    def visit(self, node: ast.AST) -> Any:
        if isinstance(node, _FORBIDDEN_NODES):
            raise _error(node, f"{type(node).__name__} is not allowed")
        return super().visit(node)

    def _validate_import(self, node: ast.Import) -> None:
        if len(node.names) != 1:
            raise _error(node, "imports must contain exactly one approved module")
        alias = node.names[0]
        if (alias.name, alias.asname) not in _EXACT_IMPORTS:
            raise _error(node, "only exact `import torch` is allowed")

    def _validate_module_assignment(self, node: ast.Assign | ast.AnnAssign) -> None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if node.value is None or not _is_immutable_literal_expression(node.value):
            raise _error(node, "module constants must be immutable literal-only expressions")
        for target in targets:
            if not isinstance(target, ast.Name):
                raise _error(target, "module constants must use plain names")
            self._validate_identifier(target, binding=True)

    def _validate_identifier(self, node: ast.Name, *, binding: bool = False) -> None:
        name = node.id
        if "__" in name:
            raise _error(node, "dunder names are forbidden")
        if name in _FORBIDDEN_NAMES:
            raise _error(node, f"name `{name}` is forbidden")
        if binding and name in _RESERVED_NAMES:
            raise _error(node, f"reserved name `{name}` cannot be rebound")

    def visit_Name(self, node: ast.Name) -> None:
        self._validate_identifier(node, binding=isinstance(node.ctx, ast.Store))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") or "__" in node.attr:
            raise _error(node, "private and dunder attributes are forbidden")
        path = _attribute_path(node)
        torch_member = (
            path is not None and len(path) == 2 and path[0] == "torch" and path[1] in _TORCH_VALUES
        )
        tensor_member = node.attr in (_TENSOR_METHODS | _TENSOR_PROPERTIES)
        if not torch_member and not tensor_member:
            rendered = ".".join(path) if path else node.attr
            raise _error(node, f"attribute `{rendered}` is outside the allowlist")
        self.generic_visit(node)

    @staticmethod
    def _validate_annotation(node: ast.AST) -> None:
        if isinstance(node, ast.Name) and node.id in _SIMPLE_ANNOTATION_NAMES:
            return
        if isinstance(node, ast.Constant) and node.value is None:
            return
        if isinstance(node, ast.Attribute) and _attribute_path(node) == ("torch", "Tensor"):
            return
        raise _error(
            node,
            "type annotations must be torch.Tensor, int, float, bool, or None",
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._inside_solve:
            raise _error(node, "nested functions are not allowed")
        if node.name != "solve":
            raise _error(node, "helper functions are not allowed")
        if node.decorator_list:
            raise _error(node, "solve cannot have decorators")
        if node.args.vararg is not None or node.args.kwarg is not None:
            raise _error(node, "variadic function parameters are not allowed")
        if node.args.kwonlyargs or node.args.posonlyargs:
            raise _error(node, "only ordinary positional parameters are allowed")
        if node.args.defaults or node.args.kw_defaults:
            raise _error(node, "function parameter defaults are not allowed")
        parameters = tuple(argument.arg for argument in node.args.args)
        if self.expected_parameters is not None and parameters != self.expected_parameters:
            expected = ", ".join(self.expected_parameters)
            raise _error(node, f"solve parameters must be exactly ({expected})")

        for argument in node.args.args:
            if argument.arg in _FORBIDDEN_NAMES or argument.arg in _RESERVED_NAMES:
                raise _error(argument, f"reserved parameter name `{argument.arg}` is forbidden")
            if "__" in argument.arg:
                raise _error(argument, "dunder parameter names are forbidden")
            if argument.annotation is not None:
                self._validate_annotation(argument.annotation)
        if node.returns is not None:
            self._validate_annotation(node.returns)
        self._inside_solve = True
        for statement in node.body:
            self._visit_solve_statement(statement)
        self._inside_solve = False

    def _visit_solve_statement(self, statement: ast.stmt) -> None:
        if not isinstance(statement, ast.Assign | ast.AnnAssign | ast.Return | ast.If | ast.Pass):
            raise _error(
                statement,
                "solve permits assignments, conditionals, and a returned Tensor only",
            )
        self.visit(statement)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        for statement in (*node.body, *node.orelse):
            self._visit_solve_statement(statement)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._validate_assignment_target(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._validate_assignment_target(node.target)
        self._validate_annotation(node.annotation)
        if node.value is not None:
            self.visit(node.value)

    def _validate_assignment_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self._validate_identifier(target, binding=True)
            return
        if isinstance(target, ast.Tuple | ast.List):
            for item in target.elts:
                self._validate_assignment_target(item)
            return
        raise _error(target, "attribute and subscript assignment are forbidden")

    def visit_Call(self, node: ast.Call) -> None:
        if not self._inside_solve:
            raise _error(node, "calls are not allowed at module scope")
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg == "out" or "__" in keyword.arg:
                raise _error(keyword, "variadic, output, and dunder keywords are forbidden")
        if isinstance(node.func, ast.Name):
            if node.func.id not in _SAFE_BUILTINS:
                raise _error(node, "call is outside the Python allowlist")
            if node.keywords:
                raise _error(node, "Python conversion calls do not accept keyword arguments")
            if any(not _is_literal_expression(argument) for argument in node.args):
                raise _error(node, "Python conversion calls accept only literal values")
        elif isinstance(node.func, ast.Attribute):
            path = _attribute_path(node.func)
            torch_call = (
                path is not None
                and len(path) == 2
                and path[0] == "torch"
                and path[1] in _TORCH_CALLS
            )
            tensor_call = node.func.attr in _TENSOR_METHODS
            if not torch_call and not tensor_call:
                raise _error(node, "call is outside the PyTorch allowlist")
        else:
            raise _error(node, "call is outside the PyTorch allowlist")
        self.generic_visit(node)


class _TorchClassPolicyValidator(_TorchPolicyValidator):
    def __init__(
        self,
        expected_class_name: str | None,
        expected_init_parameters: tuple[str, ...] | None,
        expected_forward_parameters: tuple[str, ...] | None,
    ):
        super().__init__(expected_parameters=None)
        self.expected_class_name = expected_class_name
        self.expected_init_parameters = expected_init_parameters
        self.expected_forward_parameters = expected_forward_parameters
        self.entrypoint_name: str | None = None
        self.state_attributes: set[str] = set()

    def validate(self, tree: ast.Module) -> ast.Module:
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        if len(classes) != 1:
            raise TorchSubmissionPolicyError(
                "PyTorch submission policy: exactly one class entrypoint is required"
            )
        entrypoint = classes[0]
        self.entrypoint_name = entrypoint.name
        if self.expected_class_name is not None and entrypoint.name != self.expected_class_name:
            raise _error(
                entrypoint,
                f"class entrypoint must be named `{self.expected_class_name}`",
            )
        self._validate_class_name(entrypoint)

        for node in tree.body:
            if isinstance(node, ast.Import):
                self._validate_import(node)
            elif isinstance(node, ast.Assign | ast.AnnAssign):
                self._validate_module_assignment(node)
            elif node is entrypoint:
                continue
            else:
                raise _error(
                    node,
                    "module scope permits only `import torch`, literal constants, and one class",
                )
        for node in tree.body:
            if isinstance(node, ast.Assign | ast.AnnAssign):
                self.visit(node)
        self._validate_class(entrypoint)
        return tree

    def _validate_identifier(self, node: ast.Name, *, binding: bool = False) -> None:
        super()._validate_identifier(node, binding=binding)
        if binding and node.id == self.entrypoint_name:
            raise _error(node, f"reserved class name `{node.id}` cannot be rebound")

    @staticmethod
    def _validate_public_name(node: ast.AST, name: str, label: str) -> None:
        if name.startswith("_") or "__" in name:
            raise _error(node, f"{label} must use a public name")
        if name in _FORBIDDEN_NAMES or name in _RESERVED_NAMES:
            raise _error(node, f"reserved {label} `{name}` is forbidden")

    def _validate_class_name(self, node: ast.ClassDef) -> None:
        self._validate_public_name(node, node.name, "class name")

    def _validate_class(self, node: ast.ClassDef) -> None:
        if node.bases:
            raise _error(node, "class inheritance is not allowed")
        if node.decorator_list:
            raise _error(node, "class decorators are not allowed")
        if node.keywords:
            raise _error(node, "class keywords and metaclasses are not allowed")
        if getattr(node, "type_params", ()):
            raise _error(node, "generic classes are not allowed")

        methods: dict[str, ast.FunctionDef] = {}
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef) or statement.name not in {
                "__init__",
                "forward",
            }:
                raise _error(statement, "class body permits only __init__ and forward")
            if statement.name in methods:
                raise _error(statement, f"duplicate {statement.name} method is not allowed")
            methods[statement.name] = statement
        if set(methods) != {"__init__", "forward"}:
            raise _error(node, "class must define exactly __init__ and forward")

        self._validate_init(methods["__init__"])
        self._validate_forward(methods["forward"])

    def _validate_method_signature(
        self,
        node: ast.FunctionDef,
        *,
        expected_parameters: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        if node.decorator_list:
            raise _error(node, f"{node.name} cannot have decorators")
        if getattr(node, "type_params", ()):
            raise _error(node, f"generic {node.name} methods are not allowed")
        if node.args.vararg is not None or node.args.kwarg is not None:
            raise _error(node, "variadic method parameters are not allowed")
        if node.args.kwonlyargs or node.args.posonlyargs:
            raise _error(node, "only ordinary positional method parameters are allowed")
        if node.args.defaults or node.args.kw_defaults:
            raise _error(node, "method parameter defaults are not allowed")

        parameters = tuple(argument.arg for argument in node.args.args)
        if not parameters or parameters[0] != "self":
            raise _error(node, f"{node.name} first parameter must be self")
        if expected_parameters is not None:
            expected = ("self", *expected_parameters)
            if parameters != expected:
                rendered = ", ".join(expected)
                raise _error(node, f"{node.name} parameters must be exactly ({rendered})")

        for index, argument in enumerate(node.args.args):
            if index:
                self._validate_public_name(argument, argument.arg, "parameter name")
            if argument.annotation is not None:
                self._validate_annotation(argument.annotation)
        if node.returns is not None:
            self._validate_annotation(node.returns)
        return parameters

    def _validate_init(self, node: ast.FunctionDef) -> None:
        parameters = self._validate_method_signature(
            node,
            expected_parameters=self.expected_init_parameters,
        )
        constructor_parameters = frozenset(parameters[1:])
        for statement in node.body:
            if isinstance(statement, ast.Pass):
                continue
            if isinstance(statement, ast.Assign):
                if len(statement.targets) != 1:
                    raise _error(statement, "constructor assignments must have one target")
                self._validate_init_assignment(
                    statement.targets[0],
                    statement.value,
                    constructor_parameters,
                )
                continue
            if isinstance(statement, ast.AnnAssign) and statement.value is not None:
                self._validate_annotation(statement.annotation)
                self._validate_init_assignment(
                    statement.target,
                    statement.value,
                    constructor_parameters,
                )
                continue
            raise _error(
                statement,
                "__init__ permits only direct state assignments and pass",
            )

    def _validate_init_assignment(
        self,
        target: ast.AST,
        value: ast.AST,
        constructor_parameters: frozenset[str],
    ) -> None:
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            raise _error(target, "constructor state must be assigned to self attributes")
        self._validate_public_name(target, target.attr, "state attribute")
        if target.attr in self.state_attributes:
            raise _error(target, f"state attribute `{target.attr}` is assigned more than once")
        if not (
            isinstance(value, ast.Name) and value.id in constructor_parameters
        ) and not _is_immutable_literal_expression(value):
            raise _error(
                value,
                "constructor state values must be parameters or immutable literals",
            )
        self.state_attributes.add(target.attr)
        self.visit(value)

    def _validate_forward(self, node: ast.FunctionDef) -> None:
        self._validate_method_signature(
            node,
            expected_parameters=self.expected_forward_parameters,
        )
        self._inside_solve = True
        for statement in node.body:
            self._visit_solve_statement(statement)
        self._inside_solve = False

    def _visit_solve_statement(self, statement: ast.stmt) -> None:
        if not isinstance(statement, ast.Assign | ast.AnnAssign | ast.Return | ast.If | ast.Pass):
            raise _error(
                statement,
                "forward permits assignments, conditionals, and a returned Tensor only",
            )
        self.visit(statement)

    def _validate_assignment_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name) and target.id == "self":
            raise _error(target, "self cannot be rebound")
        super()._validate_assignment_target(target)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        path = _attribute_path(node)
        if path is not None and len(path) == 2 and path[0] == "self":
            if not isinstance(node.ctx, ast.Load):
                raise _error(node, "forward cannot modify class state")
            if path[1] not in self.state_attributes:
                raise _error(node, f"state attribute `{path[1]}` is not initialized")
            self.visit(node.value)
            return
        super().visit_Attribute(node)


def _validate_contract_arguments(
    expected_parameters: tuple[str, ...] | None,
    expected_class_name: str | None,
    expected_init_parameters: tuple[str, ...] | None,
    expected_forward_parameters: tuple[str, ...] | None,
) -> bool:
    class_contract = (
        expected_class_name,
        expected_init_parameters,
        expected_forward_parameters,
    )
    if expected_parameters is not None and any(item is not None for item in class_contract):
        raise TorchSubmissionPolicyError(
            "PyTorch submission policy: function and class contracts are mutually exclusive"
        )
    if any(item is not None for item in class_contract) and not all(
        item is not None for item in class_contract
    ):
        raise TorchSubmissionPolicyError(
            "PyTorch submission policy: class name, init parameters, and forward parameters "
            "must be supplied together"
        )
    return expected_class_name is not None


def _declared_parameters(node: ast.FunctionDef, label: str) -> tuple[str, ...]:
    if node.decorator_list or getattr(node, "type_params", ()):
        raise ValueError(f"PyTorch {label} declaration cannot use decorators or type parameters")
    if node.args.vararg is not None or node.args.kwarg is not None:
        raise ValueError(f"PyTorch {label} declaration cannot use variadic parameters")
    if node.args.kwonlyargs or node.args.posonlyargs:
        raise ValueError(f"PyTorch {label} declaration requires ordinary positional parameters")
    if node.args.defaults or node.args.kw_defaults:
        raise ValueError(f"PyTorch {label} declaration cannot use parameter defaults")
    return tuple(argument.arg for argument in node.args.args)


def submission_contract_from_declaration(symbol: str, declaration: str) -> dict[str, Any]:
    """Parse a trusted manifest declaration without evaluating its annotations or body."""

    try:
        tree = ast.parse(declaration, filename="<torch-signature>", mode="exec")
    except SyntaxError as error:
        raise ValueError(f"invalid PyTorch signature declaration: {error}") from error
    if len(tree.body) != 1:
        raise ValueError("PyTorch signature declaration must contain exactly one entrypoint")

    entrypoint = tree.body[0]
    if isinstance(entrypoint, ast.FunctionDef):
        if entrypoint.name != symbol or symbol != "solve":
            raise ValueError("PyTorch function signature must declare the symbol `solve`")
        return {
            "kind": "function",
            "symbol": symbol,
            "parameters": _declared_parameters(entrypoint, "solve"),
        }

    if not isinstance(entrypoint, ast.ClassDef) or entrypoint.name != symbol:
        raise ValueError("PyTorch signature declaration does not match its symbol")
    if entrypoint.bases or entrypoint.decorator_list or entrypoint.keywords:
        raise ValueError("PyTorch class signature must be an ordinary class declaration")
    if getattr(entrypoint, "type_params", ()):
        raise ValueError("PyTorch class signature cannot use type parameters")

    methods = {
        item.name: item
        for item in entrypoint.body
        if isinstance(item, ast.FunctionDef) and item.name in {"__init__", "forward"}
    }
    if len(entrypoint.body) != 2 or set(methods) != {"__init__", "forward"}:
        raise ValueError("PyTorch class signature must declare exactly __init__ and forward")
    init_parameters = _declared_parameters(methods["__init__"], "__init__")
    forward_parameters = _declared_parameters(methods["forward"], "forward")
    if not init_parameters or init_parameters[0] != "self":
        raise ValueError("PyTorch __init__ declaration must start with self")
    if not forward_parameters or forward_parameters[0] != "self":
        raise ValueError("PyTorch forward declaration must start with self")
    return {
        "kind": "class",
        "symbol": symbol,
        "init_parameters": init_parameters[1:],
        "forward_parameters": forward_parameters[1:],
    }


def _contract_kwargs_from_json(payload: str) -> dict[str, Any]:
    try:
        contract = json.loads(payload)
    except (json.JSONDecodeError, TypeError) as error:
        raise TorchSubmissionPolicyError(
            "PyTorch submission policy: invalid compilation contract"
        ) from error
    if not isinstance(contract, dict):
        raise TorchSubmissionPolicyError(
            "PyTorch submission policy: compilation contract must be an object"
        )

    def parameters(key: str) -> tuple[str, ...]:
        value = contract.get(key)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise TorchSubmissionPolicyError(
                f"PyTorch submission policy: compilation contract `{key}` must be a string list"
            )
        return tuple(value)

    kind = contract.get("kind")
    symbol = contract.get("symbol")
    if not isinstance(symbol, str):
        raise TorchSubmissionPolicyError(
            "PyTorch submission policy: compilation contract symbol must be a string"
        )
    if kind == "function" and set(contract) == {"kind", "symbol", "parameters"}:
        if symbol != "solve":
            raise TorchSubmissionPolicyError(
                "PyTorch submission policy: function contract symbol must be `solve`"
            )
        return {"expected_parameters": parameters("parameters")}
    if kind == "class" and set(contract) == {
        "kind",
        "symbol",
        "init_parameters",
        "forward_parameters",
    }:
        return {
            "expected_class_name": symbol,
            "expected_init_parameters": parameters("init_parameters"),
            "expected_forward_parameters": parameters("forward_parameters"),
        }
    raise TorchSubmissionPolicyError(
        "PyTorch submission policy: compilation contract shape is invalid"
    )


def validate_source(
    source: str,
    *,
    expected_parameters: tuple[str, ...] | None = None,
    expected_class_name: str | None = None,
    expected_init_parameters: tuple[str, ...] | None = None,
    expected_forward_parameters: tuple[str, ...] | None = None,
    filename: str = "source.py",
) -> ast.Module:
    class_contract = _validate_contract_arguments(
        expected_parameters,
        expected_class_name,
        expected_init_parameters,
        expected_forward_parameters,
    )
    try:
        tree = ast.parse(source, filename=filename, mode="exec", type_comments=True)
    except SyntaxError as error:
        raise TorchSubmissionPolicyError(str(error)) from error
    if class_contract:
        return _TorchClassPolicyValidator(
            expected_class_name,
            expected_init_parameters,
            expected_forward_parameters,
        ).validate(tree)
    if expected_parameters is None:
        has_solve = any(
            isinstance(node, ast.FunctionDef) and node.name == "solve" for node in tree.body
        )
        has_class = any(isinstance(node, ast.ClassDef) for node in tree.body)
        if has_class and not has_solve:
            return _TorchClassPolicyValidator(None, None, None).validate(tree)
        if has_class and has_solve:
            raise TorchSubmissionPolicyError(
                "PyTorch submission policy: exactly one function or class entrypoint is required"
            )
    return _TorchPolicyValidator(expected_parameters).validate(tree)


def load_submission(
    source_path: Path,
    *,
    expected_parameters: tuple[str, ...] | None = None,
    expected_class_name: str | None = None,
    expected_init_parameters: tuple[str, ...] | None = None,
    expected_forward_parameters: tuple[str, ...] | None = None,
) -> ModuleType:
    source = source_path.read_text(encoding="utf-8")
    tree = validate_source(
        source,
        expected_parameters=expected_parameters,
        expected_class_name=expected_class_name,
        expected_init_parameters=expected_init_parameters,
        expected_forward_parameters=expected_forward_parameters,
        filename="source.py",
    )
    executable = ast.Module(
        body=[node for node in tree.body if not isinstance(node, ast.Import)],
        type_ignores=tree.type_ignores,
    )
    ast.fix_missing_locations(executable)

    import torch

    module = ModuleType("_myleetgpu_torch_submission")
    module.__file__ = str(source_path)
    module.__dict__.update(
        {
            "__builtins__": dict(_SAFE_BUILTINS),
            "torch": torch,
        }
    )
    code = compile(executable, "source.py", "exec", dont_inherit=True, optimize=2)
    exec(code, module.__dict__, module.__dict__)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if classes:
        class_name = classes[0].name
        if not isinstance(module.__dict__.get(class_name), type):
            raise TorchSubmissionPolicyError(
                f"PyTorch submission policy: class `{class_name}` must be constructible"
            )
    else:
        solve = module.__dict__.get("solve")
        if not callable(solve):
            raise TorchSubmissionPolicyError("PyTorch submission policy: solve must be callable")
    return module


def _main(argv: list[str]) -> int:
    if len(argv) not in {2, 3}:
        print(
            "usage: torch_submission_policy.py /work/source.py [contract-json]",
            file=sys.stderr,
        )
        return 2
    try:
        contract_kwargs = _contract_kwargs_from_json(argv[2]) if len(argv) == 3 else {}
        load_submission(Path(argv[1]), **contract_kwargs)
    except Exception as error:
        first_line = next(
            (line.strip() for line in str(error).splitlines() if line.strip()), "error"
        )
        print(first_line[:500], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
