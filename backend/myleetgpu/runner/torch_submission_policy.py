from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final

POLICY_VERSION: Final = "restricted_torch_v1"

_EXACT_IMPORTS = {("torch", None)}
_RESERVED_NAMES = {"bool", "float", "int", "solve", "torch"}
_SAFE_BUILTINS = {
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
_FORBIDDEN_NODES = (
    ast.Assert,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
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

        self._inside_solve = True
        for argument in node.args.args:
            if argument.arg in _FORBIDDEN_NAMES or argument.arg in _RESERVED_NAMES:
                raise _error(argument, f"reserved parameter name `{argument.arg}` is forbidden")
            if "__" in argument.arg:
                raise _error(argument, "dunder parameter names are forbidden")
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if node.returns is not None:
            self.visit(node.returns)
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
        self.visit(node.annotation)
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


def validate_source(
    source: str,
    *,
    expected_parameters: tuple[str, ...] | None = None,
    filename: str = "source.py",
) -> ast.Module:
    try:
        tree = ast.parse(source, filename=filename, mode="exec", type_comments=True)
    except SyntaxError as error:
        raise TorchSubmissionPolicyError(str(error)) from error
    return _TorchPolicyValidator(expected_parameters).validate(tree)


def load_submission(
    source_path: Path,
    *,
    expected_parameters: tuple[str, ...] | None = None,
) -> ModuleType:
    source = source_path.read_text(encoding="utf-8")
    tree = validate_source(
        source,
        expected_parameters=expected_parameters,
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
    solve = module.__dict__.get("solve")
    if not callable(solve):
        raise TorchSubmissionPolicyError("PyTorch submission policy: solve must be callable")
    return module


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: torch_submission_policy.py /work/source.py", file=sys.stderr)
        return 2
    try:
        load_submission(Path(argv[1]))
    except Exception as error:
        first_line = next(
            (line.strip() for line in str(error).splitlines() if line.strip()), "error"
        )
        print(first_line[:500], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
