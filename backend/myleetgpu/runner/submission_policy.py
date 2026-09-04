from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final

POLICY_VERSION: Final = "restricted_triton_v2"

_EXACT_IMPORTS = {
    ("torch", None),
    ("triton", None),
    ("triton.language", "tl"),
}
_RESERVED_NAMES = {"torch", "triton", "tl", "solve"}
_SAFE_BUILTINS = {
    "abs": abs,
    "bool": bool,
    "float": float,
    "int": int,
    "max": max,
    "min": min,
    "range": range,
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
    "locals",
    "memoryview",
    "open",
    "print",
    "setattr",
    "super",
    "type",
    "vars",
}
_TRITON_HOST_CALLS = {"cdiv", "next_power_of_2"}
_TL_CALLS = {
    "abs",
    "advance",
    "arange",
    "argmax",
    "argmin",
    "atomic_add",
    "atomic_and",
    "atomic_cas",
    "atomic_max",
    "atomic_min",
    "atomic_or",
    "atomic_xchg",
    "atomic_xor",
    "broadcast",
    "broadcast_to",
    "cat",
    "ceil",
    "clamp",
    "cos",
    "cdiv",
    "cumsum",
    "dot",
    "erf",
    "exp",
    "exp2",
    "expand_dims",
    "flip",
    "floor",
    "full",
    "interleave",
    "join",
    "load",
    "log",
    "log2",
    "make_block_ptr",
    "max",
    "max_contiguous",
    "maximum",
    "min",
    "minimum",
    "multiple_of",
    "num_programs",
    "permute",
    "program_id",
    "ravel",
    "reduce",
    "reshape",
    "rsqrt",
    "sigmoid",
    "sin",
    "softmax",
    "sort",
    "sqrt",
    "static_range",
    "store",
    "sum",
    "trans",
    "view",
    "where",
    "zeros",
    "zeros_like",
}
_TL_VALUES = _TL_CALLS | {
    "bfloat16",
    "const",
    "constexpr",
    "float16",
    "float32",
    "float64",
    "float8e4nv",
    "float8e5",
    "int1",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
}
_FORBIDDEN_NODES = (
    ast.Assert,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.ListComp,
    ast.Match,
    ast.NamedExpr,
    ast.Nonlocal,
    ast.Raise,
    ast.SetComp,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Yield,
    ast.YieldFrom,
)


class SubmissionPolicyError(SyntaxError):
    pass


def _error(node: ast.AST, message: str) -> SubmissionPolicyError:
    line = getattr(node, "lineno", None)
    suffix = f" (line {line})" if line is not None else ""
    return SubmissionPolicyError(f"Triton submission policy: {message}{suffix}")


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


def _is_constant_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str | int | float | bool | type(None))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub | ast.Invert):
        return _is_constant_expression(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op,
        ast.Add | ast.Sub | ast.Mult | ast.Div | ast.FloorDiv | ast.Mod | ast.Pow,
    ):
        return _is_constant_expression(node.left) and _is_constant_expression(node.right)
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return all(_is_constant_expression(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None and _is_constant_expression(key) and _is_constant_expression(value)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    return False


class _PolicyValidator(ast.NodeVisitor):
    def __init__(self, expected_parameters: tuple[str, ...] | None):
        self.expected_parameters = expected_parameters
        self.jit_functions: set[str] = set()
        self.solve: ast.FunctionDef | None = None
        self._function_kind: str | None = None

    def validate(self, tree: ast.Module) -> ast.Module:
        for node in tree.body:
            if isinstance(node, ast.Import):
                self._validate_import(node)
            elif isinstance(node, ast.Assign | ast.AnnAssign):
                self._validate_module_assignment(node)
            elif isinstance(node, ast.FunctionDef):
                if node.name == "solve":
                    if self.solve is not None:
                        raise _error(node, "exactly one solve function is required")
                    self.solve = node
                else:
                    self.jit_functions.add(node.name)
            else:
                raise _error(
                    node,
                    "module scope permits only exact imports, literal constants, "
                    "and function definitions",
                )

        if self.solve is None:
            raise SubmissionPolicyError("Triton submission policy: solve function is required")
        if not self.jit_functions:
            raise SubmissionPolicyError(
                "Triton submission policy: at least one @triton.jit kernel is required"
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
            raise _error(
                node,
                "only `import torch`, `import triton`, and "
                "`import triton.language as tl` are allowed",
            )

    def _validate_module_assignment(self, node: ast.Assign | ast.AnnAssign) -> None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None or not _is_constant_expression(value):
            raise _error(node, "module constants must be literal-only expressions")
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
        allowed = False
        if path and len(path) == 2:
            root, member = path
            allowed = (
                (root == "tl" and member in _TL_VALUES)
                or (root == "torch" and member == "Tensor")
                or (root == "triton" and member in {"jit", *_TRITON_HOST_CALLS})
                or (self._function_kind == "solve" and root == "output" and member == "zero_")
            )
        if (
            not allowed
            and self._function_kind == "kernel"
            and node.attr == "to"
            and (path is None or path[0] not in {"torch", "triton", "tl"})
        ):
            allowed = True
        if not allowed:
            rendered = ".".join(path) if path else node.attr
            raise _error(node, f"attribute `{rendered}` is outside the allowlist")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._function_kind is not None:
            raise _error(node, "nested functions are not allowed")
        if "__" in node.name:
            raise _error(node, "dunder function names are forbidden")
        if node.name in _FORBIDDEN_NAMES or (node.name in _RESERVED_NAMES and node.name != "solve"):
            raise _error(node, f"reserved function name `{node.name}` is forbidden")
        if node.args.vararg is not None or node.args.kwarg is not None:
            raise _error(node, "variadic function parameters are not allowed")
        if node.args.kwonlyargs or node.args.posonlyargs:
            raise _error(node, "only ordinary positional parameters are allowed")
        if node.args.defaults or node.args.kw_defaults:
            raise _error(node, "function parameter defaults are not allowed")

        if node.name == "solve":
            if node.decorator_list:
                raise _error(node, "solve cannot have decorators")
            parameters = tuple(argument.arg for argument in node.args.args)
            if self.expected_parameters is not None and parameters != self.expected_parameters:
                expected = ", ".join(self.expected_parameters)
                raise _error(node, f"solve parameters must be exactly ({expected})")
            kind = "solve"
        else:
            if len(node.decorator_list) != 1 or _attribute_path(node.decorator_list[0]) != (
                "triton",
                "jit",
            ):
                raise _error(node, "every helper/kernel function must use exactly @triton.jit")
            kind = "kernel"

        previous = self._function_kind
        self._function_kind = kind
        for argument in node.args.args:
            if "__" in argument.arg:
                raise _error(argument, "dunder parameter names are forbidden")
            if argument.arg in _FORBIDDEN_NAMES or argument.arg in _RESERVED_NAMES:
                raise _error(argument, f"reserved parameter name `{argument.arg}` is forbidden")
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if node.returns is not None:
            self.visit(node.returns)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for statement in node.body:
            self._validate_statement(statement)
            self.visit(statement)
        self._function_kind = previous

    def _validate_statement(self, node: ast.stmt) -> None:
        if self._function_kind == "solve":
            if not isinstance(node, ast.Assign | ast.AnnAssign | ast.Expr | ast.Return | ast.Pass):
                raise _error(node, "solve is a straight-line launcher; control flow is not allowed")
            if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Call):
                raise _error(node, "solve expression statements must be approved calls")
            if isinstance(node, ast.Return) and not (
                node.value is None
                or (isinstance(node.value, ast.Constant) and node.value.value is None)
            ):
                raise _error(node, "solve must return None")
        elif not isinstance(
            node,
            ast.Assign
            | ast.AnnAssign
            | ast.AugAssign
            | ast.Expr
            | ast.Return
            | ast.If
            | ast.For
            | ast.While
            | ast.Pass
            | ast.Break
            | ast.Continue,
        ):
            raise _error(node, f"{type(node).__name__} is not supported inside @triton.jit")

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._validate_assignment_target(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._validate_assignment_target(node.target)
        self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._validate_assignment_target(node.target)
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

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if self._function_kind == "solve" and not (
            isinstance(node.value, ast.Name) and node.value.id in self.jit_functions
        ):
            raise _error(node, "solve subscripting is reserved for @triton.jit kernel launches")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._function_kind == "solve":
            if isinstance(node.func, ast.Subscript):
                if (
                    not isinstance(node.func.value, ast.Name)
                    or node.func.value.id not in self.jit_functions
                ):
                    raise _error(node, "solve may launch only a declared @triton.jit kernel")
            elif isinstance(node.func, ast.Attribute):
                path = _attribute_path(node.func)
                if path not in {
                    *(("triton", name) for name in _TRITON_HOST_CALLS),
                    ("output", "zero_"),
                }:
                    raise _error(node, "solve call is outside the launcher allowlist")
            else:
                raise _error(node, "solve call is outside the launcher allowlist")
        elif self._function_kind == "kernel":
            if isinstance(node.func, ast.Attribute):
                path = _attribute_path(node.func)
                if not (
                    (
                        path is not None
                        and len(path) == 2
                        and path[0] == "tl"
                        and path[1] in _TL_CALLS
                    )
                    or node.func.attr == "to"
                ):
                    raise _error(node, "kernel call is outside the Triton language allowlist")
            elif not isinstance(node.func, ast.Name) or node.func.id not in (
                self.jit_functions | {"range"}
            ):
                raise _error(node, "kernel call is outside the Triton language allowlist")
        else:
            raise _error(node, "calls are not allowed at module scope")
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
        raise SubmissionPolicyError(str(error)) from error
    return _PolicyValidator(expected_parameters).validate(tree)


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
    import triton
    import triton.language as tl

    module = ModuleType("_myleetgpu_submission")
    module.__file__ = str(source_path)
    module.__dict__.update(
        {
            "__builtins__": dict(_SAFE_BUILTINS),
            "torch": torch,
            "triton": triton,
            "tl": tl,
        }
    )
    code = compile(executable, "source.py", "exec", dont_inherit=True, optimize=2)
    exec(code, module.__dict__, module.__dict__)
    solve = module.__dict__.get("solve")
    if not callable(solve):
        raise SubmissionPolicyError("Triton submission policy: solve must be callable")
    return module


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: submission_policy.py /work/source.py", file=sys.stderr)
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
