"""Pillow and numpy stay out of module scope in the library."""

import ast
from pathlib import Path

import smythe

PACKAGE = Path(smythe.__file__).parent
# Both are imported inside the functions that need them. At module level, even
# under TYPE_CHECKING, they would load on every `import smythe` or pull their
# stubs into the package's mypy run, where numpy's stubs need a newer Python
# than the configured target and stop the check.
LAZY = {"PIL", "numpy"}


def _import_time_imports(statements):
    """Top-level names imported by statements that run at module import."""
    for node in statements:
        if isinstance(node, ast.Import):
            yield from (alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                yield node.module.split(".")[0]
        elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Class bodies and if/try/with blocks run at import (or are read
            # by type checkers); function bodies run only when called.
            for field in ("body", "orelse", "finalbody", "handlers"):
                block = getattr(node, field, None)
                if isinstance(block, list):
                    yield from _import_time_imports(block)


def test_pillow_and_numpy_are_imported_inside_functions():
    offenders = [
        f"{path.relative_to(PACKAGE).as_posix()}: {name}"
        for path in sorted(PACKAGE.rglob("*.py"))
        for name in _import_time_imports(ast.parse(path.read_text(encoding="utf-8")).body)
        if name in LAZY
    ]
    assert offenders == []
