"""
Epistemic World Model — Self-Modification Module
==================================================

The self-awareness layer of the cognitive architecture.  This module allows
the system to inspect its own source code, list its constituent modules,
map internal dependencies, and propose changes — all without ever auto-
applying modifications.

Every function here is read-only with respect to the filesystem.
``propose_change`` documents a proposal for human review but writes nothing.
The system can see itself but cannot rewrite itself.

Design principles:
    - Uses ``ast`` for reliable parsing (not regex).
    - Discovers its own location dynamically via ``__file__`` / ``pathlib.Path``.
    - stdlib only — no external dependencies.
    - All returned structures are plain dicts and lists for easy serialization.
"""

from __future__ import annotations

import ast
import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Package directory — discovered dynamically from this file's location
# ---------------------------------------------------------------------------

_PACKAGE_DIR: Path = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _read_source(file_path: Path) -> str:
    """Read and return the full source text from *file_path*, or empty string."""
    if not file_path.is_file():
        return ""
    return file_path.read_text(encoding="utf-8")


def _parse_ast(source: str) -> Optional[ast.Module]:
    """Parse *source* into an AST module node, returning None on failure."""
    if not source:
        return None
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _reconstruct_signature(node: ast.FunctionDef) -> str:
    """Build a human-readable signature string from an ``ast.FunctionDef``."""
    args = node.args

    parts: List[str] = []

    # Positional-only args (before /)
    for a in args.posonlyargs:
        parts.append(a.arg)

    # Regular positional args
    num_defaults = len(args.defaults)
    num_args = len(args.args)
    for i, a in enumerate(args.args):
        name = a.arg
        # Check if this arg has a default value
        default_idx = i - (num_args - num_defaults)
        if default_idx >= 0:
            name += "=..."
        parts.append(name)

    # *args
    if args.vararg:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")

    # keyword-only args
    for i, a in enumerate(args.kwonlyargs):
        name = a.arg
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            name += "=..."
        parts.append(name)

    # **kwargs
    if args.kwarg:
        parts.append("**" + args.kwarg.arg)

    sig = ", ".join(parts)

    # Return annotation
    ret = ""
    if node.returns:
        ret = " -> " + ast.dump(node.returns)
        # Simplify common cases for readability
        if isinstance(node.returns, ast.Constant):
            ret = " -> " + repr(node.returns.value)
        elif isinstance(node.returns, ast.Name):
            ret = " -> " + node.returns.id
        elif isinstance(node.returns, ast.Attribute):
            ret = " -> " + ast.unparse(node.returns) if hasattr(ast, "unparse") else ret

    return "{}({}){}".format(node.name, sig, ret)


def _extract_functions(tree: ast.Module) -> List[Dict[str, Any]]:
    """Extract public function signatures from the top-level of *tree*."""
    functions: List[Dict[str, Any]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            functions.append({
                "name": node.name,
                "signature": _reconstruct_signature(node),
                "line": node.lineno,
            })
    return functions


def _extract_classes(tree: ast.Module) -> List[Dict[str, Any]]:
    """Extract class names and their method names from the top-level of *tree*."""
    classes: List[Dict[str, Any]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods: List[str] = []
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(item.name)
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "methods": methods,
            })
    return classes


def _extract_imports(tree: ast.Module) -> List[str]:
    """Extract import lines from *tree* as human-readable strings."""
    imports: List[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    imports.append("import {} as {}".format(alias.name, alias.asname))
                else:
                    imports.append("import {}".format(alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ", ".join(
                "{} as {}".format(a.name, a.asname) if a.asname else a.name
                for a in node.names
            )
            imports.append("from {} import {}".format(module, names))
    return imports


def _extract_ewm_imports(tree: ast.Module) -> List[str]:
    """Return the list of ewm module names that *tree* imports from.

    Parses ``from ewm.X import ...`` and ``import ewm.X`` statements,
    extracting the module name ``X``.
    """
    modules: List[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("ewm."):
                # e.g. "ewm.types" -> "types"
                parts = node.module.split(".")
                if len(parts) >= 2:
                    mod = parts[1]
                    if mod not in modules:
                        modules.append(mod)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("ewm."):
                    parts = alias.name.split(".")
                    if len(parts) >= 2:
                        mod = parts[1]
                        if mod not in modules:
                            modules.append(mod)
    return modules


# ===========================================================================
# Public API
# ===========================================================================


def inspect_module(name: str) -> Dict[str, Any]:
    """Read and analyze a module in the ewm package.

    Args:
        name: Module name without extension (e.g. "perception", "types").

    Returns:
        A dict containing module metadata, source code, function/class/import
        listings, and size information.  If the module does not exist, the
        dict has ``exists=False`` and sensible zero defaults.
    """
    file_path = _PACKAGE_DIR / "{}.py".format(name)

    if not file_path.is_file():
        return {
            "name": name,
            "path": str(file_path),
            "exists": False,
            "source": "",
            "lines": 0,
            "docstring": "",
            "functions": [],
            "classes": [],
            "imports": [],
            "size_bytes": 0,
        }

    source = _read_source(file_path)
    tree = _parse_ast(source)

    docstring = ""
    functions: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    imports: List[str] = []

    if tree is not None:
        docstring = ast.get_docstring(tree) or ""
        functions = _extract_functions(tree)
        classes = _extract_classes(tree)
        imports = _extract_imports(tree)

    return {
        "name": name,
        "path": str(file_path),
        "exists": True,
        "source": source,
        "lines": source.count("\n") + (1 if source and not source.endswith("\n") else 0),
        "docstring": docstring,
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "size_bytes": file_path.stat().st_size,
    }


def propose_change(
    module: str, description: str, rationale: str
) -> Dict[str, Any]:
    """Propose a change to a module without modifying anything.

    This is intentionally conservative.  The function documents the current
    state and the proposed change for a human or LLM to review and apply.

    Args:
        module: Module name without extension.
        description: What should change.
        rationale: Why the change is warranted.

    Returns:
        A dict documenting the proposal, including current source and a note
        that changes are never auto-applied.
    """
    info = inspect_module(module)

    return {
        "module": module,
        "description": description,
        "rationale": rationale,
        "current_source": info["source"],
        "current_lines": info["lines"],
        "status": "proposed",
        "note": (
            "Changes must be reviewed and applied manually. "
            "This system never auto-modifies source code."
        ),
        "timestamp": _now_iso(),
    }


def list_modules() -> List[Dict[str, Any]]:
    """List all Python modules in the ewm package.

    Scans the package directory for ``*.py`` files, excluding
    ``__init__.py`` and any ``__pycache__`` artifacts.

    Returns:
        A list of dicts, each containing module metadata (name, path,
        line count, size, docstring presence, function/class counts).
    """
    results: List[Dict[str, Any]] = []

    py_files = sorted(_PACKAGE_DIR.glob("*.py"))
    for f in py_files:
        if f.name == "__init__.py":
            continue

        name = f.stem
        info = inspect_module(name)

        results.append({
            "name": name,
            "path": info["path"],
            "lines": info["lines"],
            "size_bytes": info["size_bytes"],
            "has_docstring": bool(info["docstring"]),
            "function_count": len(info["functions"]),
            "class_count": len(info["classes"]),
        })

    return results


def module_dependency_graph() -> Dict[str, Any]:
    """Build a dependency graph of ewm modules.

    For each module, parses its imports to find which other ewm modules
    it imports.  Returns modules, directed edges, and in/out degree counts.

    Returns:
        A dict with ``modules``, ``edges``, ``dependency_count`` (how many
        ewm modules each module depends on), and ``dependents_count`` (how
        many ewm modules depend on each module).
    """
    modules: List[str] = []
    edges: List[Dict[str, str]] = []
    dependency_count: Dict[str, int] = {}
    dependents_count: Dict[str, int] = {}

    py_files = sorted(_PACKAGE_DIR.glob("*.py"))
    for f in py_files:
        if f.name == "__init__.py":
            continue
        modules.append(f.stem)

    # Initialize counts to 0
    for mod in modules:
        dependency_count[mod] = 0
        dependents_count[mod] = 0

    # Parse each module and find its ewm imports
    for mod in modules:
        source = _read_source(_PACKAGE_DIR / "{}.py".format(mod))
        tree = _parse_ast(source)
        if tree is None:
            continue

        ewm_deps = _extract_ewm_imports(tree)
        # Only count dependencies on modules that actually exist in the package
        for dep in ewm_deps:
            if dep in modules and dep != mod:
                edges.append({"from": mod, "to": dep})
                dependency_count[mod] = dependency_count.get(mod, 0) + 1
                dependents_count[dep] = dependents_count.get(dep, 0) + 1

    return {
        "modules": modules,
        "edges": edges,
        "dependency_count": dependency_count,
        "dependents_count": dependents_count,
    }


def system_health() -> Dict[str, Any]:
    """Quick health check of the entire ewm codebase.

    Lists all modules, aggregates total lines and size, checks whether each
    module imports successfully, and includes the dependency graph.

    Returns:
        A dict with ``total_modules``, ``total_lines``, ``total_size_bytes``,
        ``modules``, ``import_status``, and ``dependency_graph``.
    """
    modules = list_modules()
    total_lines = sum(m["lines"] for m in modules)
    total_size = sum(m["size_bytes"] for m in modules)

    # Check which modules import successfully
    import_status: Dict[str, bool] = {}
    for m in modules:
        mod_name = "ewm.{}".format(m["name"])
        # If already imported, it succeeded
        if mod_name in sys.modules:
            import_status[m["name"]] = True
            continue
        try:
            importlib.import_module(mod_name)
            import_status[m["name"]] = True
        except Exception:
            import_status[m["name"]] = False

    graph = module_dependency_graph()

    return {
        "total_modules": len(modules),
        "total_lines": total_lines,
        "total_size_bytes": total_size,
        "modules": modules,
        "import_status": import_status,
        "dependency_graph": graph,
    }
