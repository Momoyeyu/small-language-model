"""Build, execute, and validate the generated teaching notebooks."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable

import nbformat
from nbclient import NotebookClient
from nbformat import NotebookNode

try:
    from tools.chapters import CHAPTERS
except ModuleNotFoundError:
    from chapters import CHAPTERS

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def core_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "slm").glob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _source_path(path: str) -> Path:
    candidate = ROOT / path
    if not candidate.is_file():
        raise ValueError(f"source path does not exist: {path}")
    return candidate


def _assignment_names(node: ast.AST) -> set[str]:
    if isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in node.elts:
            names.update(_assignment_names(element))
        return names
    return {node.id} if isinstance(node, ast.Name) else set()


def _node_names(node: ast.AST) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        names: set[str] = set()
        for target in node.targets:
            names.update(_assignment_names(target))
        return names
    if isinstance(node, ast.AnnAssign):
        return _assignment_names(node.target)
    return set()


def _span(lines: list[str], node: ast.AST) -> str:
    start = node.lineno
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.decorator_list:
        start = min(decorator.lineno for decorator in node.decorator_list)
    return "".join(lines[start - 1:node.end_lineno]).rstrip()


def _extract_items(path: str, names: tuple[str, ...]) -> list[tuple[str, str]]:
    source_path = _source_path(path)
    text = source_path.read_text()
    tree = ast.parse(text, filename=path)
    lines = text.splitlines(keepends=True)
    nodes: dict[str, ast.AST] = {}
    for node in tree.body:
        for name in _node_names(node):
            nodes[name] = node

    items = []
    for name in names:
        if name not in nodes:
            available = ", ".join(sorted(nodes))
            raise ValueError(f"unknown source name {name!r} in {path}; available: {available}")
        items.append((name, _span(lines, nodes[name])))
    return items


def extract(path: str, names: tuple[str, ...]) -> str:
    """Return exact top-level source spans in the explicitly requested order."""
    return "\n\n".join(source for _, source in _extract_items(path, names))


def _is_bootstrap_insert(node: ast.AST) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    function = node.value.func
    return (
        isinstance(function, ast.Attribute)
        and function.attr == "insert"
        and isinstance(function.value, ast.Attribute)
        and function.value.attr == "path"
        and isinstance(function.value.value, ast.Name)
        and function.value.value.id == "sys"
    )


def demo(path: str, defined: set[str]) -> str:
    """Remove script bootstrap without reformatting the teaching code."""
    text = _source_path(path).read_text()
    lines = text.splitlines(keepends=True)
    body = ast.parse(text, filename=path).body
    edits = []
    for index, node in enumerate(body):
        replacement = None
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            replacement = ""
        elif _is_bootstrap_insert(node):
            replacement = ""
        elif isinstance(node, ast.Import):
            kept = [alias for alias in node.names if alias.name not in {"os", "sys"}]
            if len(kept) != len(node.names):
                replacement = ast.unparse(ast.Import(names=kept)) + "\n" if kept else ""
        elif isinstance(node, ast.ImportFrom) and node.module == "slm":
            kept = [
                alias for alias in node.names
                if (alias.asname or alias.name) not in defined
            ]
            if len(kept) != len(node.names):
                replacement = (
                    ast.unparse(ast.ImportFrom(module="slm", names=kept, level=0)) + "\n"
                    if kept else ""
                )
        if replacement is not None:
            edits.append((node.lineno - 1, node.end_lineno, replacement))
    for start, end, replacement in reversed(edits):
        lines[start:end] = [replacement] if replacement else []
    return "".join(lines).strip()


def _bound_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        names.update(_node_names(node))
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _cell_id(slug: str, index: int, kind: str, source: str) -> str:
    digest = hashlib.sha256(f"{kind}\0{source}".encode()).hexdigest()[:12]
    return f"{slug}-{index:02d}-{digest}"


def build(chapter: dict) -> NotebookNode:
    """Build one unexecuted notebook from an explicit chapter manifest."""
    slug = chapter["slug"]
    cells = []
    defined: set[str] = set()
    source_entries: set[tuple[str, str]] = set()

    for entry in chapter["cells"]:
        kind, payload = entry
        pending: list[tuple[str, dict]] = []
        if kind == "markdown":
            pending.append((payload.strip(), {}))
        elif kind == "code":
            source = payload.strip()
            pending.append((source, {}))
            defined.update(_bound_names(source))
        elif kind == "source":
            path, names = payload
            for name, source in _extract_items(path, tuple(names)):
                key = (path, name)
                if key in source_entries:
                    raise ValueError(f"duplicate source entry: {path}:{name}")
                source_entries.add(key)
                pending.append((source, {"origin": {"path": path, "name": name}}))
                defined.add(name)
        elif kind == "demo":
            source = demo(payload, defined)
            pending.append((source, {"origin": {"path": payload, "name": "demo"}}))
            defined.update(_bound_names(source))
        else:
            raise ValueError(f"unknown cell kind {kind!r} in {slug}")

        for source, metadata in pending:
            index = len(cells)
            if kind == "markdown":
                cell = nbformat.v4.new_markdown_cell(source, metadata=metadata)
            else:
                cell = nbformat.v4.new_code_cell(source, metadata=metadata)
            cell.id = _cell_id(slug, index, kind, source)
            cells.append(cell)

    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "chapter": {"slug": slug, "title": chapter["title"]},
            "slm_core_sha": core_digest(),
        },
    )


def signature(nb: NotebookNode) -> list[tuple[str, str]]:
    return [(cell.cell_type, cell.source) for cell in nb.cells]


def _write(nb: NotebookNode, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)


def _execute(nb: NotebookNode) -> NotebookNode:
    with tempfile.TemporaryDirectory(prefix="slm-kernel-") as temp:
        kernel_dir = Path(temp) / "kernels" / "python3"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python 3 (SLM venv)",
            "language": "python",
        }))
        old_path = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = temp + (os.pathsep + old_path if old_path else "")
        try:
            client = NotebookClient(
                nb,
                timeout=600,
                kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
                record_timing=False,
            )
            return client.execute()
        finally:
            if old_path is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = old_path


def _selected(slug: str | None) -> Iterable[dict]:
    chapters = CHAPTERS if slug is None else [chapter for chapter in CHAPTERS if chapter["slug"] == slug]
    if not chapters:
        choices = ", ".join(chapter["slug"] for chapter in CHAPTERS)
        raise ValueError(f"unknown chapter {slug!r}; choose one of: {choices}")
    return chapters


def build_files(slug: str | None = None, execute: bool = False) -> None:
    for chapter in _selected(slug):
        generated = build(chapter)
        target = NOTEBOOK_DIR / f"{chapter['slug']}.ipynb"
        failed = NOTEBOOK_DIR / f".failed-{chapter['slug']}.ipynb"
        if execute:
            try:
                completed = _execute(generated)
            except Exception:
                _write(generated, failed)
                print(f"execution failed; partial notebook: {failed}", file=sys.stderr)
                raise
            _write(completed, target)
            failed.unlink(missing_ok=True)
            print(f"executed {target.relative_to(ROOT)}")
        elif target.exists():
            recorded = nbformat.read(target, as_version=4)
            if (
                signature(recorded) == signature(generated)
                and recorded.metadata.get("slm_core_sha") == generated.metadata["slm_core_sha"]
            ):
                failed.unlink(missing_ok=True)
                print(f"unchanged {target.relative_to(ROOT)}")
                continue
            _write(generated, target)
            print(f"built {target.relative_to(ROOT)}")
        else:
            _write(generated, target)
            print(f"built {target.relative_to(ROOT)}")


def check_files(slug: str | None = None) -> None:
    failures = []
    for chapter in _selected(slug):
        target = NOTEBOOK_DIR / f"{chapter['slug']}.ipynb"
        if not target.exists():
            failures.append(f"missing {target.relative_to(ROOT)}")
            continue
        recorded = nbformat.read(target, as_version=4)
        generated = build(chapter)
        if recorded.metadata.get("slm_core_sha") != generated.metadata["slm_core_sha"]:
            failures.append(
                f"stale core dependencies in {target.relative_to(ROOT)}; run make notebooks"
            )
            continue
        if signature(recorded) != signature(generated):
            failures.append(f"stale sources in {target.relative_to(ROOT)}")
            continue
        expected_count = 1
        for index, cell in enumerate(recorded.cells):
            if cell.cell_type != "code":
                continue
            if cell.execution_count != expected_count:
                failures.append(
                    f"{target.relative_to(ROOT)} cell {index}: execution_count "
                    f"{cell.execution_count!r}, expected {expected_count}"
                )
            expected_count += 1
            if any(output.output_type == "error" for output in cell.outputs):
                failures.append(f"{target.relative_to(ROOT)} cell {index}: error output")
        print(f"checked {target.relative_to(ROOT)}")
    if failures:
        raise SystemExit("\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="build and execute selected notebooks")
    mode.add_argument("--check", action="store_true", help="check recorded sources and execution state")
    parser.add_argument("chapter", nargs="?", help="optional chapter slug")
    args = parser.parse_args()
    if args.check:
        check_files(args.chapter)
    else:
        build_files(args.chapter, execute=args.execute)


if __name__ == "__main__":
    main()
