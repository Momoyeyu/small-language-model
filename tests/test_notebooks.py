import ast
from collections import Counter

import pytest
nbformat = pytest.importorskip(
    "nbformat", reason="install requirements-notebooks.txt for notebook checks"
)
pytest.importorskip(
    "nbclient", reason="install requirements-notebooks.txt for notebook checks"
)

import torch
import torch.nn as nn
import torch.nn.functional as F

import slm
from tools import build_notebooks as builder
from tools.build_notebooks import NOTEBOOK_DIR, build, core_digest, demo, extract, signature
from tools.chapters import CHAPTERS


OPERATIONAL_EXPORTS = {"ROOT", "OUT_DIR", "get_device", "save_run"}


def recorded(chapter):
    return nbformat.read(NOTEBOOK_DIR / f"{chapter['slug']}.ipynb", as_version=4)


def source_cells(nb):
    return [
        cell for cell in nb.cells
        if cell.cell_type == "code" and cell.metadata.get("origin", {}).get("name") != "demo"
        and "origin" in cell.metadata
    ]


def test_generated_signatures_and_core_digest_match_all_recorded_notebooks():
    for chapter in CHAPTERS:
        current = recorded(chapter)
        generated = build(chapter)
        assert signature(current) == signature(generated)
        assert current.metadata["slm_core_sha"] == generated.metadata["slm_core_sha"]
        assert current.metadata["slm_core_sha"] == core_digest()


def test_recorded_notebooks_are_cleanly_executed_with_visible_outputs():
    for chapter in CHAPTERS:
        nb = recorded(chapter)
        code = [cell for cell in nb.cells if cell.cell_type == "code"]
        assert [cell.execution_count for cell in code] == list(range(1, len(code) + 1))
        assert not any(
            output.output_type == "error"
            for cell in code
            for output in cell.outputs
        )
        assert any(
            output.output_type == "stream" and output.get("text", "").strip()
            for cell in code
            for output in cell.outputs
        ), chapter["slug"]

    ppo = recorded(next(chapter for chapter in CHAPTERS if chapter["slug"] == "04_ppo"))
    assert any(
        output.output_type in {"display_data", "execute_result"} and "image/png" in output.get("data", {})
        for cell in ppo.cells if cell.cell_type == "code"
        for output in cell.outputs
    )


def test_source_metadata_is_exact_and_teaching_exports_appear_once():
    origins = []
    for chapter in CHAPTERS:
        for cell in source_cells(recorded(chapter)):
            origin = cell.metadata["origin"]
            assert cell.source == extract(origin["path"], (origin["name"],))
            origins.append(origin["name"])

    counts = Counter(origins)
    expected = set(slm.__all__) - OPERATIONAL_EXPORTS
    assert expected <= set(counts)
    assert set(counts) - expected == {"_NEG_INF"}
    assert all(counts[name] == 1 for name in expected)


def test_notebooks_do_not_hide_algorithms_behind_dynamic_source_execution():
    forbidden = ("inspect.getsource", "runpy", "from slm import *", "exec(")
    for chapter in CHAPTERS:
        for cell in recorded(chapter).cells:
            if cell.cell_type == "code":
                assert not any(token in cell.source for token in forbidden)


def test_decorated_source_is_retained():
    all_sources = {
        cell.metadata["origin"]["name"]: cell.source
        for chapter in CHAPTERS
        for cell in source_cells(recorded(chapter))
    }
    assert all_sources["eval_ppl"].startswith("@torch.no_grad()")
    assert "@torch.no_grad()\n    def generate" in all_sources["TinyLM"]


def _import_bindings(node):
    if not isinstance(node, (ast.Import, ast.ImportFrom)):
        return []
    return [(alias.name, alias.asname or alias.name.split(".")[0]) for alias in node.names]


def test_no_later_import_overwrites_an_earlier_taught_definition():
    for chapter in CHAPTERS:
        taught = set()
        for cell in recorded(chapter).cells:
            if cell.cell_type != "code":
                continue
            origin = cell.metadata.get("origin", {})
            if origin and origin.get("name") != "demo":
                taught.add(origin["name"])
            for node in ast.parse(cell.source).body:
                for imported, binding in _import_bindings(node):
                    assert binding not in taught or binding != imported, (
                        chapter["slug"], binding, cell.source
                    )


def test_demo_cleaning_keeps_third_party_imports_and_only_drops_defined_slm_names():
    cleaned = demo(
        "scripts/position/01_permutation_equivariance.py",
        {"MHA", "set_seed", "show"},
    )
    tree = ast.parse(cleaned)
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert any(isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names) for node in imports)
    assert not any(isinstance(node, ast.ImportFrom) and node.module == "slm" for node in imports)

    aliased = demo("scripts/position/01_permutation_equivariance.py", {"other_name"})
    assert "from slm import MHA, set_seed, show" in aliased


def test_demo_cleaning_preserves_comments_blank_blocks_semantics_and_runnability():
    path = "scripts/transformer/03_masked_attention.py"
    original = (builder.ROOT / path).read_text()
    defined = {"attention", "causal_mask", "padding_mask", "set_seed", "show"}
    cleaned = demo(path, defined)
    assert "# 改动未来 token 不影响过去位置的输出" in cleaned
    assert "\n\n# 改动未来 token 不影响过去位置的输出\n" in cleaned

    retained = []
    for index, node in enumerate(ast.parse(original).body):
        if index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.Import) and all(alias.name in {"os", "sys"} for alias in node.names):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "slm":
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "insert"
        ):
            continue
        retained.append(ast.dump(node, include_attributes=False))
    assert [ast.dump(node, include_attributes=False) for node in ast.parse(cleaned).body] == retained

    namespace = {
        "attention": slm.attention,
        "causal_mask": slm.causal_mask,
        "padding_mask": slm.padding_mask,
        "set_seed": slm.set_seed,
        "show": lambda name, value: None,
    }
    exec(compile(cleaned, path, "exec"), namespace)


def test_extracted_attention_and_mha_execute_locally_and_mha_uses_local_attention():
    namespace = {"math": __import__("math"), "torch": torch, "nn": nn, "F": F}
    exec(extract("slm/transformer.py", ("attention", "MHA")), namespace)
    original = namespace["attention"]
    calls = []

    def spy(*args, **kwargs):
        calls.append(args[0].shape)
        return original(*args, **kwargs)

    namespace["attention"] = spy
    model = namespace["MHA"](16, 4)
    x = torch.randn(2, 6, 16)
    assert model(x, x, x).shape == x.shape
    assert len(calls) == 4


def test_core_digest_changes_with_any_core_module_change(tmp_path, monkeypatch):
    root = tmp_path
    core = root / "slm"
    core.mkdir()
    module = core / "example.py"
    module.write_text("VALUE = 1\n")
    monkeypatch.setattr(builder, "ROOT", root)
    before = builder.core_digest()
    module.write_text("VALUE = 2\n")
    assert builder.core_digest() != before


def test_check_rejects_recorded_notebooks_with_stale_core_digest(monkeypatch):
    monkeypatch.setattr(builder, "core_digest", lambda: "changed-core")
    with pytest.raises(SystemExit, match="stale core dependencies.*run make notebooks"):
        builder.check_files("00_transformer")


def test_generation_is_deterministic():
    for chapter in CHAPTERS:
        first, second = build(chapter), build(chapter)
        assert signature(first) == signature(second)
        assert [cell.id for cell in first.cells] == [cell.id for cell in second.cells]


def test_duplicate_source_entry_is_rejected():
    chapter = {
        "slug": "duplicate",
        "title": "duplicate",
        "cells": [
            ("source", ("slm/common.py", ("show",))),
            ("source", ("slm/common.py", ("show",))),
        ],
    }
    with pytest.raises(ValueError, match="duplicate source entry.*slm/common.py:show"):
        build(chapter)


def test_unknown_source_name_has_actionable_error():
    with pytest.raises(ValueError, match="unknown source name.*missing_name.*slm/common.py"):
        extract("slm/common.py", ("missing_name",))
