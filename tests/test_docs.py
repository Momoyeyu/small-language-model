import ast
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGES = {"index.html": "zh-CN", "en.html": "en"}
SECTIONS = {"overview", "modules", "flows", "notebooks", "extension", "boundaries"}
MODULES = {"common", "envs", "lm", "moe", "norm", "position", "rl", "transformer"}
REPO_SOURCE = "https://github.com/Momoyeyu/small-language-model/blob/master/slm/{}.py"


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.lang = None
        self.ids = []
        self.section_ids = set()
        self.hrefs = []
        self.assets = []
        self.script_sources = []
        self.module_relations = {}
        self.hidden_elements = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "html":
            self.lang = attributes.get("lang")
        if "id" in attributes:
            self.ids.append(attributes["id"])
            if tag == "section":
                self.section_ids.add(attributes["id"])
        if "href" in attributes:
            self.hrefs.append(attributes["href"])
            if tag == "link":
                self.assets.append(attributes["href"])
        if "src" in attributes:
            self.assets.append(attributes["src"])
            if tag == "script":
                self.script_sources.append(attributes["src"])
        if "hidden" in attributes:
            self.hidden_elements.append(tag)
        if "data-module" in attributes:
            module = attributes["data-module"]
            assert module not in self.module_relations
            dependencies = set(filter(None, attributes.get("data-depends", "").split()))
            self.module_relations[module] = dependencies


def parse_page(path):
    parser = PageParser()
    parser.feed(path.read_text())
    return parser


@pytest.fixture(scope="module")
def pages():
    return {name: parse_page(DOCS / name) for name in PAGES}


def test_pages_have_languages_unique_ids_and_equivalent_sections(pages):
    for name, language in PAGES.items():
        page = pages[name]
        assert page.lang == language
        assert len(page.ids) == len(set(page.ids))
        assert page.section_ids == SECTIONS
        assert not page.hidden_elements
    assert pages["index.html"].section_ids == pages["en.html"].section_ids


def test_local_assets_and_fragment_links_resolve_from_each_page(pages):
    for name, page in pages.items():
        origin = DOCS / name
        for reference in page.hrefs + page.assets:
            parsed = urlsplit(reference)
            if parsed.scheme in {"http", "https"}:
                continue
            assert not reference.startswith("/"), f"root-absolute URL is not Pages-safe: {reference}"
            target_path = unquote(parsed.path)
            target = origin if not target_path else origin.parent / target_path
            assert target.exists(), f"broken local reference from {name}: {reference}"
            if parsed.fragment:
                target_page = pages[name] if target.resolve() == origin.resolve() else parse_page(target)
                assert parsed.fragment in target_page.ids, f"broken fragment from {name}: {reference}"


def test_required_canonical_source_links_are_present(pages):
    expected = {REPO_SOURCE.format(module) for module in MODULES}
    for name, page in pages.items():
        assert expected <= set(page.hrefs), name


def test_no_remote_runtime_assets_or_placeholder_copy(pages):
    forbidden = ("TODO", "TBD", "PLACEHOLDER", "LOREM IPSUM")
    for name, page in pages.items():
        assert all(not urlsplit(source).scheme for source in page.script_sources)
        text = (DOCS / name).read_text()
        assert not any(token in text.upper() for token in forbidden)
    css = (DOCS / "style.css").read_text()
    assert "@import" not in css
    assert "http://" not in css and "https://" not in css


def actual_core_imports():
    relations = {}
    for path in sorted((ROOT / "slm").glob("*.py")):
        if path.name == "__init__.py":
            continue
        dependencies = set()
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module in MODULES:
                dependencies.add(node.module)
        relations[path.stem] = dependencies
    return relations


def test_module_dependency_schema_matches_actual_relative_imports(pages):
    expected = actual_core_imports()
    assert set(expected) == MODULES
    assert expected == {module: set() for module in MODULES} | {
        "moe": {"transformer"},
        "rl": {"envs"},
    }
    for name, page in pages.items():
        assert page.module_relations == expected, name


def test_static_navigation_works_without_javascript(pages):
    for name, page in pages.items():
        local_fragments = {href.removeprefix("#") for href in page.hrefs if href.startswith("#")}
        assert SECTIONS <= local_fragments
        assert "./index.html" in page.hrefs
        assert "./en.html" in page.hrefs
        assert page.script_sources == ["./site.js"]
