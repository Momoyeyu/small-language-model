import ast
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MODULES = {"common", "envs", "lm", "moe", "norm", "position", "rl", "transformer"}
REPO_SOURCE = "https://github.com/Momoyeyu/small-language-model/blob/master/slm/{}.py"
FORBIDDEN = ("TODO", "TBD", "PLACEHOLDER", "LOREM IPSUM", "/USERS/")


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.lang = None
        self.hrefs = []
        self.assets = []
        self.script_sources = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "html":
            self.lang = attributes.get("lang")
        if "href" in attributes:
            self.hrefs.append(attributes["href"])
            if tag == "link":
                self.assets.append(attributes["href"])
        if "src" in attributes:
            self.assets.append(attributes["src"])
            if tag == "script":
                self.script_sources.append(attributes["src"])


def parse_page(path):
    parser = PageParser()
    parser.feed(path.read_text())
    return parser


def chapter_files(manifest, language):
    return [
        chapter["file"]
        for section in manifest["languages"][language]["sections"]
        for chapter in section["chapters"]
    ]


def chapter_ids(manifest, language):
    return [
        chapter["id"]
        for section in manifest["languages"][language]["sections"]
        for chapter in section["chapters"]
    ]


@pytest.fixture(scope="module")
def manifest():
    return json.loads((DOCS / "manifest.json").read_text())


def doc_files():
    return sorted(
        path
        for path in DOCS.rglob("*")
        if path.is_file() and path.suffix in {".html", ".css", ".js", ".json", ".md"}
    )


def test_manifest_structure_and_language_parity(manifest):
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["sourceRevision"])
    languages = manifest["languages"]
    assert set(languages) == {"zh", "en"}
    assert chapter_ids(manifest, "zh") == chapter_ids(manifest, "en")
    for language in languages:
        files = chapter_files(manifest, language)
        assert len(files) == len(set(files)) == 6
        for chapter_file in files:
            assert (DOCS / chapter_file).is_file(), chapter_file


def test_chapter_pairs_have_matching_headings(manifest):
    for zh_file, en_file in zip(chapter_files(manifest, "zh"), chapter_files(manifest, "en")):
        zh = (DOCS / zh_file).read_text()
        en = (DOCS / en_file).read_text()
        zh_headings = re.findall(r"^#{1,3} ", zh, flags=re.M)
        en_headings = re.findall(r"^#{1,3} ", en, flags=re.M)
        assert zh_headings == en_headings, (zh_file, en_file)


def test_index_shell_and_english_redirect():
    index = parse_page(DOCS / "index.html")
    assert index.lang == "zh-CN"
    assert "./style.css?v=2" in index.hrefs
    assert index.script_sources == ["./site.js?v=2"]
    assert "#overview" in index.hrefs and "#en/overview" in index.hrefs
    assert 'class="language-switch"' in (DOCS / "index.html").read_text()
    for reference in index.hrefs + index.assets:
        parsed = urlsplit(reference)
        if parsed.scheme in {"http", "https"} or parsed.scheme == "data":
            continue
        target = unquote(parsed.path)
        assert not reference.startswith("/"), f"root-absolute URL is not Pages-safe: {reference}"
        assert (DOCS / (target or "index.html")).exists(), f"broken local reference: {reference}"

    en = (DOCS / "en.html").read_text()
    assert "index.html#en/overview" in en


def test_markdown_links_resolve(manifest):
    pattern = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")
    for language in ("zh", "en"):
        for chapter_file in chapter_files(manifest, language):
            origin = DOCS / chapter_file
            for match in pattern.finditer(origin.read_text()):
                url = match.group(1)
                parsed = urlsplit(url)
                if parsed.scheme or url.startswith("#"):
                    continue
                assert (origin.parent / unquote(parsed.path)).resolve().exists(), f"{chapter_file}: {url}"


def test_architecture_diagram_is_present_and_local(manifest):
    html = DOCS / "architecture.html"
    assert html.is_file()
    text = html.read_text()
    assert len(text) > 100_000
    assert "<svg" in text
    candidate = json.loads((DOCS / "candidate.json").read_text())
    assert candidate["meta"]["repository"]["revision"] == manifest["sourceRevision"]


def test_no_remote_runtime_assets_placeholders_or_machine_paths():
    index = parse_page(DOCS / "index.html")
    assert all(not urlsplit(source).scheme for source in index.script_sources)
    css = (DOCS / "style.css").read_text()
    assert "@import" not in css
    assert "http://" not in css and "https://" not in css
    for path in doc_files():
        if path.name == "architecture.html":
            continue
        text = path.read_text()
        assert not any(token in text.upper() for token in FORBIDDEN), path


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


def test_declared_core_edges_match_actual_imports(manifest):
    expected = actual_core_imports()
    assert set(expected) == MODULES
    declared = {tuple(edge) for edge in manifest["coreEdges"]}
    actual = {(module, dep) for module, deps in expected.items() for dep in deps}
    assert declared == actual


def test_module_source_links_present_in_module_chapters(manifest):
    expected = {REPO_SOURCE.format(module) for module in MODULES}
    for language in ("zh", "en"):
        modules_file = next(
            file for file in chapter_files(manifest, language) if "modules" in file
        )
        text = (DOCS / modules_file).read_text()
        assert expected <= set(re.findall(r"https?://[^)\s]+", text)), modules_file
