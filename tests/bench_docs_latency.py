from __future__ import annotations

import json
import math
import os
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright


DOCS = Path.cwd() / "docs"
CHROME = os.environ.get("CHROME_BIN", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
DELAY = 0.24
STEPS = [
    ("#modules", "八个核心模块与真实依赖", False),
    ("#flows", "运行数据流", False),
    ("#notebooks", "Notebook 构建链", False),
    ("#modules", "八个核心模块与真实依赖", False),
    ("#en/modules", "Eight core modules and the real dependency graph", False),
    ("#modules", "八个核心模块与真实依赖", False),
    ("#diagram", "SLM 架构总览", True),
    ("#modules", "八个核心模块与真实依赖", False),
    ("#diagram", "SLM 架构总览", True),
    ("#modules", "八个核心模块与真实依赖", False),
]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DOCS), **kwargs)

    def do_GET(self):
        if self.path.startswith(("/chapters/", "/architecture.html")):
            time.sleep(DELAY)
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *_):
        pass


def percentile(values: list[float], fraction: float) -> float:
    samples = sorted(values)
    index = (len(samples) - 1) * fraction
    low = math.floor(index)
    return samples[low] + (samples[math.ceil(index)] - samples[low]) * (index - low)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    chapters, diagrams = [], []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, executable_path=CHROME)
            try:
                for _ in range(3):
                    context = browser.new_context(viewport={"width": 1280, "height": 800})
                    page = context.new_page()
                    page.goto(f"http://127.0.0.1:{server.server_port}/#overview")
                    page.wait_for_function("document.querySelector('#content h1')?.textContent.includes('总览：单一实现')")
                    for hash_value, heading, diagram in STEPS:
                        elapsed = page.evaluate("""async ({hash, heading, diagram}) => {
                          const link = [...document.querySelectorAll('a[href]')].find(a => a.getAttribute('href') === hash);
                          if (!link) throw new Error(`missing navigation link: ${hash}`);
                          const start = performance.now();
                          link.click();
                          while (performance.now() - start < 15000) {
                            const ready = location.hash === hash &&
                              document.querySelector('#content h1')?.textContent === heading;
                            const frame = diagram ? document.querySelector('iframe.diagram') : null;
                            if (ready && (!diagram || frame?.contentDocument?.querySelector('svg'))) {
                              if (document.querySelector('#content .error')) throw new Error(`render error: ${hash}`);
                              return performance.now() - start;
                            }
                            await new Promise(requestAnimationFrame);
                          }
                          throw new Error(`navigation timed out: ${hash}`);
                        }""", {"hash": hash_value, "heading": heading, "diagram": diagram})
                        (diagrams if diagram else chapters).append(elapsed)
                    context.close()
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
    chapter_p75, diagram_p75 = percentile(chapters, 0.75), percentile(diagrams, 0.75)
    score = 0.75 * chapter_p75 + 0.25 * diagram_p75
    print(json.dumps({"chapter_p75_ms": round(chapter_p75, 1), "diagram_p75_ms": round(diagram_p75, 1), "chapter_samples_ms": [round(s, 1) for s in chapters], "diagram_samples_ms": [round(s, 1) for s in diagrams], "delay_ms": DELAY * 1000}))
    print(f"score_ms={score:.1f}")


if __name__ == "__main__":
    main()
