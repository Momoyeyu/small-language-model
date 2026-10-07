# Repository workflow

## Generated notebooks

- `tools/chapters.py` is the explicit pedagogical manifest. Change teaching order, prose, source selections, and experiments there.
- `tools/build_notebooks.py` extracts first-use implementations from `slm/`, cleans demo bootstrap code without reformatting teaching blocks, and generates `notebooks/*.ipynb` deterministically.
- Never hand-edit generated notebook cells or outputs. Run `.venv/bin/python tools/build_notebooks.py --execute` after changing the manifest, demos, or core source.
- Every generated notebook records a conservative digest of all `slm/*.py` files. Any core change invalidates all notebooks, including chapters that only import previously taught symbols.
- A normal build preserves an executed notebook only when its complete cell signature and core digest both match. Changed content produces an unexecuted notebook until execution succeeds.
- Validate recorded artifacts with `.venv/bin/python tools/build_notebooks.py --check`; a stale digest must be resolved with `make notebooks`.
- Notebook execution uses the repository virtual environment through a temporary kernelspec and does not modify global Jupyter configuration.

## Dependencies and verification

- `requirements.txt` is sufficient for the reusable core and core tests. Notebook tests skip at module collection when optional notebook packages are absent.
- `requirements-notebooks.txt` pins the optional generator, execution, and notebook-test dependencies. Install it for the complete suite and generated artifact checks.
- Build and execute with `make notebooks`, then validate with `make check-notebooks`.
- The complete CPU suite, including slow training checks, is `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SLM_SLOW=1 SLM_DEVICE=cpu .venv/bin/python -m pytest -q`.
- Do not claim the complete suite passed unless that exact gate was run after the final relevant edit. Never tune scientific seeds, constants, or tolerances merely to force convergence.

## Static architecture documentation

- `docs/index.html` and `docs/en.html` are complete bilingual static pages sharing `docs/style.css`; `docs/site.js` may enhance navigation but content must remain usable without JavaScript.
- Do not add external runtime dependencies, CDNs, remote fonts, generated frameworks, machine paths, hostnames, or credentials to the docs. Do not claim deployment until remote status and URL are verified.
- Validate links, fragments, assets, language parity, and declared core import edges with `make check-docs`.
- Preview locally with `make docs`. GitHub Pages publishing is branch-based from `master/docs`; repository settings and the actual deployment remain maintainer-owned.
- Keep one current implementation under `slm/`; do not add legacy import wrappers or migration-history sections to reader-facing docs.
