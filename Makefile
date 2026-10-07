PYTHON ?= python

.PHONY: notebooks check-notebooks docs check-docs test

notebooks:
	$(PYTHON) tools/build_notebooks.py --execute

check-notebooks:
	$(PYTHON) tools/build_notebooks.py --check

docs:
	$(PYTHON) -m http.server 8000 --bind 127.0.0.1 --directory docs

check-docs:
	$(PYTHON) -m pytest tests/test_docs.py -q

test:
	$(PYTHON) -m pytest
