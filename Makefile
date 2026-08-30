# cinegeist task runner.
# No `make`? (common on Windows) — the equivalent direct commands are in README.md.

PY ?= python
VENV := .venv

ifeq ($(OS),Windows_NT)
	BIN := $(VENV)/Scripts
else
	BIN := $(VENV)/bin
endif

.PHONY: setup lint format test test-network check catalog catalog-refresh web-shard

setup:  ## create the venv, install the package (with dev extras), install the git hook
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev]"
	git config core.hooksPath .githooks
	@echo "Done. Activate the venv, then run 'make check'."

lint:  ## ruff lint + format check (what CI enforces)
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format:  ## auto-format and auto-fix
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

test:  ## unit tests, no network
	$(BIN)/pytest

test-network:  ## integration tests that reach real APIs
	$(BIN)/pytest -m network

check: lint test  ## run before every PR

catalog:
	@echo "Catalog build arrives in session 2 (see plan.md)."

catalog-refresh:
	@echo "Incremental catalog refresh arrives in session 2 (see plan.md)."

web-shard:
	@echo "Web shard build arrives in session 5 (see plan.md)."
