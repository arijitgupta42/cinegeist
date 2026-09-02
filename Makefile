# cinegeist task runner.
# No `make`? (common on Windows) — the equivalent direct commands are in README.md.

PY ?= python
VENV := .venv

ifeq ($(OS),Windows_NT)
	BIN := $(VENV)/Scripts
else
	BIN := $(VENV)/bin
endif

.PHONY: setup lint format test test-network check catalog catalog-refresh spec web-shard \
	web-install web-test web-build web-dev eval

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

eval:  ## score the recommender against synthetic personas (precision@3); no catalog or key needed
	$(BIN)/cinegeist eval

test-network:  ## integration tests that reach real APIs
	$(BIN)/pytest -m network

check: lint test  ## run before every PR

spec:  ## regenerate the shared spec/ fixtures from the Python reference (plan.md §8.6)
	$(BIN)/python scripts/build_spec_fixtures.py

catalog:  ## build data/cinegeist.db and data/genome.npy (downloads MovieLens; slow, resumable)
	$(BIN)/cinegeist catalog build

catalog-refresh:
	@echo "Incremental catalog refresh arrives later in session 2 (see plan.md)."

web-shard:  ## build the browser demo shard into web/public/shard (needs data/; slow) then commit
	$(BIN)/python scripts/build_web_shard.py

web-install:  ## install the browser demo's npm dependencies (uses the committed lockfile)
	cd web && npm ci

web-test:  ## run the demo's vitest suite, including the shared spec/ fixtures (plan.md §8.6)
	cd web && npm test

web-build:  ## typecheck and build the static demo bundle into web/dist
	cd web && npm run build

web-dev:  ## run the demo's dev server (needs a built shard in web/public/shard)
	cd web && npm run dev
