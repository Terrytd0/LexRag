# LexRAG developer commands. Requires `make` (Git Bash/WSL/macOS/Linux --
# there's no native `make` on plain Windows cmd/PowerShell) and uv
# (https://docs.astral.sh/uv/). All targets delegate to `uv run ...` so the
# project's managed .venv is always what actually executes, never whatever
# Python happens to be on PATH.

.DEFAULT_GOAL := help

.PHONY: help install sync run evaluate test test-cov lint format format-check typecheck check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create/refresh the .venv, install all deps (incl. dev), install git hooks
	uv sync --extra dev
	uv run pre-commit install

sync: ## Sync the .venv to exactly match pyproject.toml / uv.lock
	uv sync --extra dev

run: ## Run the API locally (uv run python -m api; reads APP_HOST/APP_PORT from .env)
	uv run python -m api

evaluate: ## Run the full golden-dataset evaluation (needs the live stack + OPENAI_API_KEY)
	uv run python scripts/run_evaluation.py

test: ## Run the unit test suite
	uv run pytest

test-cov: ## Run tests with a coverage report
	uv run pytest --cov=api --cov=configs --cov=domain --cov=ingestion --cov=retrieval \
		--cov=generation --cov=evaluation --cov-report=term-missing

lint: ## Lint with ruff
	uv run ruff check .

format: ## Auto-format with ruff (rewrites files)
	uv run ruff format .

format-check: ## Check formatting without modifying files (what CI runs)
	uv run ruff format --check .

typecheck: ## Type-check with mypy
	uv run mypy .

check: format-check lint typecheck test ## Full quality gate: format check + lint + typecheck + tests (what CI runs)

clean: ## Remove caches, coverage output, and build artifacts (keeps .venv)
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build ./*.egg-info
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
