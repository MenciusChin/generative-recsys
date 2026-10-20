.DEFAULT_GOAL := help
.PHONY: help install hooks lint format test check

help: ## Show available targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-10s %s\n", $$1, $$2}'

install: ## Create or sync the environment from uv.lock
	uv sync --locked

hooks: ## Install the pre-commit git hook
	uv run --frozen pre-commit install

lint: ## Check lint and formatting without changing files
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

format: ## Auto-fix lint issues and format code
	uv run --frozen ruff check --fix .
	uv run --frozen ruff format .

test: ## Run the test suite
	uv run --frozen pytest

check: lint test ## Run lint and tests (what CI runs)
