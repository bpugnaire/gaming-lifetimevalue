.PHONY: help install install-dev format test clean notebook

help:
	@echo "Available commands:"
	@echo "  install      Install project dependencies"
	@echo "  install-dev  Install with development dependencies"
	@echo "  format       Format and lint code"
	@echo "  test         Run tests"
	@echo "  notebook     Start Marimo notebook server"

install:
	uv sync --frozen

install-dev:
	uv sync --all-groups

format:
	uv run ruff format src/
	uv run ruff check --fix src/

test:
	uv run pytest src/tests/ -v

notebook:
	uv run marimo edit

mlflow-ui:
	uv run mlflow ui --backend-store-uri mlruns

