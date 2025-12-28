.PHONY: help install install-dev format test clean notebook

help:
	@echo "Available commands:"
	@echo "  install      Install project dependencies"
	@echo "  install-dev  Install with development dependencies"
	@echo "  format       Format and lint code"
	@echo "  test         Run tests"
	@echo "  notebook     Start Marimo notebook server"
	@echo " mlflow-ui    Start MLflow UI server"
	@echo " preprocessing Run data preprocessing pipeline"
	@echo " training     Run model training pipeline"
	@echo " inference    Run model inference pipeline"

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
	uv run mlflow ui --backend-store-uri 

preprocessing
	uv run python src/gaming_lifetimevalue/pipelines/preprocessing.py

training
	uv run python src/gaming_lifetimevalue/pipelines/training.py

inference
	uv run python src/gaming_lifetimevalue/pipelines/inference.py

