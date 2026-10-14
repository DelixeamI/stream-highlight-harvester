.PHONY: help install lint format test test-cov demo serve openapi docker-build docker-up clean

help:
	@echo "Available commands for Stream Highlight Harvester:"
	@echo "  make install       Install all project dependencies with dev extras"
	@echo "  make lint          Run Ruff static analysis and linting checks"
	@echo "  make format        Format codebase using Ruff"
	@echo "  make test          Run full unit test suite"
	@echo "  make demo          Run instant 2-second synthetic stream demo pipeline"
	@echo "  make serve         Start FastAPI asynchronous microservice on port 8000"
	@echo "  make openapi       Export OpenAPI 3.1 schema to docs/openapi.json"
	@echo "  make docker-build  Build production CUDA 12.8 container image"
	@echo "  make docker-up     Start API + Ollama LLM stack in Docker Compose"
	@echo "  make clean         Remove cache files and temporary build artifacts"

install:
	uv sync --extra dev

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest -v tests/unit/

demo:
	uv run narezka demo --open

serve:
	uv run narezka serve --port 8000 --reload

openapi:
	uv run narezka openapi --output docs/openapi.json

docker-build:
	docker build -t narezka:latest .

docker-up:
	docker compose up -d

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ *.egg-info build dist
