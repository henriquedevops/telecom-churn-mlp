.PHONY: install lint format test run

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

test:
	pytest tests/ --cov=src --cov-report=term-missing

run:
	uvicorn src.api.app:app --reload --port 8000
