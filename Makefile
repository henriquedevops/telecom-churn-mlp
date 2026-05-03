.PHONY: install lint format test run download-data

install:
	pip install -e ".[dev]"

download-data:
	kaggle datasets download blastchar/telco-customer-churn -p data/raw/ --unzip
	mv data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/telco_churn.csv

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

test:
	pytest tests/ --cov=src --cov-report=term-missing

run:
	uvicorn src.api.app:app --reload --port 8000
