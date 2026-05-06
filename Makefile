.PHONY: install lint format test run download-data docker-build docker-run docker-push

IMAGE_NAME ?= telecom-churn-mlp
PORT       ?= 8000
GCP_PROJECT ?= my-gcp-project

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

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm -p $(PORT):$(PORT) $(IMAGE_NAME)

docker-push:
	docker tag $(IMAGE_NAME) gcr.io/$(GCP_PROJECT)/$(IMAGE_NAME)
	docker push gcr.io/$(GCP_PROJECT)/$(IMAGE_NAME)
