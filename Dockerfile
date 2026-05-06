FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch — must come first to prevent pip from pulling the 2.5 GB GPU build
RUN pip install --no-cache-dir \
    "torch>=2.2" \
    --index-url https://download.pytorch.org/whl/cpu

# Remaining production dependencies
RUN pip install --no-cache-dir \
    "scikit-learn>=1.4" \
    "mlflow>=2.12" \
    "fastapi>=0.111" \
    "uvicorn[standard]>=0.29" \
    "pydantic>=2.7" \
    "pandas>=2.2" \
    "numpy>=1.26" \
    "pandera>=0.19" \
    "python-dotenv>=1.0" \
    "joblib>=1.4"

COPY pyproject.toml .
RUN pip install --no-cache-dir -e . --no-deps

COPY src/ src/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
