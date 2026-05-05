import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from src.api.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from src.config import CHURN_THRESHOLD
from src.models import load_artifacts, predict

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.preprocessor, app.state.model = load_artifacts()
        logger.info("Artefatos carregados — modelo e preprocessador prontos")
    except FileNotFoundError as exc:
        logger.error("Falha ao carregar artefatos: %s", exc)
        raise
    yield


app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Predição de churn em telecom com MLP PyTorch.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Erro não tratado em %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Erro interno do servidor."},
    )


@app.get("/health", response_model=HealthResponse, tags=["infra"])
def health(request: Request):
    model_loaded = (
        hasattr(request.app.state, "model") and request.app.state.model is not None
    )
    return HealthResponse(status="ok", model_loaded=model_loaded)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_churn(features: CustomerFeatures, request: Request):
    start = time.perf_counter()

    df = pd.DataFrame([features.model_dump()])
    probs, labels = predict(df, request.app.state.preprocessor, request.app.state.model)

    latency_ms = (time.perf_counter() - start) * 1000
    prob = float(probs[0])
    label = int(labels[0])

    logger.info(
        "predict — latency=%.1fms churn_prob=%.4f churn=%d",
        latency_ms,
        prob,
        label,
    )

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=label,
        threshold=CHURN_THRESHOLD,
    )
