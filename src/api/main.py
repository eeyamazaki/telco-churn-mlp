"""Aplicação FastAPI para previsão de churn de clientes.

Endpoints:
    GET  /health   — verificação de saúde, confirma que o modelo está carregado
    POST /predict  — retorna a probabilidade de churn e a decisão binária

Middleware:
    - Log de latência: cada requisição loga método, path, status e duração
"""

import time
from datetime import datetime

import pandera.pandas as pa
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.inference import ChurnPredictor
from src.logger import get_logger
from src.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionInput,
    PredictionResponse,
)
from src.schemas.output import ChurnPrediction

logger = get_logger(__name__)

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts customer churn probability using a MLP (PyTorch) model.",
    version="2.0.0",
)

# Predictor e versão carregados uma única vez na inicialização da aplicação
predictor = ChurnPredictor()
_MODEL_VERSION = "mlp-v1"


# ── Middleware de latência ────────────────────────────────────────────────────


@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    """Loga método, path, status e duração de cada requisição."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness check — confirma que o modelo está carregado e pronto."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(),
        model_version=_MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(customer: PredictionInput) -> PredictionResponse:
    """Prediz a probabilidade de churn para um único cliente.

    Retorna o score de probabilidade e a decisão binária com base no threshold
    F1-ótimo carregado de models/mlp_config.json. Para uso em campanhas de retenção,
    considere reduzir o threshold (FN custa ~28x mais que FP).
    """
    start = time.perf_counter()
    result = predictor.predict_single(customer.to_dict())
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    churn_prob = result["churn_probability"]
    is_churn = result["churn_prediction"] == 1

    prediction = ChurnPrediction(
        churn_probability=churn_prob,
        prediction="Churn" if is_churn else "No Churn",
        threshold_used=predictor.threshold,
        confidence=churn_prob if is_churn else round(1.0 - churn_prob, 4),
    )

    return PredictionResponse(
        prediction=prediction,
        timestamp=datetime.now(),
        model_version=_MODEL_VERSION,
        latency_ms=latency_ms,
    )


# ── Handlers de erros ─────────────────────────────────────────────────────────


@app.exception_handler(pa.errors.SchemaError)
async def schema_error_handler(request: Request, exc: pa.errors.SchemaError):
    """Retorna 422 quando o Pandera rejeita o DataFrame de entrada."""
    logger.warning("schema validation error", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error_code="SCHEMA_VALIDATION_ERROR",
            error_message=str(exc),
            timestamp=datetime.now(),
        ).model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            error_message="Internal server error",
            timestamp=datetime.now(),
        ).model_dump(mode="json"),
    )
