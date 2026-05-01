"""Aplicação FastAPI para previsão de churn de clientes.

Endpoints:
    GET  /health   — verificação de saúde, confirma que o modelo está carregado
    POST /predict  — retorna a probabilidade de churn e a decisão binária
    POST /predict/batch  — recebe Excel ou CSV com dados brutos, retorna CSV com predições

Middleware:
    - Log de latência: cada requisição loga método, path, status e duração
"""

import io
import time
from contextlib import asynccontextmanager
from datetime import datetime

import pandera.pandas as pa
from fastapi import FastAPI, File, Request, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from src.config import TARGET_COLUMN
from src.data.cleaning import clean
from src.data.loaders import load_from_upload
from src.inference import ChurnPredictor
from src.logger import get_logger, setup_logging
from src.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionInput,
    PredictionResponse,
)

from autenticacao import (
    create_token,
    get_current_user,
    authenticate_user,
    TOKEN_EXPIRE_MINUTES,
    LoginRequest
)

from services import MODEL_LOADED

from src.schemas.output import ChurnPrediction

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configura o logging estruturado no startup da aplicação"""
    setup_logging(json_logs=True)
    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts customer churn probability using a MLP (PyTorch) model.",
    version="2.0.0",
    lifespan=lifespan,
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

# ── Login ─────────────────────────────────────────────────────────────────

@app.post("/login")
def login(credentials: LoginRequest):

    user = authenticate_user(credentials.username, credentials.password)

    if not user:
        raise HTTPException(status_code=401, detail="Credenciais inválidas")

    token = create_token(user["username"], user["role"])

    return {
        "access_token": token,
        "expires_in": TOKEN_EXPIRE_MINUTES * 60
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "modelo_carregado": MODEL_LOADED
    }

@app.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return current_user

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness check — confirma que o modelo está carregado e pronto."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(),
        model_version=_MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(customer: PredictionInput, current_user: dict = Depends(get_current_user)) -> PredictionResponse:
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


@app.post("/predict/batch", tags=["inference"])
def predict_batch(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)) -> StreamingResponse:  # noqa: B008
    """Recebe Excel ou CSV com dados brutos e retorna CSV com predições.

    O arquivo deve conter as colunas do dataset Telco Customer Churn original.
    Colunas extras são removidas automaticamente pelo pipeline de limpeza.
    Retorna CSV com duas colunas adicionais:
        - churn_probability : probabilidade de churn (0.0 a 1.0)
        - churn_prediction  : decisão binária (0 = No Churn, 1 = Churn)
    """
    start = time.perf_counter()

    df_raw = load_from_upload(file.file.read(), file.filename)
    df_clean = clean(df_raw).drop(columns=[TARGET_COLUMN], errors="ignore")
    predictions = predictor.predict(df_clean)

    df_out = df_clean.copy()
    df_out["churn_probability"] = predictions["churn_probability"].values
    df_out["churn_prediction"] = predictions["churn_prediction"].values

    stream = io.StringIO()
    df_out.to_csv(stream, index=False)
    stream.seek(0)

    logger.info(
        "batch prediction complete",
        n_samples=len(df_clean),
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=predictions.csv",
            "X-Model-Version": _MODEL_VERSION,
            "X-Threshold": str(predictor.threshold),
        },
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)