"""
Schemas para validação de dados (Pydantic e Pandera).

- input.py: Validação de requisições API (1 registro)
- output.py: Validação de respostas API (1 registro)
- common.py: Validação de DataFrames (batch/training)
"""

# ── Pydantic Schemas (API) ──────────────────────
# ── Pandera Schemas (DataFrame) ──────────────────
from src.schemas.common import (
    SCHEMA_REGISTRY,
    get_schema,
    processed_data_schema,
    processed_inference_schema,
)
from src.schemas.input import PredictionInput
from src.schemas.output import (
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
)

__all__ = [
    # Pydantic (Input)
    "PredictionInput",
    # Pydantic (Output)
    "PredictionResponse",
    "ErrorResponse",
    "HealthResponse",
    # Pandera
    "processed_data_schema",
    "processed_inference_schema",
    "SCHEMA_REGISTRY",
    "get_schema",
]
