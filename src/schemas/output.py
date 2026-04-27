"""
Schema de validação para saída (resposta) da API de predição.

Este módulo define a estrutura esperada para respostas HTTP das requisições
POST /predict, garantindo consistência e documentação automática.

Pydantic BaseModel
- Validação de tipos automatizada
- Documentação Swagger automática
- Serialização JSON com tratamento de tipos especiais (datetime)
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChurnPrediction(BaseModel):
    """
    Resultado isolado da predição de churn.

    Contém a classe predita, probabilidade e nível de confiança.
    """

    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probabilidade de churn (0.0 a 1.0)"
    )

    prediction: Literal["No Churn", "Churn"] = Field(
        ..., description="Classe predita: Churn ou No Churn"
    )

    threshold_used: float = Field(
        ..., ge=0.0, le=1.0, description="Threshold de decisão aplicado (ex: 0.37)"
    )

    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Nível de confiança da predição (0.0 a 1.0)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "churn_probability": 0.73,
                "prediction": "Churn",
                "threshold_used": 0.37,
                "confidence": 0.73,
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    Resposta completa do endpoint /predict.

    Inclui resultado da predição, metadados (timestamp, versão do modelo)
    e informações de desempenho (latência).

    Esta é a estrutura que será retornada como JSON para o cliente.
    """

    success: Literal[True] = Field(
        default=True, description="Indicador de sucesso da requisição"
    )

    prediction: ChurnPrediction = Field(
        ..., description="Dados da predição (probabilidade, classe, threshold)"
    )

    timestamp: datetime = Field(..., description="Data e hora da predição (ISO 8601)")

    model_version: str = Field(
        ..., description="Versão do modelo usado (ex: mlp-64-32-v1)"
    )

    latency_ms: float | None = Field(
        None, ge=0, description="Tempo de processamento em milissegundos (opcional)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "prediction": {
                    "churn_probability": 0.73,
                    "prediction": "Churn",
                    "threshold_used": 0.37,
                    "confidence": 0.73,
                },
                "timestamp": "2026-04-16T14:30:00",
                "model_version": "mlp-64-32-v1",
                "latency_ms": 145.2,
            }
        }
    }


class ErrorResponse(BaseModel):
    """
    Resposta de erro do endpoint /predict.

    Retornada quando a requisição falha (validação, erro interno, etc).
    Permite que cliente saiba exatamente o que deu errado.
    """

    success: Literal[False] = Field(
        default=False, description="Indicador de sucesso (sempre False para erros)"
    )

    error_code: str = Field(
        ..., description="Código do erro (ex: VALIDATION_ERROR, INFERENCE_ERROR)"
    )

    error_message: str = Field(
        ..., description="Mensagem descritiva do erro em linguagem natural"
    )

    timestamp: datetime = Field(..., description="Data e hora do erro (ISO 8601)")

    details: dict | None = Field(
        None, description="Detalhes adicionais do erro (opcional)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_message": "tenure_months deve estar entre 0 e 120",
                "timestamp": "2026-04-16T14:30:00",
                "details": {
                    "field": "tenure_months",
                    "value": 150,
                    "constraint": "le=120",
                },
            }
        }
    }


class HealthResponse(BaseModel):
    """
    Resposta do endpoint /health (health check).

    Simples verificação se API está funcionando.
    """

    status: Literal["ok", "degraded", "error"] = Field(
        ..., description="Status do serviço"
    )

    timestamp: datetime = Field(..., description="Timestamp da verificação")

    model_version: str | None = Field(
        None, description="Versão do modelo carregado (se disponível)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "timestamp": "2026-04-16T14:30:00",
                "model_version": "mlp-64-32-v1",
            }
        }
    }
