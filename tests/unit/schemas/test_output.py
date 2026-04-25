"""
Testes para src/schemas/output.py (PredictionResponse, ErrorResponse, HealthResponse).
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.schemas.output import (
    ChurnPrediction,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
)

# ════════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def valid_churn_prediction_data() -> dict:
    return {
        "churn_probability": 0.73,
        "prediction": "Churn",
        "threshold_used": 0.37,
        "confidence": 0.73,
    }


@pytest.fixture
def valid_prediction_response_data(valid_churn_prediction_data) -> dict:
    return {
        "success": True,
        "prediction": valid_churn_prediction_data,
        "timestamp": datetime(2026, 4, 22, 14, 30, 0),
        "model_version": "mlp-64-32-v1",
        "latency_ms": 145.2,
    }


@pytest.fixture
def valid_error_response_data() -> dict:
    return {
        "error_code": "VALIDATION_ERROR",
        "error_message": "tenure_months deve estar entre 0 e 72",
        "timestamp": datetime(2026, 4, 22, 14, 30, 0),
        "details": {"field": "tenure_months", "value": 150},
    }


@pytest.fixture
def valid_health_response_data() -> dict:
    return {
        "status": "ok",
        "timestamp": datetime(2026, 4, 22, 14, 30, 0),
        "model_version": "mlp-64-32-v1",
    }


# ════════════════════════════════════════════════════════════════════════════════
# ChurnPrediction
# ════════════════════════════════════════════════════════════════════════════════

class TestChurnPredictionCreation:
    """Testes para criação de ChurnPrediction."""

    def test_create_with_valid_data(self, valid_churn_prediction_data):
        pred = ChurnPrediction(**valid_churn_prediction_data)
        assert pred.churn_probability == 0.73
        assert pred.prediction == "Churn"

    @pytest.mark.parametrize("valid_prob", [0.0, 0.01, 0.5, 0.99, 1.0])
    def test_churn_probability_valid_values(self, valid_churn_prediction_data, valid_prob):
        valid_churn_prediction_data["churn_probability"] = valid_prob
        pred = ChurnPrediction(**valid_churn_prediction_data)
        assert pred.churn_probability == valid_prob

    @pytest.mark.parametrize("invalid_prob", [-0.01, 1.01, 2.0, "alto", None, [1, 2]])
    def test_churn_probability_invalid_values(self, valid_churn_prediction_data, invalid_prob):
        valid_churn_prediction_data["churn_probability"] = invalid_prob
        with pytest.raises(ValidationError):
            ChurnPrediction(**valid_churn_prediction_data)

    @pytest.mark.parametrize("valid_pred", ["Churn", "No Churn"])
    def test_prediction_valid_values(self, valid_churn_prediction_data, valid_pred):
        valid_churn_prediction_data["prediction"] = valid_pred
        pred = ChurnPrediction(**valid_churn_prediction_data)
        assert pred.prediction == valid_pred

    @pytest.mark.parametrize("invalid_pred", ["churn", "yes", "1", "", None])
    def test_prediction_invalid_values(self, valid_churn_prediction_data, invalid_pred):
        valid_churn_prediction_data["prediction"] = invalid_pred
        with pytest.raises(ValidationError):
            ChurnPrediction(**valid_churn_prediction_data)

    @pytest.mark.parametrize("valid_threshold", [0.0, 0.37, 0.5, 1.0])
    def test_threshold_valid_values(self, valid_churn_prediction_data, valid_threshold):
        valid_churn_prediction_data["threshold_used"] = valid_threshold
        pred = ChurnPrediction(**valid_churn_prediction_data)
        assert pred.threshold_used == valid_threshold

    @pytest.mark.parametrize("invalid_threshold", [-0.01, 1.01, "medio", None])
    def test_threshold_invalid_values(self, valid_churn_prediction_data, invalid_threshold):
        valid_churn_prediction_data["threshold_used"] = invalid_threshold
        with pytest.raises(ValidationError):
            ChurnPrediction(**valid_churn_prediction_data)

    @pytest.mark.parametrize("valid_confidence", [0.0, 0.5, 0.90, 0.95, 0.99, 1.0])
    def test_confidence_valid_values(self, valid_churn_prediction_data, valid_confidence):
        valid_churn_prediction_data["confidence"] = valid_confidence
        pred = ChurnPrediction(**valid_churn_prediction_data)
        assert pred.confidence == valid_confidence

    @pytest.mark.parametrize("invalid_confidence", [-0.01, 1.01, "medio", None])
    def test_confidence_invalid_values(self, valid_churn_prediction_data, invalid_confidence):
        valid_churn_prediction_data["confidence"] = invalid_confidence
        with pytest.raises(ValidationError):
            ChurnPrediction(**valid_churn_prediction_data)


# ════════════════════════════════════════════════════════════════════════════════
# PredictionResponse
# ════════════════════════════════════════════════════════════════════════════════

class TestPredictionResponseCreation:
    """Testes para criação de PredictionResponse."""

    def test_create_with_valid_data(self, valid_prediction_response_data):
        response = PredictionResponse(**valid_prediction_response_data)
        assert response.success is True
        assert response.model_version == "mlp-64-32-v1"
        assert isinstance(response.prediction, ChurnPrediction)

    def test_success_defaults_to_true(self, valid_prediction_response_data):
        del valid_prediction_response_data["success"]
        response = PredictionResponse(**valid_prediction_response_data)
        assert response.success is True

    def test_success_cannot_be_false(self, valid_prediction_response_data):
        valid_prediction_response_data["success"] = False
        with pytest.raises(ValidationError):
            PredictionResponse(**valid_prediction_response_data)

    def test_latency_ms_is_optional(self, valid_prediction_response_data):
        del valid_prediction_response_data["latency_ms"]
        response = PredictionResponse(**valid_prediction_response_data)
        assert response.latency_ms is None

    @pytest.mark.parametrize("valid_latency", [0.0, 50.0, 1000.0])
    def test_latency_ms_valid_values(self, valid_prediction_response_data, valid_latency):
        valid_prediction_response_data["latency_ms"] = valid_latency
        response = PredictionResponse(**valid_prediction_response_data)
        assert response.latency_ms == valid_latency

    @pytest.mark.parametrize("invalid_latency", [-1.0, "rapido"])
    def test_latency_ms_invalid_values(self, valid_prediction_response_data, invalid_latency):
        valid_prediction_response_data["latency_ms"] = invalid_latency
        with pytest.raises(ValidationError):
            PredictionResponse(**valid_prediction_response_data)

    @pytest.mark.parametrize("missing_field", ["prediction", "timestamp", "model_version"])
    def test_missing_required_fields(self, valid_prediction_response_data, missing_field):
        del valid_prediction_response_data[missing_field]
        with pytest.raises(ValidationError):
            PredictionResponse(**valid_prediction_response_data)

    def test_timestamp_accepts_string_iso(self, valid_prediction_response_data):
        valid_prediction_response_data["timestamp"] = "2026-04-22T14:30:00"
        response = PredictionResponse(**valid_prediction_response_data)
        assert isinstance(response.timestamp, datetime)

    def test_nested_prediction_validated(self, valid_prediction_response_data):
        valid_prediction_response_data["prediction"]["churn_probability"] = 1.5
        with pytest.raises(ValidationError):
            PredictionResponse(**valid_prediction_response_data)


# ════════════════════════════════════════════════════════════════════════════════
# ErrorResponse
# ════════════════════════════════════════════════════════════════════════════════

class TestErrorResponseCreation:
    """Testes para criação de ErrorResponse."""

    def test_create_with_valid_data(self, valid_error_response_data):
        response = ErrorResponse(**valid_error_response_data)
        assert response.success is False
        assert response.error_code == "VALIDATION_ERROR"

    def test_success_defaults_to_false(self, valid_error_response_data):
        response = ErrorResponse(**valid_error_response_data)
        assert response.success is False

    def test_success_cannot_be_true(self, valid_error_response_data):
        valid_error_response_data["success"] = True
        with pytest.raises(ValidationError):
            ErrorResponse(**valid_error_response_data)

    def test_details_is_optional(self, valid_error_response_data):
        del valid_error_response_data["details"]
        response = ErrorResponse(**valid_error_response_data)
        assert response.details is None

    @pytest.mark.parametrize("missing_field", ["error_code", "error_message", "timestamp"])
    def test_missing_required_fields(self, valid_error_response_data, missing_field):
        del valid_error_response_data[missing_field]
        with pytest.raises(ValidationError):
            ErrorResponse(**valid_error_response_data)


# ════════════════════════════════════════════════════════════════════════════════
# HealthResponse
# ════════════════════════════════════════════════════════════════════════════════

class TestHealthResponseCreation:
    """Testes para criação de HealthResponse."""

    def test_create_with_valid_data(self, valid_health_response_data):
        response = HealthResponse(**valid_health_response_data)
        assert response.status == "ok"
        assert response.model_version == "mlp-64-32-v1"

    @pytest.mark.parametrize("valid_status", ["ok", "degraded", "error"])
    def test_status_valid_values(self, valid_health_response_data, valid_status):
        valid_health_response_data["status"] = valid_status
        response = HealthResponse(**valid_health_response_data)
        assert response.status == valid_status

    @pytest.mark.parametrize("invalid_status", ["running", "down", "healthy", "", None])
    def test_status_invalid_values(self, valid_health_response_data, invalid_status):
        valid_health_response_data["status"] = invalid_status
        with pytest.raises(ValidationError):
            HealthResponse(**valid_health_response_data)

    def test_model_version_is_optional(self, valid_health_response_data):
        del valid_health_response_data["model_version"]
        response = HealthResponse(**valid_health_response_data)
        assert response.model_version is None

    @pytest.mark.parametrize("missing_field", ["status", "timestamp"])
    def test_missing_required_fields(self, valid_health_response_data, missing_field):
        del valid_health_response_data[missing_field]
        with pytest.raises(ValidationError):
            HealthResponse(**valid_health_response_data)
