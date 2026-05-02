"""
Smoke tests para a API FastAPI (src/api/main.py).

Testa o pipeline completo de ponta a ponta em memória usando TestClient:
    JSON de entrada → validação Pydantic → FeatureEngineer
    → preprocessor → ChurnMLP (PyTorch) → resposta JSON

Não requer servidor em execução — o TestClient sobe a aplicação internamente.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.inference.predictor import DEFAULT_THRESHOLD

client = TestClient(app)


# ════════════════════════════════════════════════════════════════════════════════
# AUTENTICAÇÃO
# ════════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def auth_headers() -> dict:
    """Realiza login e retorna o header Authorization com token JWT."""
    response = client.post("/login", json={"username": "user", "password": "user123"})
    assert response.status_code == 200, f"Login falhou: {response.json()}"
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ════════════════════════════════════════════════════════════════════════════════
# PAYLOAD BASE
# ════════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def payload_alto_risco() -> dict:
    """Cliente com alto risco de churn: fiber optic + mês a mês + sem serviços."""
    return {
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "No",
        "dependents": "No",
        "tenure_months": 2,
        "monthly_charges": 95.0,
        "total_charges": 190.0,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "contract": "Month-to-month",
        "internet_service_type": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
    }


@pytest.fixture
def payload_baixo_risco() -> dict:
    """Cliente com baixo risco de churn: contrato longo + muitos serviços."""
    return {
        "gender": "Male",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "Yes",
        "tenure_months": 60,
        "monthly_charges": 85.0,
        "total_charges": 5100.0,
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "paperless_billing": "No",
        "payment_method": "Bank transfer (automatic)",
        "contract": "Two year",
        "internet_service_type": "DSL",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "No",
        "streaming_movies": "No",
    }


# ════════════════════════════════════════════════════════════════════════════════
# /health
# ════════════════════════════════════════════════════════════════════════════════


class TestHealthEndpoint:
    def test_status_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_ok(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_campos_obrigatorios_presentes(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_version" in data

    def test_model_version_nao_vazio(self):
        data = client.get("/health").json()
        assert data["model_version"] != ""


# ════════════════════════════════════════════════════════════════════════════════
# /predict — casos de sucesso
# ════════════════════════════════════════════════════════════════════════════════


class TestPredictEndpointSuccess:
    def test_status_200(self, payload_alto_risco, auth_headers):
        response = client.post(
            "/predict", json=payload_alto_risco, headers=auth_headers
        )
        assert response.status_code == 200

    def test_estrutura_resposta(self, payload_alto_risco, auth_headers):
        """Resposta deve conter todos os campos do PredictionResponse."""
        data = client.post(
            "/predict", json=payload_alto_risco, headers=auth_headers
        ).json()
        assert data["success"] is True
        assert "prediction" in data
        assert "timestamp" in data
        assert "model_version" in data
        assert "latency_ms" in data

    def test_estrutura_predicao_aninhada(self, payload_alto_risco, auth_headers):
        """O objeto 'prediction' deve ter os 4 campos de ChurnPrediction."""
        pred = client.post(
            "/predict", json=payload_alto_risco, headers=auth_headers
        ).json()["prediction"]
        assert "churn_probability" in pred
        assert "prediction" in pred
        assert "threshold_used" in pred
        assert "confidence" in pred

    def test_probabilidade_entre_0_e_1(self, payload_alto_risco, auth_headers):
        pred = client.post(
            "/predict", json=payload_alto_risco, headers=auth_headers
        ).json()["prediction"]
        assert 0.0 <= pred["churn_probability"] <= 1.0

    def test_prediction_label_valido(self, payload_alto_risco, auth_headers):
        pred = client.post(
            "/predict", json=payload_alto_risco, headers=auth_headers
        ).json()["prediction"]
        assert pred["prediction"] in ("Churn", "No Churn")

    def test_threshold_correto(self, payload_alto_risco, auth_headers):
        """O threshold retornado deve ser o DEFAULT_THRESHOLD carregado do mlp_config.json."""
        pred = client.post(
            "/predict", json=payload_alto_risco, headers=auth_headers
        ).json()["prediction"]
        assert pred["threshold_used"] == pytest.approx(DEFAULT_THRESHOLD)

    def test_latencia_positiva(self, payload_alto_risco, auth_headers):
        data = client.post(
            "/predict", json=payload_alto_risco, headers=auth_headers
        ).json()
        assert data["latency_ms"] > 0

    def test_alto_risco_prediz_churn(self, payload_alto_risco, auth_headers):
        """Cliente de alto risco deve ser classificado como Churn."""
        pred = client.post(
            "/predict", json=payload_alto_risco, headers=auth_headers
        ).json()["prediction"]
        assert pred["prediction"] == "Churn"
        assert pred["churn_probability"] > 0.5

    def test_baixo_risco_prediz_no_churn(self, payload_baixo_risco, auth_headers):
        """Cliente de baixo risco deve ser classificado como No Churn."""
        pred = client.post(
            "/predict", json=payload_baixo_risco, headers=auth_headers
        ).json()["prediction"]
        assert pred["prediction"] == "No Churn"


# ════════════════════════════════════════════════════════════════════════════════
# /predict — validação de entrada (422)
# ════════════════════════════════════════════════════════════════════════════════


class TestPredictEndpointValidation:
    def test_campo_faltando_retorna_422(self, payload_alto_risco, auth_headers):
        del payload_alto_risco["contract"]
        response = client.post(
            "/predict", json=payload_alto_risco, headers=auth_headers
        )
        assert response.status_code == 422

    def test_tenure_zero_retorna_422(self, payload_alto_risco, auth_headers):
        """tenure_months=0 viola ge=1 — deve retornar 422."""
        payload_alto_risco["tenure_months"] = 0
        response = client.post(
            "/predict", json=payload_alto_risco, headers=auth_headers
        )
        assert response.status_code == 422

    def test_tenure_acima_do_limite_retorna_422(self, payload_alto_risco, auth_headers):
        """tenure_months=73 viola le=72 (detector de drift) — deve retornar 422."""
        payload_alto_risco["tenure_months"] = 73
        response = client.post(
            "/predict", json=payload_alto_risco, headers=auth_headers
        )
        assert response.status_code == 422

    def test_internet_service_invalido_retorna_422(
        self, payload_alto_risco, auth_headers
    ):
        payload_alto_risco["internet_service_type"] = "5G"
        response = client.post(
            "/predict", json=payload_alto_risco, headers=auth_headers
        )
        assert response.status_code == 422

    def test_contrato_invalido_retorna_422(self, payload_alto_risco, auth_headers):
        payload_alto_risco["contract"] = "Six months"
        response = client.post(
            "/predict", json=payload_alto_risco, headers=auth_headers
        )
        assert response.status_code == 422

    def test_internet_sem_servicos_consistente(self, payload_alto_risco, auth_headers):
        """internet_service_type='No' com online_security='Yes' deve retornar 422."""
        payload_alto_risco["internet_service_type"] = "No"
        payload_alto_risco["online_security"] = "Yes"
        response = client.post(
            "/predict", json=payload_alto_risco, headers=auth_headers
        )
        assert response.status_code == 422

    def test_total_charges_inconsistente_retorna_422(
        self, payload_alto_risco, auth_headers
    ):
        """total_charges muito baixo vs monthly*tenure deve retornar 422."""
        payload_alto_risco["tenure_months"] = 24
        payload_alto_risco["monthly_charges"] = 100.0
        payload_alto_risco["total_charges"] = 500.0  # esperado mínimo: 2160
        response = client.post(
            "/predict", json=payload_alto_risco, headers=auth_headers
        )
        assert response.status_code == 422
