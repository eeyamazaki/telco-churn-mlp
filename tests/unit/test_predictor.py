"""
Testes unitários para src/inference/predictor.py (ChurnPredictor).

Carrega os artefatos reais de models/ (preprocessor_mlp.pkl, mlp_weights.pt,
mlp_config.json) e valida o comportamento do pipeline de inferência completo.

O predictor é instanciado uma única vez por módulo (scope="module") para
evitar recarregar os artefatos a cada teste.
"""

import pandas as pd
import pytest
from pandera.errors import SchemaError

from src.inference.predictor import DEFAULT_THRESHOLD, ChurnPredictor

# ════════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def predictor() -> ChurnPredictor:
    """Instância única do predictor para todos os testes — evita recarregar artefatos."""
    return ChurnPredictor()


@pytest.fixture
def cliente_alto_risco() -> pd.DataFrame:
    """Cliente com perfil de alto risco: contrato mensal, fiber optic, sem serviços."""
    return pd.DataFrame(
        [
            {
                "Gender": "Female",
                "Multiple Lines": "No",
                "Senior Citizen": 0,
                "Partner": 0,
                "Dependents": 0,
                "Tenure Months": 2,
                "Monthly Charges": 95.0,
                "Total Charges": 190.0,
                "Phone Service": 1,
                "Paperless Billing": 1,
                "Internet Service": "Fiber optic",
                "Online Security": "No",
                "Online Backup": "No",
                "Device Protection": "No",
                "Tech Support": "No",
                "Streaming TV": "No",
                "Streaming Movies": "No",
                "Contract": "Month-to-month",
                "Payment Method": "Electronic check",
            }
        ]
    )


@pytest.fixture
def cliente_baixo_risco() -> pd.DataFrame:
    """Cliente com perfil de baixo risco: contrato longo, muitos serviços."""
    return pd.DataFrame(
        [
            {
                "Gender": "Male",
                "Multiple Lines": "Yes",
                "Senior Citizen": 0,
                "Partner": 1,
                "Dependents": 1,
                "Tenure Months": 60,
                "Monthly Charges": 85.0,
                "Total Charges": 5100.0,
                "Phone Service": 1,
                "Paperless Billing": 0,
                "Internet Service": "DSL",
                "Online Security": "Yes",
                "Online Backup": "Yes",
                "Device Protection": "Yes",
                "Tech Support": "Yes",
                "Streaming TV": "No",
                "Streaming Movies": "No",
                "Contract": "Two year",
                "Payment Method": "Bank transfer (automatic)",
            }
        ]
    )


# ════════════════════════════════════════════════════════════════════════════════
# INICIALIZAÇÃO
# ════════════════════════════════════════════════════════════════════════════════


class TestChurnPredictorInit:
    """Verifica que o predictor carrega os artefatos e o threshold corretamente."""

    def test_threshold_carregado_do_config(self, predictor):
        """O threshold deve ser o valor F1-ótimo salvo em mlp_config.json."""
        assert predictor.threshold == DEFAULT_THRESHOLD

    def test_threshold_entre_0_e_1(self, predictor):
        assert 0.0 < predictor.threshold < 1.0

    def test_threshold_customizavel(self):
        """Deve aceitar threshold customizado no construtor."""
        p = ChurnPredictor(threshold=0.3)
        assert p.threshold == 0.3


# ════════════════════════════════════════════════════════════════════════════════
# predict() — estrutura do retorno
# ════════════════════════════════════════════════════════════════════════════════


class TestPredictOutput:
    """Verifica que predict() retorna um DataFrame com a estrutura correta."""

    def test_retorna_dataframe(self, predictor, cliente_alto_risco):
        result = predictor.predict(cliente_alto_risco)
        assert isinstance(result, pd.DataFrame)

    def test_colunas_corretas(self, predictor, cliente_alto_risco):
        result = predictor.predict(cliente_alto_risco)
        assert set(result.columns) == {"churn_probability", "churn_prediction"}

    def test_uma_linha_por_cliente(self, predictor, cliente_alto_risco):
        result = predictor.predict(cliente_alto_risco)
        assert len(result) == 1

    def test_preserva_indice_original(self, predictor, cliente_alto_risco):
        """O índice do DataFrame de entrada deve ser preservado no resultado."""
        cliente_alto_risco.index = [99]
        result = predictor.predict(cliente_alto_risco)
        assert list(result.index) == [99]


# ════════════════════════════════════════════════════════════════════════════════
# predict() — valores
# ════════════════════════════════════════════════════════════════════════════════


class TestPredictValues:
    """Verifica que os valores retornados são semanticamente corretos."""

    def test_probabilidade_entre_0_e_1(self, predictor, cliente_alto_risco):
        result = predictor.predict(cliente_alto_risco)
        prob = result["churn_probability"].iloc[0]
        assert 0.0 <= prob <= 1.0

    def test_predicao_binaria(self, predictor, cliente_alto_risco):
        """churn_prediction deve ser 0 ou 1."""
        result = predictor.predict(cliente_alto_risco)
        pred = result["churn_prediction"].iloc[0]
        assert pred in (0, 1)

    def test_predicao_consistente_com_threshold(self, predictor, cliente_alto_risco):
        """churn_prediction deve ser 1 se probabilidade >= threshold, 0 caso contrário."""
        result = predictor.predict(cliente_alto_risco)
        prob = result["churn_probability"].iloc[0]
        pred = result["churn_prediction"].iloc[0]
        expected = 1 if prob >= predictor.threshold else 0
        assert pred == expected

    def test_alto_risco_prediz_churn(self, predictor, cliente_alto_risco):
        """Perfil de alto risco (fiber optic + mês a mês + tenure=2) deve predizer churn."""
        result = predictor.predict(cliente_alto_risco)
        assert result["churn_prediction"].iloc[0] == 1

    def test_baixo_risco_prediz_no_churn(self, predictor, cliente_baixo_risco):
        """Perfil de baixo risco (Two year + muitos serviços + tenure=60) não deve churnar."""
        result = predictor.predict(cliente_baixo_risco)
        assert result["churn_prediction"].iloc[0] == 0


# ════════════════════════════════════════════════════════════════════════════════
# predict() — batch
# ════════════════════════════════════════════════════════════════════════════════


class TestPredictBatch:
    """Verifica inferência em lote (múltiplos clientes)."""

    def test_batch_retorna_linhas_corretas(
        self, predictor, cliente_alto_risco, cliente_baixo_risco
    ):
        """Batch com 2 clientes deve retornar 2 linhas."""
        batch = pd.concat([cliente_alto_risco, cliente_baixo_risco], ignore_index=True)
        result = predictor.predict(batch)
        assert len(result) == 2

    def test_batch_todas_probabilidades_validas(
        self, predictor, cliente_alto_risco, cliente_baixo_risco
    ):
        batch = pd.concat([cliente_alto_risco, cliente_baixo_risco], ignore_index=True)
        result = predictor.predict(batch)
        assert (result["churn_probability"] >= 0.0).all()
        assert (result["churn_probability"] <= 1.0).all()


# ════════════════════════════════════════════════════════════════════════════════
# predict_single()
# ════════════════════════════════════════════════════════════════════════════════


class TestPredictSingle:
    """Verifica o atalho de inferência para um único cliente via dicionário."""

    def test_retorna_dicionario(self, predictor, cliente_alto_risco):
        result = predictor.predict_single(cliente_alto_risco.iloc[0].to_dict())
        assert isinstance(result, dict)

    def test_chaves_corretas(self, predictor, cliente_alto_risco):
        result = predictor.predict_single(cliente_alto_risco.iloc[0].to_dict())
        assert set(result.keys()) == {"churn_probability", "churn_prediction"}

    def test_probabilidade_arredondada(self, predictor, cliente_alto_risco):
        """churn_probability deve ter no máximo 4 casas decimais."""
        result = predictor.predict_single(cliente_alto_risco.iloc[0].to_dict())
        prob = result["churn_probability"]
        assert prob == round(prob, 4)

    def test_predicao_e_inteiro(self, predictor, cliente_alto_risco):
        result = predictor.predict_single(cliente_alto_risco.iloc[0].to_dict())
        assert isinstance(result["churn_prediction"], int)


# ════════════════════════════════════════════════════════════════════════════════
# Validação de schema — erros esperados
# ════════════════════════════════════════════════════════════════════════════════


class TestSchemaValidation:
    """O predictor deve rejeitar DataFrames inválidos antes de chegar no modelo."""

    def test_coluna_faltando_levanta_schema_error(self, predictor, cliente_alto_risco):
        df_invalido = cliente_alto_risco.drop(columns=["Contract"])
        with pytest.raises(SchemaError):
            predictor.predict(df_invalido)

    def test_tenure_zero_levanta_schema_error(self, predictor, cliente_alto_risco):
        """tenure_months=0 viola o schema (ge=1) e deve ser barrado antes do modelo."""
        df_invalido = cliente_alto_risco.copy()
        df_invalido["Tenure Months"] = 0
        with pytest.raises(SchemaError):
            predictor.predict(df_invalido)

    def test_tenure_acima_limite_levanta_schema_error(
        self, predictor, cliente_alto_risco
    ):
        df_invalido = cliente_alto_risco.copy()
        df_invalido["Tenure Months"] = 73
        with pytest.raises(SchemaError):
            predictor.predict(df_invalido)

    def test_contrato_invalido_levanta_schema_error(
        self, predictor, cliente_alto_risco
    ):
        df_invalido = cliente_alto_risco.copy()
        df_invalido["Contract"] = "Semestral"
        with pytest.raises(SchemaError):
            predictor.predict(df_invalido)
