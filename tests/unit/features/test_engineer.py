"""
Testes unitários para src/features/engineer.py (FeatureEngineer).

Valida cada uma das 6 features derivadas individualmente,
garantindo que a lógica de transformação está correta e que
o transformador é compatível com scikit-learn (fit/transform).
"""

import pandas as pd
import pytest

from src.config import OPTIONAL_SERVICES_COLS
from src.features.engineer import FeatureEngineer

# ════════════════════════════════════════════════════════════════════════════════
# FIXTURE BASE
# ════════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_row() -> dict:
    """Linha base válida com todas as colunas exigidas pelo FeatureEngineer."""
    return {
        "Gender": "Female",
        "Senior Citizen": 0,
        "Partner": 1,
        "Dependents": 0,
        "Tenure Months": 24,
        "Monthly Charges": 72.0,
        "Total Charges": 1728.0,
        "Phone Service": 1,
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": 1,
        "Payment Method": "Electronic check",
    }


@pytest.fixture
def base_df(base_row) -> pd.DataFrame:
    return pd.DataFrame([base_row])


@pytest.fixture
def engineer() -> FeatureEngineer:
    return FeatureEngineer()


# ════════════════════════════════════════════════════════════════════════════════
# CONTRATO SKLEARN
# ════════════════════════════════════════════════════════════════════════════════

class TestSklearnContract:
    """Garante compatibilidade com a API do scikit-learn."""

    def test_fit_retorna_self(self, engineer, base_df):
        """fit() deve retornar o próprio transformador."""
        result = engineer.fit(base_df)
        assert result is engineer

    def test_fit_nao_altera_dados(self, engineer, base_df):
        """fit() não deve modificar o DataFrame de entrada."""
        original_cols = list(base_df.columns)
        original_shape = base_df.shape
        engineer.fit(base_df)
        assert list(base_df.columns) == original_cols
        assert base_df.shape == original_shape

    def test_transform_retorna_dataframe(self, engineer, base_df):
        """transform() deve retornar um DataFrame."""
        result = engineer.transform(base_df)
        assert isinstance(result, pd.DataFrame)

    def test_transform_adiciona_6_colunas(self, engineer, base_df):
        """transform() deve adicionar exatamente 6 novas colunas."""
        n_antes = base_df.shape[1]
        result = engineer.transform(base_df)
        assert result.shape[1] == n_antes + 6

    def test_transform_nao_modifica_entrada(self, engineer, base_df):
        """transform() deve trabalhar em cópia, sem modificar o DataFrame original."""
        cols_originais = set(base_df.columns)
        engineer.transform(base_df)
        assert set(base_df.columns) == cols_originais

    def test_fit_transform_equivalente(self, engineer, base_df):
        """fit_transform() deve produzir resultado igual a fit() seguido de transform()."""
        result_fit_transform = engineer.fit_transform(base_df)
        result_manual = engineer.fit(base_df).transform(base_df)
        pd.testing.assert_frame_equal(result_fit_transform, result_manual)


# ════════════════════════════════════════════════════════════════════════════════
# FEATURE 1: services_count
# ════════════════════════════════════════════════════════════════════════════════

class TestServicesCount:
    """services_count: contagem de serviços opcionais ativos (0–6)."""

    def test_zero_servicos(self, engineer, base_df):
        """Sem nenhum serviço ativo → 0."""
        result = engineer.transform(base_df)
        assert result["services_count"].iloc[0] == 0

    @pytest.mark.parametrize("n_servicos", [1, 2, 3, 4, 5, 6])
    def test_n_servicos_ativos(self, engineer, base_row, n_servicos):
        """n serviços marcados como 'Yes' → services_count == n."""
        for svc in OPTIONAL_SERVICES_COLS[:n_servicos]:
            base_row[svc] = "Yes"
        df = pd.DataFrame([base_row])
        result = engineer.transform(df)
        assert result["services_count"].iloc[0] == n_servicos

    def test_no_internet_service_nao_conta(self, engineer, base_row):
        """'No internet service' não deve ser contado como serviço ativo."""
        for svc in OPTIONAL_SERVICES_COLS:
            base_row[svc] = "No internet service"
        df = pd.DataFrame([base_row])
        result = engineer.transform(df)
        assert result["services_count"].iloc[0] == 0


# ════════════════════════════════════════════════════════════════════════════════
# FEATURE 2: tenure_group
# ════════════════════════════════════════════════════════════════════════════════

class TestTenureGroup:
    """tenure_group: new (1-12), growing (13-36), loyal (37-72)."""

    @pytest.mark.parametrize("tenure, grupo_esperado", [
        (1,  "new"),
        (12, "new"),
        (13, "growing"),
        (36, "growing"),
        (37, "loyal"),
        (72, "loyal"),
    ])
    def test_grupos_corretos(self, engineer, base_row, tenure, grupo_esperado):
        base_row["Tenure Months"] = tenure
        df = pd.DataFrame([base_row])
        result = engineer.transform(df)
        assert result["tenure_group"].iloc[0] == grupo_esperado

    def test_tenure_group_dtype_string(self, engineer, base_df):
        """tenure_group deve ser dtype string (object), compatível com pa.String no schema."""
        result = engineer.transform(base_df)
        assert result["tenure_group"].dtype == object


# ════════════════════════════════════════════════════════════════════════════════
# FEATURE 3: monthly_per_tenure
# ════════════════════════════════════════════════════════════════════════════════

class TestMonthlyPerTenure:
    """monthly_per_tenure: Monthly Charges / (Tenure Months + 1)."""

    @pytest.mark.parametrize("monthly, tenure, esperado", [
        (60.0, 11,  5.0),    # 60 / 12 = 5.0
        (60.0,  1, 30.0),    # 60 / 2 = 30.0
        (72.0, 23,  3.0),    # 72 / 24 = 3.0
    ])
    def test_calculo_correto(self, engineer, base_row, monthly, tenure, esperado):
        base_row["Monthly Charges"] = monthly
        base_row["Tenure Months"] = tenure
        df = pd.DataFrame([base_row])
        result = engineer.transform(df)
        assert result["monthly_per_tenure"].iloc[0] == pytest.approx(esperado)

    def test_sem_divisao_por_zero(self, engineer, base_row):
        """Com Tenure Months = 0, o denominador é 1 (não divide por zero)."""
        base_row["Tenure Months"] = 0
        df = pd.DataFrame([base_row])
        result = engineer.transform(df)
        assert result["monthly_per_tenure"].iloc[0] == base_row["Monthly Charges"]


# ════════════════════════════════════════════════════════════════════════════════
# FEATURE 4: has_protection
# ════════════════════════════════════════════════════════════════════════════════

class TestHasProtection:
    """has_protection: 1 se Online Security='Yes' OU Device Protection='Yes'."""

    def test_sem_protecao(self, engineer, base_df):
        result = engineer.transform(base_df)
        assert result["has_protection"].iloc[0] == 0

    def test_apenas_online_security(self, engineer, base_row):
        base_row["Online Security"] = "Yes"
        result = engineer.transform(pd.DataFrame([base_row]))
        assert result["has_protection"].iloc[0] == 1

    def test_apenas_device_protection(self, engineer, base_row):
        base_row["Device Protection"] = "Yes"
        result = engineer.transform(pd.DataFrame([base_row]))
        assert result["has_protection"].iloc[0] == 1

    def test_ambos_ativos(self, engineer, base_row):
        base_row["Online Security"] = "Yes"
        base_row["Device Protection"] = "Yes"
        result = engineer.transform(pd.DataFrame([base_row]))
        assert result["has_protection"].iloc[0] == 1

    def test_resultado_binario(self, engineer, base_df):
        """has_protection deve ser apenas 0 ou 1."""
        result = engineer.transform(base_df)
        assert result["has_protection"].iloc[0] in (0, 1)


# ════════════════════════════════════════════════════════════════════════════════
# FEATURE 5: is_senior_alone
# ════════════════════════════════════════════════════════════════════════════════

class TestIsSeniorAlone:
    """is_senior_alone: 1 apenas se Senior=1, Partner=0, Dependents=0."""

    @pytest.mark.parametrize("senior, partner, dependents, esperado", [
        (1, 0, 0, 1),   # idoso sem suporte → 1
        (1, 1, 0, 0),   # idoso com parceiro → 0
        (1, 0, 1, 0),   # idoso com dependentes → 0
        (1, 1, 1, 0),   # idoso com família → 0
        (0, 0, 0, 0),   # não idoso → sempre 0
        (0, 1, 0, 0),
    ])
    def test_combinacoes(self, engineer, base_row, senior, partner, dependents, esperado):
        base_row["Senior Citizen"] = senior
        base_row["Partner"] = partner
        base_row["Dependents"] = dependents
        result = engineer.transform(pd.DataFrame([base_row]))
        assert result["is_senior_alone"].iloc[0] == esperado


# ════════════════════════════════════════════════════════════════════════════════
# FEATURE 6: contract_risk_score
# ════════════════════════════════════════════════════════════════════════════════

class TestContractRiskScore:
    """contract_risk_score: Month-to-month=3, One year=1, Two year=0."""

    @pytest.mark.parametrize("contrato, score_esperado", [
        ("Month-to-month", 3),
        ("One year",       1),
        ("Two year",       0),
    ])
    def test_scores_corretos(self, engineer, base_row, contrato, score_esperado):
        base_row["Contract"] = contrato
        result = engineer.transform(pd.DataFrame([base_row]))
        assert result["contract_risk_score"].iloc[0] == score_esperado


# ════════════════════════════════════════════════════════════════════════════════
# PROCESSAMENTO EM BATCH
# ════════════════════════════════════════════════════════════════════════════════

class TestBatchProcessing:
    """Garante que o transformador funciona corretamente com múltiplas linhas."""

    def test_multiplas_linhas(self, engineer, base_row):
        """transform() deve processar múltiplas linhas corretamente."""
        rows = []
        for contrato, _score in [("Month-to-month", 3), ("One year", 1), ("Two year", 0)]:
            row = base_row.copy()
            row["Contract"] = contrato
            rows.append(row)
        df = pd.DataFrame(rows)
        result = engineer.transform(df)
        assert list(result["contract_risk_score"]) == [3, 1, 0]

    def test_preserva_indice_original(self, engineer, base_row):
        """O índice do DataFrame original deve ser preservado após transform."""
        df = pd.DataFrame([base_row, base_row], index=[10, 20])
        result = engineer.transform(df)
        assert list(result.index) == [10, 20]
