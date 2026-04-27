"""Testes para src/data/cleaning.py."""

import pandas as pd
import pytest

from src.config import BINARY_COLS, COLS_TO_DROP
from src.data.cleaning import (
    clean,
    convert_total_charges,
    drop_columns,
    drop_duplicates,
    drop_nulls,
    encode_binary_columns,
)

# ════════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def df_str_charges() -> pd.DataFrame:
    """DataFrame com 'Total Charges' como string, incluindo um vazio."""
    return pd.DataFrame(
        {
            "Total Charges": ["100.0", "200.5", ""],
            "Tenure Months": [12, 24, 0],
        }
    )


@pytest.fixture
def df_binary() -> pd.DataFrame:
    """DataFrame com colunas binárias Yes/No e Senior Citizen já como int."""
    return pd.DataFrame(
        {
            "Senior Citizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "Phone Service": ["Yes", "No"],
            "Paperless Billing": ["No", "Yes"],
            "Monthly Charges": [50.0, 80.0],
        }
    )


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """DataFrame mínimo simulando os dados brutos do Telco (33 colunas).

    Contém:
    - 1 linha com Total Charges vazio → será dropada por drop_nulls
    - 1 linha duplicada (linhas 0 e 3 idênticas após limpeza) → será dropada
    - Todas as colunas da blacklist para testar drop_columns
    """
    return pd.DataFrame(
        {
            # Blacklist — serão removidas
            "Count": [1, 1, 1, 1],
            "Country": ["USA"] * 4,
            "State": ["CA"] * 4,
            "CustomerID": ["A", "B", "C", "D"],
            "Lat Long": ["1,2"] * 4,
            "Latitude": [34.0] * 4,
            "Longitude": [-118.0] * 4,
            "Zip Code": [90001] * 4,
            "City": ["LA"] * 4,
            "Churn Score": [50, 60, 70, 50],
            "CLTV": [1000, 2000, 3000, 1000],
            "Churn Reason": [None, None, None, None],
            "Churn Label": ["No", "No", "No", "No"],
            # Total Charges como string (posicao 2 vazia → NaN → drop)
            "Total Charges": ["100.0", "200.0", "", "100.0"],
            # Binárias
            "Senior Citizen": [0, 0, 1, 0],
            "Partner": ["Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "No", "No"],
            "Phone Service": ["Yes", "Yes", "No", "Yes"],
            "Paperless Billing": ["Yes", "No", "Yes", "Yes"],
            # Outras features
            "Gender": ["Male", "Female", "Male", "Male"],
            "Tenure Months": [12, 24, 0, 12],
            "Multiple Lines": ["No", "Yes", "No phone service", "No"],
            "Internet Service": ["DSL", "Fiber optic", "No", "DSL"],
            "Online Security": ["No", "Yes", "No internet service", "No"],
            "Online Backup": ["No", "Yes", "No internet service", "No"],
            "Device Protection": ["No", "No", "No internet service", "No"],
            "Tech Support": ["No", "No", "No internet service", "No"],
            "Streaming TV": ["No", "No", "No internet service", "No"],
            "Streaming Movies": ["No", "No", "No internet service", "No"],
            "Contract": [
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Month-to-month",
            ],
            "Payment Method": [
                "Electronic check",
                "Mailed check",
                "Electronic check",
                "Electronic check",
            ],
            "Monthly Charges": [50.0, 75.0, 30.0, 50.0],
            "Churn Value": [0, 0, 0, 0],
        }
    )


# ════════════════════════════════════════════════════════════════════════════════
# convert_total_charges
# ════════════════════════════════════════════════════════════════════════════════


class TestConvertTotalCharges:
    """Testes para convert_total_charges()."""

    def test_converte_string_para_float(self, df_str_charges):
        """Deve converter valores string para float64."""
        df = convert_total_charges(df_str_charges)

        assert df["Total Charges"].dtype == "float64"
        assert df["Total Charges"].iloc[0] == pytest.approx(100.0)

    def test_string_vazia_vira_nan(self, df_str_charges):
        """Deve converter string vazia para NaN via coerce."""
        df = convert_total_charges(df_str_charges)

        assert df["Total Charges"].isna().sum() == 1
        assert pd.isna(df["Total Charges"].iloc[2])

    def test_coluna_ausente_lanca_value_error(self):
        """Deve levantar ValueError se 'Total Charges' não existir."""
        df_sem_coluna = pd.DataFrame({"Outra Coluna": [1, 2]})

        with pytest.raises(ValueError, match="Total Charges"):
            convert_total_charges(df_sem_coluna)

    def test_nao_modifica_dataframe_original(self, df_str_charges):
        """Deve retornar cópia — o DataFrame original não deve ser alterado."""
        dtype_original = df_str_charges["Total Charges"].dtype
        convert_total_charges(df_str_charges)

        assert df_str_charges["Total Charges"].dtype == dtype_original


# ════════════════════════════════════════════════════════════════════════════════
# drop_nulls
# ════════════════════════════════════════════════════════════════════════════════


class TestDropNulls:
    """Testes para drop_nulls()."""

    def test_remove_linhas_com_nan(self):
        """Deve remover linhas onde Total Charges é NaN."""
        df = pd.DataFrame({"Total Charges": [1.0, float("nan"), 3.0]})

        result = drop_nulls(df)

        assert len(result) == 2
        assert result["Total Charges"].isna().sum() == 0

    def test_reinicia_indice(self):
        """Deve reiniciar o índice após remoção."""
        df = pd.DataFrame({"Total Charges": [1.0, float("nan"), 3.0]})

        result = drop_nulls(df)

        assert list(result.index) == list(range(len(result)))

    def test_sem_nulos_retorna_mesmo_tamanho(self):
        """Não deve remover linhas se não houver NaN."""
        df = pd.DataFrame({"Total Charges": [1.0, 2.0, 3.0]})

        result = drop_nulls(df)

        assert len(result) == 3


# ════════════════════════════════════════════════════════════════════════════════
# drop_columns
# ════════════════════════════════════════════════════════════════════════════════


class TestDropColumns:
    """Testes para drop_columns()."""

    def test_remove_colunas_conhecidas(self):
        """Deve remover colunas da blacklist presentes no DataFrame."""
        df = pd.DataFrame(
            {
                "Country": ["USA"],
                "CustomerID": ["A"],
                "Monthly Charges": [50.0],
            }
        )

        result = drop_columns(df)

        assert "Country" not in result.columns
        assert "CustomerID" not in result.columns

    def test_ignora_colunas_ausentes(self):
        """Não deve lançar erro se coluna da blacklist não existir no DataFrame."""
        df = pd.DataFrame({"Monthly Charges": [50.0]})

        result = drop_columns(df)  # Country, CustomerID etc. não existem

        assert "Monthly Charges" in result.columns

    def test_preserva_colunas_fora_da_blacklist(self):
        """Colunas não listadas na blacklist devem ser mantidas."""
        df = pd.DataFrame(
            {
                "Country": ["USA"],
                "Partner": ["Yes"],
                "Monthly Charges": [50.0],
            }
        )

        result = drop_columns(df)

        assert "Partner" in result.columns
        assert "Monthly Charges" in result.columns
        assert "Country" not in result.columns


# ════════════════════════════════════════════════════════════════════════════════
# drop_duplicates
# ════════════════════════════════════════════════════════════════════════════════


class TestDropDuplicates:
    """Testes para drop_duplicates()."""

    def test_remove_linhas_duplicadas(self):
        """Deve remover linhas com perfil de features idêntico."""
        df = pd.DataFrame(
            {
                "Partner": ["Yes", "Yes", "No"],
                "Monthly Charges": [50.0, 50.0, 80.0],
            }
        )

        result = drop_duplicates(df)

        assert len(result) == 2

    def test_mantém_primeira_ocorrencia(self):
        """Deve manter a primeira ocorrência do par duplicado."""
        df = pd.DataFrame(
            {
                "Partner": ["Yes", "Yes"],
                "Monthly Charges": [50.0, 50.0],
            }
        )

        result = drop_duplicates(df)

        assert result.iloc[0]["Partner"] == "Yes"
        assert result.iloc[0]["Monthly Charges"] == pytest.approx(50.0)

    def test_reinicia_indice(self):
        """Deve reiniciar o índice após remoção."""
        df = pd.DataFrame(
            {
                "Partner": ["Yes", "No", "Yes"],
                "Monthly Charges": [50.0, 80.0, 50.0],
            }
        )

        result = drop_duplicates(df)

        assert list(result.index) == list(range(len(result)))

    def test_sem_duplicatas_retorna_mesmo_tamanho(self):
        """Não deve remover linhas se não houver duplicatas."""
        df = pd.DataFrame(
            {
                "Partner": ["Yes", "No"],
                "Monthly Charges": [50.0, 80.0],
            }
        )

        result = drop_duplicates(df)

        assert len(result) == 2


# ════════════════════════════════════════════════════════════════════════════════
# encode_binary_columns
# ════════════════════════════════════════════════════════════════════════════════


class TestEncodeBinaryColumns:
    """Testes para encode_binary_columns()."""

    def test_yes_vira_1(self, df_binary):
        """Deve converter 'Yes' para 1."""
        result = encode_binary_columns(df_binary)

        assert result["Partner"].iloc[0] == 1
        assert result["Phone Service"].iloc[0] == 1

    def test_no_vira_0(self, df_binary):
        """Deve converter 'No' para 0."""
        result = encode_binary_columns(df_binary)

        assert result["Partner"].iloc[1] == 0
        assert result["Dependents"].iloc[0] == 0

    def test_dtype_resultado_e_int(self, df_binary):
        """Colunas binárias devem ter dtype int64 após encoding."""
        result = encode_binary_columns(df_binary)

        for col in ["Partner", "Dependents", "Phone Service", "Paperless Billing"]:
            assert result[col].dtype == "int64", f"{col} deveria ser int64"

    def test_ignora_colunas_ausentes(self):
        """Não deve lançar erro se coluna binária não existir no DataFrame."""
        df = pd.DataFrame({"Monthly Charges": [50.0]})

        result = encode_binary_columns(df)  # nenhuma binária presente

        assert "Monthly Charges" in result.columns
        assert "Partner" not in result.columns

    def test_nao_modifica_dataframe_original(self, df_binary):
        """Deve retornar cópia — o DataFrame original não deve ser alterado."""
        original_dtype = df_binary["Partner"].dtype
        encode_binary_columns(df_binary)

        assert df_binary["Partner"].dtype == original_dtype


# ════════════════════════════════════════════════════════════════════════════════
# clean (integração)
# ════════════════════════════════════════════════════════════════════════════════


class TestClean:
    """Testes de integração para clean()."""

    def test_retorna_dataframe(self, raw_df):
        """Deve retornar um DataFrame."""
        result = clean(raw_df)

        assert isinstance(result, pd.DataFrame)

    def test_total_charges_e_float64(self, raw_df):
        """Total Charges deve ser float64 após limpeza."""
        result = clean(raw_df)

        assert result["Total Charges"].dtype == "float64"

    def test_sem_nulos_em_total_charges(self, raw_df):
        """Não deve restar NaN em Total Charges."""
        result = clean(raw_df)

        assert result["Total Charges"].isna().sum() == 0

    def test_remove_colunas_da_blacklist(self, raw_df):
        """Colunas da blacklist não devem estar no resultado."""

        result = clean(raw_df)

        for col in COLS_TO_DROP:
            assert col not in result.columns, (
                f"Coluna '{col}' deveria ter sido removida"
            )

    def test_colunas_binarias_sao_int64(self, raw_df):
        """Variáveis binárias devem ser int64 após encoding."""

        result = clean(raw_df)

        for col in BINARY_COLS:
            assert result[col].dtype == "int64", f"{col} deveria ser int64"

    def test_remove_linhas_nulas_e_duplicadas(self, raw_df):
        """Deve reduzir de 4 para 2 linhas: 1 nula + 1 duplicada removidas."""
        result = clean(raw_df)

        assert len(result) == 2
