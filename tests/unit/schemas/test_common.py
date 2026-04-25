"""
Testes para src/schemas/common.py.
"""

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from src.schemas.common import (
    SCHEMA_REGISTRY,
    engineered_data_schema,
    engineered_inference_schema,
    get_schema,
    processed_data_schema,
    processed_inference_schema,
)

# ════════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def valid_processed_row() -> dict:
    return {
        "Gender": "Female",
        "Multiple Lines": "No",
        "Senior Citizen": 0,
        "Partner": 1,
        "Dependents": 0,
        "Tenure Months": 24,
        "Monthly Charges": 65.0,
        "Total Charges": 1560.0,
        "Phone Service": 1,
        "Paperless Billing": 1,
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "Yes",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Payment Method": "Electronic check",
        "Churn Value": 0,
    }


@pytest.fixture
def valid_processed_df(valid_processed_row) -> pd.DataFrame:
    return pd.DataFrame([valid_processed_row])

@pytest.fixture
def valid_processed_inference_df(valid_processed_row) -> pd.DataFrame:
    row = valid_processed_row.copy()
    del row["Churn Value"]
    return pd.DataFrame([row])


@pytest.fixture
def valid_engineered_row(valid_processed_row) -> dict:
    return {
        **valid_processed_row,
        "services_count": 2,
        "tenure_group": "growing",
        "monthly_per_tenure": 65.0 / 24,
        "has_protection": 0,
        "is_senior_alone": 0,
        "contract_risk_score": 3,
    }


@pytest.fixture
def valid_engineered_df(valid_engineered_row) -> pd.DataFrame:
    return pd.DataFrame([valid_engineered_row])


@pytest.fixture
def valid_inference_row(valid_engineered_row) -> dict:
    """Igual ao engineered, mas sem Churn Value."""
    row = valid_engineered_row.copy()
    del row["Churn Value"]
    return row


@pytest.fixture
def valid_inference_df(valid_inference_row) -> pd.DataFrame:
    return pd.DataFrame([valid_inference_row])


# ════════════════════════════════════════════════════════════════════════════════
# processed_data_schema
# ════════════════════════════════════════════════════════════════════════════════

class TestProcessedSchemaCreation:

    def test_valid_dataframe_passes(self, valid_processed_df):
        result = processed_data_schema.validate(valid_processed_df)
        assert isinstance(result, pd.DataFrame)

    def test_extra_column_raises(self, valid_processed_df):
        df = valid_processed_df.copy()
        df["coluna_extra"] = 99
        with pytest.raises((SchemaError, SchemaErrors)):
            processed_data_schema.validate(df)

    def test_missing_column_raises(self, valid_processed_df):
        df = valid_processed_df.drop(columns=["Churn Value"])
        with pytest.raises(SchemaError):
            processed_data_schema.validate(df)

@pytest.mark.parametrize("column", [
        "Senior Citizen", "Partner", "Dependents",
        "Phone Service", "Paperless Billing", "Churn Value",
    ])
class TestProcessedSchemaBinaryColumns:

    def test_valid_binary_values(self, valid_processed_df, column):
        for value in [0, 1]:
            df = valid_processed_df.copy()
            df[column] = value
            processed_data_schema.validate(df)


    def test_invalid_binary_values(self, valid_processed_df, column):
        for value in [2, -1, None, "um"]:
            df = valid_processed_df.copy()
            df[column] = value
            with pytest.raises(SchemaError):
                processed_data_schema.validate(df)

@pytest.mark.parametrize("column", [
        "Online Security", "Online Backup", "Device Protection",
        "Tech Support", "Streaming TV", "Streaming Movies",
    ])
class TestProcessedSchemaInternetServiceColumns:

    def test_valid_values(self, valid_processed_df, column):
        for value in ["Yes", "No", "No internet service"]:
            df = valid_processed_df.copy()
            df[column] = value
            processed_data_schema.validate(df)

    def test_invalid_values(self, valid_processed_df, column):
        for value in ["yes", "no", "N/A", "", 1, 0]:
            df = valid_processed_df.copy()
            df[column] = value
            with pytest.raises(SchemaError):
                processed_data_schema.validate(df)


class TestProcessedSchemaNumericRanges:

    @pytest.mark.parametrize("tenure", [1, 36, 72])
    def test_tenure_valid_values(self, valid_processed_df, tenure):
        df = valid_processed_df.copy()
        df["Tenure Months"] = tenure
        processed_data_schema.validate(df)

    @pytest.mark.parametrize("tenure", [0, -1, 73, None, "um"])
    def test_tenure_invalid_values(self, valid_processed_df, tenure):
        df = valid_processed_df.copy()
        df["Tenure Months"] = tenure
        with pytest.raises(SchemaError):
            processed_data_schema.validate(df)

    @pytest.mark.parametrize("monthly", [0.01, 65.0, 119.0])
    def test_monthly_charges_valid_values(self, valid_processed_df, monthly):
        df = valid_processed_df.copy()
        df["Monthly Charges"] = monthly
        processed_data_schema.validate(df)

    @pytest.mark.parametrize("monthly", [0.0, 119.01, None, "um"])
    def test_monthly_charges_invalid_values(self, valid_processed_df, monthly):
        df = valid_processed_df.copy()
        df["Monthly Charges"] = monthly
        with pytest.raises(SchemaError):
            processed_data_schema.validate(df)

    @pytest.mark.parametrize("total", [0.0, 1000.0, 8650.0])
    def test_total_charges_valid_values(self, valid_processed_df, total):
        df = valid_processed_df.copy()
        df["Total Charges"] = total
        processed_data_schema.validate(df)

    @pytest.mark.parametrize("total", [-0.01, 8651.0, None, "um"])
    def test_total_charges_invalid_values(self, valid_processed_df, total):
        df = valid_processed_df.copy()
        df["Total Charges"] = total
        with pytest.raises(SchemaError):
            processed_data_schema.validate(df)


class TestProcessedSchemaContractPayment:

    @pytest.mark.parametrize("contract", ["Month-to-month", "One year", "Two year"])
    def test_contract_valid_values(self, valid_processed_df, contract):
        df = valid_processed_df.copy()
        df["Contract"] = contract
        processed_data_schema.validate(df)

    @pytest.mark.parametrize("contract", ["monthly", "annual", "", None, 1])
    def test_contract_invalid_values(self, valid_processed_df, contract):
        df = valid_processed_df.copy()
        df["Contract"] = contract
        with pytest.raises(SchemaError):
            processed_data_schema.validate(df)

    @pytest.mark.parametrize("method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ])
    def test_payment_method_valid_values(self, valid_processed_df, method):
        df = valid_processed_df.copy()
        df["Payment Method"] = method
        processed_data_schema.validate(df)

    @pytest.mark.parametrize("method", ["Pix", "Boleto", "", None, 1])
    def test_payment_method_invalid_values(self, valid_processed_df, method):
        df = valid_processed_df.copy()
        df["Payment Method"] = method
        with pytest.raises(SchemaError):
            processed_data_schema.validate(df)


# ════════════════════════════════════════════════════════════════════════════════
# processed_inference_schema (batch inference)
# ════════════════════════════════════════════════════════════════════════════════
class TestProcessedInferenceSchemaCreation:

    def test_valid_dataframe_passes(self, valid_processed_inference_df):
        result = processed_inference_schema.validate(valid_processed_inference_df)
        assert isinstance(result, pd.DataFrame)

    def test_churn_value_must_be_absent(self, valid_processed_inference_df):
        df = valid_processed_inference_df.copy()
        df["Churn Value"] = 0
        with pytest.raises((SchemaError, SchemaErrors)):
            processed_inference_schema.validate(df)

    def test_missing_column_raises(self, valid_processed_inference_df):
        df = valid_processed_inference_df.drop(columns=["Tenure Months"])
        with pytest.raises(SchemaError):
            processed_inference_schema.validate(df)

    def test_strict_numeric_constraints_tenure(self, valid_processed_inference_df):
        df = valid_processed_inference_df.copy()
        df["Tenure Months"] = 73  # excede limite estrito
        with pytest.raises(SchemaError):
            processed_inference_schema.validate(df)


# ════════════════════════════════════════════════════════════════════════════════
# engineered_data_schema (treinamento)
# ════════════════════════════════════════════════════════════════════════════════

class TestEngineeredSchemaCreation:

    def test_valid_dataframe_passes(self, valid_engineered_df):
        result = engineered_data_schema.validate(valid_engineered_df)
        assert isinstance(result, pd.DataFrame)

    def test_missing_engineered_column_raises(self, valid_engineered_df):
        df = valid_engineered_df.drop(columns=["services_count"])
        with pytest.raises(SchemaError):
            engineered_data_schema.validate(df)

    def test_churn_value_required(self, valid_engineered_df):
        """engineered_data_schema exige Churn Value (para treinamento)."""
        df = valid_engineered_df.drop(columns=["Churn Value"])
        with pytest.raises(SchemaError):
            engineered_data_schema.validate(df)


def test_services_count_valid_values(valid_engineered_df):
    for value in [0, 1, 2, 3, 4, 5, 6]:
        df = valid_engineered_df.copy()
        df["services_count"] = value
        engineered_data_schema.validate(df)


def test_services_count_invalid_values(valid_engineered_df):
    for value in [-1, 7, None, "um"]:
        df = valid_engineered_df.copy()
        df["services_count"] = value
        with pytest.raises(SchemaError):
            engineered_data_schema.validate(df)


def test_tenure_group_valid_values(valid_engineered_df):
    for value in ["new", "growing", "loyal"]:
        df = valid_engineered_df.copy()
        df["tenure_group"] = value
        engineered_data_schema.validate(df)


def test_tenure_group_invalid_values(valid_engineered_df):
    for value in ["New", "medium", "", None]:
        df = valid_engineered_df.copy()
        df["tenure_group"] = value
        with pytest.raises(SchemaError):
            engineered_data_schema.validate(df)


def test_contract_risk_score_valid_values(valid_engineered_df):
    for value in [0, 1, 3]:
        df = valid_engineered_df.copy()
        df["contract_risk_score"] = value
        engineered_data_schema.validate(df)


def test_contract_risk_score_invalid_values(valid_engineered_df):
    for value in [2, -1, 4, None]:
        df = valid_engineered_df.copy()
        df["contract_risk_score"] = value
        with pytest.raises(SchemaError):
            engineered_data_schema.validate(df)

# ════════════════════════════════════════════════════════════════════════════════
# engineered_inference_schema (batch inference)
# ════════════════════════════════════════════════════════════════════════════════

class TestEngineeredInferenceSchemaCreation:

    def test_valid_dataframe_passes(self, valid_inference_df):
        result = engineered_inference_schema.validate(valid_inference_df)
        assert isinstance(result, pd.DataFrame)

    def test_churn_value_must_be_absent(self, valid_inference_df):
        """engineered_inference_schema rejeita Churn Value (strict=True)."""
        df = valid_inference_df.copy()
        df["Churn Value"] = 0
        with pytest.raises((SchemaError, SchemaErrors)):
            engineered_inference_schema.validate(df)

    def test_missing_engineered_column_raises(self, valid_inference_df):
        df = valid_inference_df.drop(columns=["services_count"])
        with pytest.raises(SchemaError):
            engineered_inference_schema.validate(df)


# ════════════════════════════════════════════════════════════════════════════════
# SCHEMA_REGISTRY e get_schema()
# ════════════════════════════════════════════════════════════════════════════════

class TestSchemaRegistry:

    def test_registry_has_all_keys(self):
        assert set(SCHEMA_REGISTRY.keys()) == {
            "processed", "processed_inference", "engineered", "engineered_inference"
            }

    def test_get_schema_processed(self):
        assert get_schema("processed") is processed_data_schema

    def test_get_schema_processed_inference(self):
        assert get_schema("processed_inference") is processed_inference_schema

    def test_get_schema_engineered(self):
        assert get_schema("engineered") is engineered_data_schema

    def test_get_schema_engineered_inference(self):
        assert get_schema("engineered_inference") is engineered_inference_schema

    def test_get_schema_invalid_key_raises(self):
        with pytest.raises(KeyError, match= r"Schema '.*' não encontrado"):
            get_schema("invalid")
