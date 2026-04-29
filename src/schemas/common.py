"""
Schemas Pandera para validação de DataFrames em training/inference batch.
...
"""

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

# ════════════════════════════════════════════════════════════════════════════════
# BLOCOS DE COLUNAS REUTILIZÁVEIS (privados)
# ════════════════════════════════════════════════════════════════════════════════

_CATEGORICAL_COLS = {
    "Gender": Column(
        pa.String,
        checks=[Check.isin(["Female", "Male"])],
        nullable=False,
        description="Gênero do cliente",
    ),
    "Multiple Lines": Column(
        pa.String,
        checks=[Check.isin(["No", "No phone service", "Yes"])],
        nullable=False,
        description="Múltiplas linhas telefônicas",
    ),
}

_BINARY_COLS = {
    "Senior Citizen": Column(
        pa.Int,
        checks=[Check.isin([0, 1])],
        nullable=False,
        description="0=Não sênior, 1=Sênior (já binarizado)",
    ),
    "Partner": Column(
        pa.Int,
        checks=[Check.isin([0, 1])],
        nullable=False,
        description="0=Sem parceiro, 1=Com parceiro (já binarizado)",
    ),
    "Dependents": Column(
        pa.Int,
        checks=[Check.isin([0, 1])],
        nullable=False,
        description="0=Sem dependentes, 1=Com dependentes (já binarizado)",
    ),
    "Phone Service": Column(
        pa.Int,
        checks=[Check.isin([0, 1])],
        nullable=False,
        description="0=Sem serviço, 1=Com serviço (já binarizado)",
    ),
    "Paperless Billing": Column(
        pa.Int,
        checks=[Check.isin([0, 1])],
        nullable=False,
        description="0=Com papel, 1=Eletrônico (já binarizado)",
    ),
}

# Constraints ESTRITAS — usadas em processed (dados brutos de entrada)
_NUMERIC_COLS_STRICT = {
    "Tenure Months": Column(
        pa.Int,
        checks=[
            Check.greater_than_or_equal_to(1),
            Check.less_than_or_equal_to(72),
        ],
        nullable=False,
        description="Meses como cliente (1-72). Valor > 72 indica possível data drift.",
    ),
    "Monthly Charges": Column(
        pa.Float,
        checks=[
            Check.greater_than(0),
            Check.less_than_or_equal_to(119),
        ],
        nullable=False,
        description="Custo mensal em USD (> 0 e <= 119)",
    ),
    "Total Charges": Column(
        pa.Float,
        checks=[
            Check.greater_than_or_equal_to(0),
            Check.less_than_or_equal_to(8650),
        ],
        nullable=False,
        description="Custo total acumulado em USD",
    ),
}

_INTERNET_SERVICE_COLS = {
    "Internet Service": Column(
        pa.String,
        checks=[Check.isin(["Fiber optic", "DSL", "No"])],
        nullable=False,
        description="Tipo de internet (validado)",
    ),
    "Online Security": Column(
        pa.String,
        checks=[Check.isin(["Yes", "No", "No internet service"])],
        nullable=False,
    ),
    "Online Backup": Column(
        pa.String,
        checks=[Check.isin(["Yes", "No", "No internet service"])],
        nullable=False,
    ),
    "Device Protection": Column(
        pa.String,
        checks=[Check.isin(["Yes", "No", "No internet service"])],
        nullable=False,
    ),
    "Tech Support": Column(
        pa.String,
        checks=[Check.isin(["Yes", "No", "No internet service"])],
        nullable=False,
    ),
    "Streaming TV": Column(
        pa.String,
        checks=[Check.isin(["Yes", "No", "No internet service"])],
        nullable=False,
    ),
    "Streaming Movies": Column(
        pa.String,
        checks=[Check.isin(["Yes", "No", "No internet service"])],
        nullable=False,
    ),
}

_CONTRACT_PAYMENT_COLS = {
    "Contract": Column(
        pa.String,
        checks=[Check.isin(["Month-to-month", "One year", "Two year"])],
        nullable=False,
    ),
    "Payment Method": Column(
        pa.String,
        checks=[
            Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            )
        ],
        nullable=False,
    ),
}

_CHURN_VALUE_COL = {
    "Churn Value": Column(
        pa.Int,
        checks=[Check.isin([0, 1])],
        nullable=False,
        description="0=Sem churn, 1=Churn",
    ),
}


# ════════════════════════════════════════════════════════════════════════════════
# SCHEMAS PÚBLICOS
# ════════════════════════════════════════════════════════════════════════════════

# SCHEMA 1: Dados processados — treinamento (com alvo, constraints estritas)
processed_data_schema = DataFrameSchema(
    {
        **_CATEGORICAL_COLS,
        **_BINARY_COLS,
        **_NUMERIC_COLS_STRICT,
        **_INTERNET_SERVICE_COLS,
        **_CONTRACT_PAYMENT_COLS,
        **_CHURN_VALUE_COL,
    },
    strict=True,
    coerce=False,
    description="Schema para dados após limpeza — treinamento (com Churn Value)",
)

# SCHEMA 2: Dados processados — inferência (sem alvo, constraints estritas)
processed_inference_schema = DataFrameSchema(
    {
        **_CATEGORICAL_COLS,
        **_BINARY_COLS,
        **_NUMERIC_COLS_STRICT,
        **_INTERNET_SERVICE_COLS,
        **_CONTRACT_PAYMENT_COLS,
    },
    strict=True,
    coerce=False,
    description="Schema para dados após limpeza — inferência (sem Churn Value)",
)

# ════════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ════════════════════════════════════════════════════════════════════════════════

SCHEMA_REGISTRY = {
    "processed": processed_data_schema,
    "processed_inference": processed_inference_schema,
}


def get_schema(name: str) -> DataFrameSchema:
    """
    Recupera schema pelo nome do registry.

    Args:
        name: Chave do schema. Valores válidos:
              'processed'           — dados após limpeza (treinamento)
              'processed_inference' — dados após limpeza (inferência)

    Returns:
        DataFrameSchema: Schema solicitado.

    Raises:
        KeyError: Se o nome não existir no SCHEMA_REGISTRY.
    """

    if name not in SCHEMA_REGISTRY:
        raise KeyError(
            f"Schema '{name}' não encontrado. "
            f"Disponíveis: {list(SCHEMA_REGISTRY.keys())}"
        )
    return SCHEMA_REGISTRY[name]
