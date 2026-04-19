"""
Schemas Pandera para validação de DataFrames em training/inference batch.

Este módulo define a estrutura esperada de dados em diferentes etapas
do pipeline:
1. processed_data_schema: Após limpeza (loaders + cleaning)
2. engineered_data_schema: Após feature engineering

Schema Validation Pattern:
- Define contrato de dados para múltiplas etapas
- Detecta problemas em nível de DataFrame (dimensões, NaN, valores inválidos)
- Reutilizável em training, testes e batch inference

Diferença de input.py / output.py:
- Pydantic valida 1 registro por vez (API)
- Pandera valida DataFrame inteiro (training/batch)
"""

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check


# ── SCHEMA 1: Dados Processados (Após cleaning) ────────────────────
# Aplicado após: loaders.load_data() + cleaning.clean_all()
# ANTES de: feature engineering

processed_data_schema = DataFrameSchema(
    {
        # ── Demográficos (já binarizados: 0/1) ─────────────────
        'Senior Citizen': Column(
            pa.Int,
            checks=[Check.isin([0, 1])],
            nullable=False,
            description="0=Não sênior, 1=Sênior (já binarizado)"
        ),
        
        'Partner': Column(
            pa.Int,
            checks=[Check.isin([0, 1])],
            nullable=False,
            description="0=Sem parceiro, 1=Com parceiro (já binarizado)"
        ),
        
        'Dependents': Column(
            pa.Int,
            checks=[Check.isin([0, 1])],
            nullable=False,
            description="0=Sem dependentes, 1=Com dependentes (já binarizado)"
        ),
        
        # ── Informações de Conta ───────────────────────────────
        'Tenure Months': Column(
            pa.Int,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(72),
            ],
            nullable=False,
            description="Meses como cliente (0-72)"
        ),
        
        'Monthly Charges': Column(
            pa.Float,
            checks=[
                Check.greater_than(0),
                Check.less_than_or_equal_to(119),
            ],
            nullable=False,
            description="Custo mensal em USD (> 0 e <= 119)"
        ),
        
        'Total Charges': Column(
            pa.Float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(8650)
            ],
            nullable=False,
            description="Custo total acumulado em USD"
        ),
        
        'Phone Service': Column(
            pa.Int,
            checks=[Check.isin([0, 1])],
            nullable=False,
            description="0=Sem serviço, 1=Com serviço (já binarizado)"
        ),
        
        'Paperless Billing': Column(
            pa.Int,
            checks=[Check.isin([0, 1])],
            nullable=False,
            description="0=Com papel, 1=Eletrônico (já binarizado)"
        ),
        
        # ── Serviços (Yes/No/No internet service) ────────────────
        # ✅ AGORA com Check.isin() correto!
        'Internet Service Type': Column(
            pa.String,
            checks=[Check.isin(['Fiber optic', 'DSL', 'No'])],
            nullable=False,
            description="Tipo de internet (validado)"
        ),
        
        'Online Security': Column(
            pa.String,
            checks=[Check.isin(['Yes', 'No', 'No internet service'])],
            nullable=False,
            description="Serviço de segurança online (validado)"
        ),
        
        'Online Backup': Column(
            pa.String,
            checks=[Check.isin(['Yes', 'No', 'No internet service'])],
            nullable=False,
            description="Serviço de backup online (validado)"
        ),
        
        'Device Protection': Column(
            pa.String,
            checks=[Check.isin(['Yes', 'No', 'No internet service'])],
            nullable=False,
            description="Proteção de dispositivo (validado)"
        ),
        
        'Tech Support': Column(
            pa.String,
            checks=[Check.isin(['Yes', 'No', 'No internet service'])],
            nullable=False,
            description="Suporte técnico (validado)"
        ),
        
        'Streaming TV': Column(
            pa.String,
            checks=[Check.isin(['Yes', 'No', 'No internet service'])],
            nullable=False,
            description="Streaming de TV (validado)"
        ),
        
        'Streaming Movies': Column(
            pa.String,
            checks=[Check.isin(['Yes', 'No', 'No internet service'])],
            nullable=False,
            description="Streaming de filmes (validado)"
        ),
        
        # ── Contrato e Pagamento ───────────────────────────────
        'Contract': Column(
            pa.String,
            checks=[Check.isin(['Month-to-month', 'One year', 'Two year'])],
            nullable=False,
            description="Tipo de contrato (validado)"
        ),
        
        'Payment Method': Column(
            pa.String,
            checks=[Check.isin([
                'Electronic check',
                'Mailed check',
                'Bank transfer (automatic)',
                'Credit card (automatic)'
            ])],
            nullable=False,
            description="Método de pagamento (validado)"
        ),
        
        # ── Alvo ────────────────────────────────────────────────
        'Churn Value': Column(
            pa.Int,
            checks=[Check.isin([0, 1])],
            nullable=False,
            description="0=Sem churn, 1=Churn"
        ),
    },
    strict=True,  # Rejeita colunas extras
    coerce=False,  # Não converte tipos automaticamente
    description="Schema para dados após limpeza (antes de feature engineering)"
)


# ── SCHEMA 2: Dados Engineered (Feature Engineering) ────────────────
# Aplicado após: feature engineering
# PRONTO para: treinamento do modelo

engineered_data_schema = DataFrameSchema(
    {
        # ── Features originais (processadas) ─────────────────────
        'Senior Citizen': Column(
            pa.Int,
            checks=[Check.isin([0, 1])],
            nullable=False
        ),
        'Partner': Column(
            pa.Int,
            checks=[Check.isin([0, 1])],
            nullable=False
        ),
        'Dependents': Column(
            pa.Int,
            checks=[Check.isin([0, 1])],
            nullable=False
        ),
        'Phone Service': Column(
            pa.Int,
            checks=[Check.isin([0, 1])],
            nullable=False
        ),
        'Paperless Billing': Column(
            pa.Int,
            checks=[Check.isin([0, 1])],
            nullable=False
        ),
        
        'Tenure Months': Column(
            pa.Int,
            checks=[Check.greater_than_or_equal_to(0)],
            nullable=False
        ),
        'Monthly Charges': Column(
            pa.Float,
            checks=[Check.greater_than(0)],
            nullable=False
        ),
        'Total Charges': Column(
            pa.Float,
            checks=[Check.greater_than_or_equal_to(0)],
            nullable=False
        ),
        
        # ── Features categóricas originais (com validação!) ─────
        'Contract': Column(
            pa.String,
            checks=[Check.isin(['Month-to-month', 'One year', 'Two year'])],
            nullable=False
        ),
        'Internet Service Type': Column(
            pa.String,
            checks=[Check.isin(['Fiber optic', 'DSL', 'No'])],
            nullable=False
        ),
        'Online Security': Column(
            pa.String,
            checks=[Check.isin(['Yes', 'No', 'No internet service'])],
            nullable=False
        ),
        'Online Backup': Column(
            pa.String,
            checks=[Check.isin(['Yes', 'No', 'No internet service'])],
            nullable=False
        ),
        'Device Protection': Column(
            pa.String,
            checks=[Check.isin(['Yes', 'No', 'No internet service'])],
            nullable=False
        ),
        'Tech Support': Column(
            pa.String,
            checks=[Check.isin(['Yes', 'No', 'No internet service'])],
            nullable=False
        ),
        'Streaming TV': Column(
            pa.String,
            checks=[Check.isin(['Yes', 'No', 'No internet service'])],
            nullable=False
        ),
        'Streaming Movies': Column(
            pa.String,
            checks=[Check.isin(['Yes', 'No', 'No internet service'])],
            nullable=False
        ),
        'Payment Method': Column(
            pa.String,
            checks=[Check.isin([
                'Electronic check',
                'Mailed check',
                'Bank transfer (automatic)',
                'Credit card (automatic)'
            ])],
            nullable=False
        ),
        
        # ── NOVAS FEATURES criadas no feature engineering ──────
        'services_count': Column(
            pa.Int,
            checks=[Check.isin([0, 1, 2, 3, 4, 5, 6])],
            nullable=False,
            description="Contagem de serviços opcionais (0-6)"
        ),
        
        'tenure_group': Column(
            pa.String,
            checks=[Check.isin(['new', 'growing', 'loyal'])],
            nullable=False,
            description="Faixas de tenure: new(0-12), growing(12-36), loyal(36+)"
        ),
        
        'monthly_per_tenure': Column(
            pa.Float,
            checks=[Check.greater_than_or_equal_to(0)],
            nullable=False,
            description="Custo mensal normalizado por tenure"
        ),
        
        'has_protection': Column(
            pa.Int,
            checks=[Check.isin([0, 1])],
            nullable=False,
            description="1 se cliente tem proteção (security OR device)"
        ),
        
        'is_senior_alone': Column(
            pa.Int,
            checks=[Check.isin([0, 1])],
            nullable=False,
            description="1 se idoso sem parceiro nem dependentes"
        ),
        
        'contract_risk_score': Column(
            pa.Int,
            checks=[Check.isin([0, 1, 3])],
            nullable=False,
            description="0=Two year, 1=One year, 3=Month-to-month"
        ),
        
        # ── Alvo ────────────────────────────────────────────────
        'Churn Value': Column(
            pa.Int,
            checks=[Check.isin([0, 1])],
            nullable=False
        ),
    },
    strict=True,
    coerce=False,
    description="Schema para dados após feature engineering (pronto para treinamento)"
)


# ── REGISTRY: Centralize todas as schemas ──────────────────────────
SCHEMA_REGISTRY = {
    'processed': processed_data_schema,      # Após limpeza
    'engineered': engineered_data_schema,    # Após features
}


def get_schema(name: str) -> DataFrameSchema:
    """
    Recupera schema pelo nome do registry.
    
    Args:
        name: 'processed' ou 'engineered'
        
    Returns:
        DataFrameSchema: Schema solicitado
        
    Raises:
        KeyError: Se schema não existe
    """
    if name not in SCHEMA_REGISTRY:
        raise KeyError(
            f"Schema '{name}' não encontrado. "
            f"Disponíveis: {list(SCHEMA_REGISTRY.keys())}"
        )
    return SCHEMA_REGISTRY[name]