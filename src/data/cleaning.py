"""Limpeza dos dados brutos do dataset Telco Customer Churn.

Responsabilidades:
    - Corrigir tipos de dados (Total Charges: str → float)
    - Remover registros inválidos (nulos após conversão, duplicatas)
    - Remover colunas sem valor preditivo (identificadores, sem variância, data leakage)
    - Encodar variáveis binárias (Yes/No → 1/0)

Uso típico:
    from src.data.loaders import load_raw_data
    from src.data.cleaning import clean

    df = load_raw_data()
    df_clean = clean(df)
"""

import pandas as pd

from src.config import BINARY_COLS, COLS_TO_DROP
from src.logger import get_logger

logger = get_logger(__name__)


def convert_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Converte coluna 'Total Charges' de string para float.
    Strings vazias são convertidas para NaN via errors='coerce'.
    Essa função trabalha com dados brutos onde Total Charges vem como string.

    Args:
        df (pd.DataFrame): DataFrame contendo a coluna 'Total Charges'

    Returns:
        pd.DataFrame: DataFrame com 'Total Charges' convertido para float64.

    Raises:
        ValueError: Se a coluna 'Total Charges' não existir no DataFrame.
    """

    if "Total Charges" not in df.columns:
        logger.error(
            "missing required column",
            column="Total Charges",
            available=df.columns.to_list(),
        )
        raise ValueError(
            f"Coluna 'Total Charges' não encontrada.\n"
            f"Colunas disponíveis: {df.columns.to_list()}"
        )

    df = df.copy()
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    n_nulls = df["Total Charges"].isna().sum()

    logger.info("total_charges converted", nulls_generated=int(n_nulls))

    return df


def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Remove registros com 'Total Charges' nulo.

    Esses registros correspondem a clientes com Tenure Months == 0,
    que ainda não geraram nenhuma cobrança e não possuem sinal preditivo.
    Deve ser chamada após convert_total_charges().

    Args:
        df: DataFrame após a conversão de 'Total Charges'.

    Returns:
        DataFrame sem linhas com 'Total Charges' nulo, com índice reiniciado.
    """

    before = len(df)
    df = df.dropna(subset=["Total Charges"]).reset_index(drop=True)
    dropped = before - len(df)

    logger.info("null rows dropped", rows_dropped=dropped, rows_remaining=len(df))

    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas sem valor preditivo para o modelo de churn.

    Categorias de remoção:
    - Sem variância: Count, Country, State (valor único no dataset)
    - Identificadores: CustomerID, Lat Long
    - Geolocalização: Latitude, Longitude, Zip Code, City
    - Data leakage: Churn Score, CLTV, Churn Reason
    - Redundante: Churn Label (duplicata de Churn Value)

    Colunas ausentes no DataFrame são ignoradas silenciosamente.

    Args:
        df: DataFrame após remoção de nulos da coluna 'Total Charges'.

    Returns:
        DataFrame com colunas desnecessárias removidas (33 → 20).
    """

    existing = [col for col in COLS_TO_DROP if col in df.columns]
    df = df.drop(columns=existing)

    logger.info("columns dropped", count=len(existing), shape=df.shape)

    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas com perfil de features idêntico.

    Deve ser chamada após drop_columns() para que identificadores
    (CustomerID) já tenham sido removidos. Mantém a primeira ocorrência.

    Args:
        df: DataFrame após remoção de colunas desnecessárias.

    Returns:
        DataFrame sem duplicatas, com índice reiniciado.
    """

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = before - len(df)

    logger.info("duplicate rows dropped", rows_dropped=dropped, rows_remaining=len(df))

    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Converte variáveis binárias Yes/No para inteiros 1/0.

    Variáveis convertidas: Senior Citizen, Partner, Dependents,
    Phone Service, Paperless Billing.

    Senior Citizen já é armazenado como int no dataset original;
    o replace não o altera, mas o astype garante consistência de tipo.

    Args:
        df: DataFrame após remoção de colunas e duplicatas.

    Returns:
        DataFrame com variáveis binárias como int64.
    """

    existing = [col for col in BINARY_COLS if col in df.columns]
    df = df.copy()

    df[existing] = (
        df[existing]
        .apply(lambda col: col.map({"Yes": 1, "No": 0}) if col.dtype == object else col)
        .astype(int)
    )

    logger.info("binary columns encoded", columns=existing)

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de limpeza completo dos dados brutos do Telco dataset.

    Executa as etapas em sequência:
    1. Converte 'Total Charges' de string para float
    2. Remove registros com 'Total Charges' nulo (Tenure Months == 0)
    3. Remove colunas sem valor preditivo
    4. Remove linhas com perfil de features duplicado
    5. Encoda variáveis binárias Yes/No → 1/0

    Args:
        df: DataFrame bruto retornado por load_raw_data().

    Returns:
        DataFrame limpo, pronto para o pipeline de features.
        Esperado: ~7.010 linhas × 20 colunas.
    """

    logger.info("cleaning started", rows=df.shape[0], cols=df.shape[1])
    df = convert_total_charges(df)
    df = drop_nulls(df)
    df = drop_columns(df)
    df = drop_duplicates(df)
    df = encode_binary_columns(df)
    logger.info("cleaning finished", rows=df.shape[0], cols=df.shape[1])
    return df


def clean_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara dados brutos para inferência, preservando todas as colunas e linhas válidas.

    Diferente de clean(), não remove colunas nem deduplica — apenas corrige tipos
    e descarta clientes com tenure=0 (Total Charges nulo), que o modelo não consegue prever.
    """
    logger.info("cleaning started", rows=df.shape[0], cols=df.shape[1])
    df = convert_total_charges(df)
    df = drop_nulls(df)
    df = encode_binary_columns(df)
    logger.info("cleaning finished", rows=df.shape[0], cols=df.shape[1])

    return df
