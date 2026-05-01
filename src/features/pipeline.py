"""Preprocessador sklearn para o dataset Telco Churn.

Encapsula a construção do ColumnTransformer que normaliza features numéricas
e codifica features categóricas, pronto para uso em pipelines de treino e inferência.

Uso típico:
    from src.features.pipeline import build_preprocessor

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.logger import get_logger

logger = get_logger(__name__)


def build_preprocessor() -> ColumnTransformer:
    """Cria o preprocessador sklearn com escalamento numérico e encoding categórico.

    Retorna um ColumnTransformer não fitado. O fit deve ocorrer apenas com
    dados de treino para evitar data leakage.

    Retorno
    -------
    ColumnTransformer
        Preprocessador com dois transformadores:
        - StandardScaler  → NUMERIC_FEATURES
        - OneHotEncoder   → CATEGORICAL_FEATURES
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    logger.debug(
        "preprocessor built",
        numeric_cols=len(NUMERIC_FEATURES),
        categorical_cols=len(CATEGORICAL_FEATURES),
    )

    return preprocessor
