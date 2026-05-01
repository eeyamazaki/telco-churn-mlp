"""Pacote de features — engenharia e pré-processamento do dataset Telco Customer Churn.

Expõe a API pública do pacote:
    - FeatureEngineer: transformador sklearn com 6 features derivadas do EDA
    - build_preprocessor: factory que cria o ColumnTransformer (scaler + encoder)

Uso típico:
    from src.features import FeatureEngineer, build_preprocessor

    engineer = FeatureEngineer()
    df_engineered = engineer.fit_transform(df_clean)

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(df_engineered)
"""

from src.features.engineer import FeatureEngineer
from src.features.pipeline import build_preprocessor

__all__ = ["FeatureEngineer", "build_preprocessor"]
