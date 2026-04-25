"""Pipeline de inferência para previsão de churn.

Carrega os artefatos treinados (preprocessor, pesos MLP, config) uma única vez
na inicialização e expõe a função predict() utilizada pelo endpoint FastAPI.

Ordem do pipeline (espelha o notebook 04_mlp.ipynb):
    DataFrame bruto
        → FeatureEngineer  (adiciona 6 features derivadas)
        → ColumnTransformer (StandardScaler + OneHotEncoder)
        → ChurnMLP (PyTorch, 49 features → logits)
        → probabilidade de churn + predição binária
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
import pandera.pandas as pa
import torch

from src.config import MODELS_DIR
from src.features import FeatureEngineer
from src.logger import get_logger
from src.models import ChurnMLP
from src.schemas.common import processed_inference_schema

logger = get_logger(__name__)

_config = json.loads((MODELS_DIR / "mlp_config.json").read_text())

# Threshold F1-ótimo derivado da curva Precision-Recall no test set (seção 4 do notebook 04).
# Para maximizar recall (minimizar FN), use o threshold de negócio da seção 9.
DEFAULT_THRESHOLD: float = _config["threshold"]


class ChurnPredictor:
    """Carrega os artefatos treinados e executa o pipeline completo de inferência.

    Parâmetros
    ----------
    threshold : float
        Threshold de decisão para a predição binária.
        Padrão: valor F1-ótimo carregado de models/mlp_config.json.

    Exemplos
    --------
    >>> predictor = ChurnPredictor()
    >>> result = predictor.predict(customer_df)
    >>> result["churn_probability"], result["churn_prediction"]
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self._feature_engineer = FeatureEngineer()

        logger.info("loading model artifacts", models_dir=str(MODELS_DIR))

        config = json.loads((MODELS_DIR / "mlp_config.json").read_text())
        self._preprocessor = joblib.load(MODELS_DIR / "preprocessor_mlp.pkl")

        self._model = ChurnMLP(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            dropout_rate=config["dropout_rate"],
        )
        self._model.load_state_dict(
            torch.load(MODELS_DIR / "mlp_weights.pt", weights_only=True)
        )
        self._model.eval()

        logger.info(
            "artifacts loaded",
            model="ChurnMLP",
            hidden_dims=config["hidden_dims"],
            threshold=self.threshold,
        )

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """Executa o pipeline completo de inferência sobre os dados de entrada.

        Parâmetros
        ----------
        x : pd.DataFrame
            Dados brutos do cliente. Deve conter as mesmas colunas presentes em
            data/processed/telco_churn_cleaned.csv (sem a coluna 'Churn Value').

        Retorno
        -------
        pd.DataFrame
            Uma linha por cliente com as colunas:
            - churn_probability : float, score do modelo em [0, 1]
            - churn_prediction  : int (0 ou 1), decisão binária no threshold
        """
        # Etapa 0 — validação do schema de entrada (detecta colunas faltando, tipos errados
        # e valores fora dos domínios esperados antes de qualquer processamento)
        try:
            processed_inference_schema.validate(x)
        except pa.errors.SchemaError as exc:
            logger.error("schema validation failed", error=str(exc))
            raise

        # Etapa 1 — feature engineering
        x_eng = self._feature_engineer.transform(x)

        # Etapa 2 — encoding e scaling
        x_enc = self._preprocessor.transform(x_eng)

        # Etapa 3 — forward pass PyTorch (model.eval() garante BatchNorm com running stats)
        x_tensor = torch.tensor(x_enc, dtype=torch.float32)
        with torch.no_grad():
            proba = torch.sigmoid(self._model(x_tensor)).numpy().flatten()

        prediction = (proba >= self.threshold).astype(int)

        logger.info(
            "prediction complete",
            n_samples=len(x),
            n_churn_predicted=int(prediction.sum()),
        )

        return pd.DataFrame(
            {"churn_probability": proba, "churn_prediction": prediction},
            index=x.index,
        )

    def predict_single(self, customer: dict) -> dict:
        """Atalho para inferência de um único cliente (utilizado pela API).

        Parâmetros
        ----------
        customer : dict
            Registro de um único cliente como dicionário plano.

        Retorno
        -------
        dict
            Chaves: churn_probability (float), churn_prediction (int).
        """
        df = pd.DataFrame([customer])
        result = self.predict(df)
        return {
            "churn_probability": round(float(result["churn_probability"].iloc[0]), 4),
            "churn_prediction": int(result["churn_prediction"].iloc[0]),
        }
