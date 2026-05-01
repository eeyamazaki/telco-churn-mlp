"""Verificação de disponibilidade dos artefatos do modelo na inicialização."""

import joblib

from src.config import MODELS_DIR
from src.logger import get_logger

logger = get_logger(__name__)

_PREPROCESSOR_PATH = MODELS_DIR / "preprocessor_mlp.pkl"

try:
    preprocessor = joblib.load(_PREPROCESSOR_PATH)
    MODEL_LOADED = True
    logger.info("preprocessor carregado", path=str(_PREPROCESSOR_PATH))
except FileNotFoundError:
    preprocessor = None
    MODEL_LOADED = False
    logger.warning("preprocessor nao encontrado", path=str(_PREPROCESSOR_PATH))
except Exception as exc:
    preprocessor = None
    MODEL_LOADED = False
    logger.error("erro ao carregar preprocessor", error=str(exc))
