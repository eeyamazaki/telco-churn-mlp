import joblib
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PREPROCESS_PATH = os.path.join(BASE_DIR, "models", "preprocessor_mlp.pkl")

try:
    preprocess = joblib.load(PREPROCESS_PATH)

    print(f"Preprocessor carregado: {PREPROCESS_PATH}")

    MODEL_LOADED = True

except FileNotFoundError:
    preprocess = None
    MODEL_LOADED = False

    print("Arquivo do preprocessor não encontrado.")

except Exception as e:
    preprocess = None
    MODEL_LOADED = False

    print(f"Erro ao carregar preprocessor: {e}")