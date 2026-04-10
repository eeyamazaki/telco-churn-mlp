"""Centralized project configuration.

All paths, hyperparameters, and constants live here so every module
pulls from the same source of truth.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# ── Random seed ───────────────────────────────────────
RANDOM_SEED = 42

# ── Data ──────────────────────────────────────────────
TARGET_COLUMN = "Churn"
RAW_DATA_FILE = "Telco_customer_churn.xlsx"

# ── Training defaults ─────────────────────────────────
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# ── MLflow ────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "telco-churn"
MLFLOW_TRACKING_URI = "mlruns"
