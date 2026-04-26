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
TARGET_COLUMN = "Churn Value"
RAW_DATA_FILE = "Telco_customer_churn.xlsx"

# ── Training defaults ─────────────────────────────────
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 15

# ── MLflow ────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "telco-churn-mlp"
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")

# ── Feature columns ───────────────────────────────────
COLS_TO_DROP = [
    # Sem variância — valor único em todo o dataset
    "Count",
    "Country",
    "State",
    # Identificadores — não carregam sinal preditivo
    "CustomerID",
    "Lat Long",
    # Geolocalização — granularidade excessiva para modelo tabular
    "Latitude",
    "Longitude",
    "Zip Code",
    "City",
    # Data leakage — derivados do churn já conhecido
    "Churn Score",
    "CLTV",
    "Churn Reason",
    # Redundante — duplicata categórica do target numérico
    "Churn Label",
]

BINARY_COLS = [
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Paperless Billing",
]

OPTIONAL_SERVICES_COLS = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]

NUMERIC_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Paperless Billing",
    "services_count",
    "monthly_per_tenure",
    "has_protection",
    "is_senior_alone",
    "contract_risk_score",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Payment Method",
    "tenure_group",
]
