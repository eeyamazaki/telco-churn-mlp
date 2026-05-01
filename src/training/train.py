"""Script de treinamento do MLP para previsão de churn.

Reproduz fielmente o pipeline do notebook 04_mlp.ipynb:
    1. Carrega dados processados e aplica FeatureEngineer
    2. Split estratificado 70 / 15 / 15
    3. Encoda com ColumnTransformer (StandardScaler + OneHotEncoder)
    4. Treina ChurnMLP com pos_weight e early stopping
    5. Encontra threshold F1-ótimo na curva Precision-Recall
    6. Registra parâmetros, métricas e curvas de loss no MLflow
    7. Salva artefatos em models/ (preprocessor, pesos, config)

Execução:
    python -m src.training.train
"""

import json

import joblib
import mlflow
import mlflow.data
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    DATA_PROCESSED_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from src.features import FeatureEngineer, build_preprocessor
from src.logger import get_logger, setup_logging
from src.models import ChurnMLP
from src.schemas.common import processed_data_schema

logger = get_logger(__name__)

# ── Hiperparâmetros ───────────────────────────────────────────────────────────
HIDDEN_DIMS: list[int] = [64, 32]
DROPOUT_RATE: float = 0.3
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
MAX_EPOCHS: int = 150
PATIENCE: int = 15
LR_SCHEDULER_PATIENCE: int = 8
LR_SCHEDULER_FACTOR: float = 0.5

# Splits estratificados: 70% treino / 15% validação / 15% teste
# O conjunto de validação é necessário para early stopping no MLP —
# diferente dos modelos sklearn que usam cross-validation.
TEST_SIZE: float = 0.15
VAL_FRACTION_OF_TEMP: float = 0.176  # ≈ 15% do total após remover o test set

DATA_FILE: str = "telco_churn_cleaned.csv"


# ── Funções utilitárias ───────────────────────────────────────────────────────


def _make_tensor_dataset(x_enc: np.ndarray, y: pd.Series) -> TensorDataset:
    """Converte arrays numpy em TensorDataset para o DataLoader."""
    x_t = torch.tensor(x_enc, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    return TensorDataset(x_t, y_t)


def _find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Encontra o threshold que maximiza o F1-Score na curva Precision-Recall.

    O pos_weight no BCEWithLogitsLoss desloca a distribuição de probabilidades,
    tornando o threshold padrão (0.5) subótimo para datasets desbalanceados.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    return float(thresholds[np.argmax(f1_scores[:-1])])


# ── Loop de treinamento ───────────────────────────────────────────────────────


def train_model(
    model: ChurnMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: pd.Series,
    device: torch.device,
) -> dict[str, list[float]]:
    """Treina o modelo com early stopping e ReduceLROnPlateau.

    Parâmetros
    ----------
    model       : instância ChurnMLP já alocada no device
    train_loader: DataLoader do conjunto de treino
    val_loader  : DataLoader do conjunto de validação
    y_train     : labels de treino para calcular o pos_weight
    device      : dispositivo de execução (cpu / cuda)

    Retorno
    -------
    dict com listas 'train_loss' e 'val_loss' por época
    """
    # pos_weight corrige o desbalanceamento de classes (26% churn)
    # dando peso proporcional à razão negativos/positivos na loss
    neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=LR_SCHEDULER_PATIENCE, factor=LR_SCHEDULER_FACTOR
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights: dict | None = None
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        # ── fase de treino ──
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── fase de validação ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                val_loss += criterion(
                    model(x_batch.to(device)), y_batch.to(device)
                ).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            # clone profundo para não compartilhar referências com o optimizer
            best_weights = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                "training epoch",
                epoch=epoch + 1,
                train_loss=round(train_loss, 4),
                val_loss=round(val_loss, 4),
            )

        if no_improve >= PATIENCE:
            logger.info(
                "early stopping", epoch=epoch + 1, best_val_loss=round(best_val_loss, 4)
            )
            break

    model.load_state_dict(best_weights)
    return history


# ── Avaliação ────────────────────────────────────────────────────────────────


def evaluate(
    model: ChurnMLP,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (y_true, y_proba) para o conjunto dado."""
    model.eval()
    proba_list, true_list = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            proba = torch.sigmoid(model(x_batch.to(device))).cpu().numpy()
            proba_list.append(proba)
            true_list.append(y_batch.numpy())
    return np.vstack(true_list).flatten(), np.vstack(proba_list).flatten()


# ── Pipeline principal ────────────────────────────────────────────────────────


def main() -> None:
    """Executa o pipeline completo de treinamento e registro de artefatos."""

    setup_logging()

    # Garante reprodutibilidade em todas as bibliotecas
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("training started", device=str(device), random_seed=RANDOM_SEED)

    # ── 1. Dados ──────────────────────────────────────────────────────────────
    data_path = DATA_PROCESSED_DIR / DATA_FILE
    df = pd.read_csv(data_path)
    logger.info("data loaded", path=str(data_path), shape=df.shape)

    # Validação de anomalias e schema
    try:
        processed_data_schema.validate(df)
        logger.info("schema validation passed")
    except Exception as exc:
        logger.warning(
            "schema validation failed - proceeding with caution", error=str(exc)
        )

    feature_engineer = FeatureEngineer()
    x = feature_engineer.fit_transform(df.drop(columns=[TARGET_COLUMN]))
    y = df[TARGET_COLUMN]
    logger.info(
        "feature engineering done", n_features=x.shape[1], churn_rate=round(y.mean(), 4)
    )

    # ── 2. Split estratificado 70/15/15 ──────────────────────────────────────
    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=VAL_FRACTION_OF_TEMP,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )
    logger.info(
        "data split",
        train=len(x_train),
        val=len(x_val),
        test=len(x_test),
    )

    # ── 3. Encoding (fit apenas no treino — evita data leakage) ──────────────
    preprocessor = build_preprocessor()

    x_train_enc = preprocessor.fit_transform(x_train)
    x_val_enc = preprocessor.transform(x_val)
    x_test_enc = preprocessor.transform(x_test)

    input_dim = x_train_enc.shape[1]
    logger.info("encoding done", input_dim=input_dim)

    # ── 4. DataLoaders ────────────────────────────────────────────────────────
    train_loader = DataLoader(
        _make_tensor_dataset(x_train_enc, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        _make_tensor_dataset(x_val_enc, y_val), batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(
        _make_tensor_dataset(x_test_enc, y_test), batch_size=BATCH_SIZE
    )

    # ── 5. Modelo e treinamento ───────────────────────────────────────────────
    model = ChurnMLP(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        dropout_rate=DROPOUT_RATE,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model created", hidden_dims=HIDDEN_DIMS, n_params=n_params)

    history = train_model(model, train_loader, val_loader, y_train, device)

    # ── 6. Avaliação no conjunto de teste ─────────────────────────────────────
    y_true, y_proba = evaluate(model, test_loader, device)
    best_threshold = _find_optimal_threshold(y_true, y_proba)
    y_pred = (y_proba >= best_threshold).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc": round(average_precision_score(y_true, y_proba), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
    }
    logger.info("evaluation complete", threshold=round(best_threshold, 4), **metrics)

    # ── 7. MLflow ─────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df_eng = x.copy()
    df_eng[TARGET_COLUMN] = y
    mlflow_dataset = mlflow.data.from_pandas(
        df_eng,
        source=str(data_path),
        name="telco_churn_processed",
        targets=TARGET_COLUMN,
    )

    with mlflow.start_run(run_name=f"mlp-{'-'.join(str(h) for h in HIDDEN_DIMS)}"):
        mlflow.log_input(mlflow_dataset, context="training")
        mlflow.log_params(
            {
                "hidden_dims": HIDDEN_DIMS,
                "dropout_rate": DROPOUT_RATE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "threshold": round(best_threshold, 4),
                "random_seed": RANDOM_SEED,
            }
        )
        mlflow.log_metrics(metrics)

        # curvas de loss por época
        for step, (tl, vl) in enumerate(
            zip(history["train_loss"], history["val_loss"], strict=True)
        ):
            mlflow.log_metric("train_loss", tl, step=step)
            mlflow.log_metric("val_loss", vl, step=step)

        mlflow.pytorch.log_model(model, name="model")

    logger.info("mlflow run logged", experiment=MLFLOW_EXPERIMENT_NAME)

    # ── 8. Artefatos em disco ─────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, MODELS_DIR / "preprocessor_mlp.pkl")
    torch.save(model.state_dict(), MODELS_DIR / "mlp_weights.pt")

    config = {
        "input_dim": int(input_dim),
        "hidden_dims": HIDDEN_DIMS,
        "dropout_rate": DROPOUT_RATE,
        "threshold": round(best_threshold, 4),
    }
    (MODELS_DIR / "mlp_config.json").write_text(json.dumps(config, indent=2))

    logger.info(
        "artifacts saved",
        models_dir=str(MODELS_DIR),
        threshold=config["threshold"],
    )


if __name__ == "__main__":
    main()
