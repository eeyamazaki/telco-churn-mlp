"""Script de pré-processamento dos dados brutos do Telco Customer Churn.

Execução:
    python -m src.data.preprocess
"""

from src.config import DATA_PROCESSED_DIR
from src.data.cleaning import clean
from src.data.loaders import load_raw_data
from src.logger import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """Carrega o arquivo raw, aplica limpeza e salva o CSV processado."""
    setup_logging()

    logger.info("preprocess started")

    df_raw = load_raw_data()
    df_clean = clean(df_raw)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED_DIR / "telco_churn_cleaned.csv"
    df_clean.to_csv(output_path, index=False)

    logger.info("preprocess finished", output=str(output_path), shape=df_clean.shape)


if __name__ == "__main__":
    main()
