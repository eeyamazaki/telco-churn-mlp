"""Funções de carregamento de dados para o projeto Telco Churn.

Fornece duas funções principais:
    - load_data: carregamento genérico a partir de qualquer caminho CSV ou XLSX.
    - load_raw_data: atalho para carregar o arquivo bruto definido em config.py.

Uso típico:
    from src.data.loaders import load_raw_data
    df = load_raw_data()

    from src.data.loaders import load_data
    df = load_data('data/processed/telco_cleaned.csv')
"""

from pathlib import Path

import pandas as pd

from src.config import DATA_RAW_DIR, RAW_DATA_FILE
from src.logger import get_logger

logger = get_logger(__name__)

_VALID_EXTENSIONS = [".csv", ".xlsx"]


def load_data(path: Path | str) -> pd.DataFrame:
    """Carrega dados de arquivos CSV ou XLSX com validação.
    Extensões suportadas: .csv, .xlsx

    Args:
        path (Path | str): Caminho do arquivo a ser carregado

    Raises:
        FileNotFoundError: Se arquivo não existir no caminho
        ValueError: Se a extensão do arquivo não for suportada.

    Returns:
        pd.DataFrame: DataFrame com os dados carregados
        
    Example:
        >>> df = load_data('data/raw/telco.xlsx')
        >>> df = load_data(Path('data/processed/clean.csv'))
    """

    path = Path(path)

    # Validação: Arquivo existe?
    if not path.exists():
        logger.error("file not found", path = str(path))
        raise FileNotFoundError(f"Arquivo não encontrado: {path.name}")

    # Validação: Extensão é válida?
    if path.suffix.lower() not in _VALID_EXTENSIONS:
        logger.error("unsupported extension", extension = path.suffix, valid = _VALID_EXTENSIONS)
        raise ValueError(
            f"Extensão '{path.suffix}' não suporta\n"
            f"Extensões válidas {_VALID_EXTENSIONS}"
        )

    # Carregamento
    df = pd.read_excel(path) if path.suffix.lower() == '.xlsx' else pd.read_csv(path)

    # Logging
    logger.info("data loaded", file= path.name, rows = df.shape[0], cols = df.shape[1])

    return df

def load_raw_data() -> pd.DataFrame:
    """Carrega o arquivo de dados brutos.

    Conveniência para não precisar informar o caminho em cada chamada.
    O arquivo e diretório são controlados pelas constantes
    RAW_DATA_FILE e DATA_RAW_DIR em src/config.py.

    Raises:
        FileNotFoundError: Se o arquivo bruto não existir.

    Returns:
        DataFrame com os dados brutos carregados.

    Exemplo:
        >>> df = load_raw_data()
    """

    path = DATA_RAW_DIR / RAW_DATA_FILE
    logger.debug("loading raw data", path = str(path))

    return load_data(path)
