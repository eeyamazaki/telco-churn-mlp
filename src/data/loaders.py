"""Funções de carregamento de dados para o projeto Telco Churn.

Fornece duas funções principais:
    - load_data: carregamento genérico a partir de qualquer caminho CSV ou XLSX.
    - load_raw_data: atalho para carregar o arquivo bruto definido em config.py.
    - load_from_upload: carregamento a partir de bytes de upload HTTP (CSV ou XLSX).

Uso típico:
    from src.data.loaders import load_raw_data
    df = load_raw_data()

    from src.data.loaders import load_data
    df = load_data('data/processed/telco_cleaned.csv')
"""

import io
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
        logger.error("file not found", path=str(path))
        raise FileNotFoundError(f"Arquivo não encontrado: {path.name}")

    # Validação: Extensão é válida?
    if path.suffix.lower() not in _VALID_EXTENSIONS:
        logger.error(
            "unsupported extension", extension=path.suffix, valid=_VALID_EXTENSIONS
        )
        raise ValueError(
            f"Extensão '{path.suffix}' não suporta\n"
            f"Extensões válidas {_VALID_EXTENSIONS}"
        )

    # Carregamento
    df = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)

    # Logging
    logger.info("data loaded", file=path.name, rows=df.shape[0], cols=df.shape[1])

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
    logger.debug("loading raw data", path=str(path))

    return load_data(path)


def load_from_upload(content: bytes, filename: str) -> pd.DataFrame:
    """Carrega dados de bytes de um upload HTTP (Excel ou CSV) em memória.

    Usado pelo endpoint /predict/batch da API para evitar escrita em disco.
    A detecção do formato é feita pela extensão do nome do arquivo.
    Extensões suportadas: .csv, .xlsx

    Args:
        content: Conteúdo bruto do arquivo em bytes (lido do UploadFile).
        filename: Nome original do arquivo, usado para detectar a extensão.

    Raises:
        ValueError: Se a extensão do arquivo não for suportada.

    Returns:
        pd.DataFrame: DataFrame com os dados carregados.

    Example:
        >>> content = file.file.read()
        >>> df = load_from_upload(content, file.filename)
    """

    suffix = Path(filename).suffix.lower()

    if suffix not in _VALID_EXTENSIONS:
        logger.error("unsupported extension", extension=suffix, valid=_VALID_EXTENSIONS)
        raise ValueError(
            f"Extensão '{suffix}' não suportada.\n"
            f"Extensões válidas: {_VALID_EXTENSIONS}"
        )

    buf = io.BytesIO(content)
    df = pd.read_excel(buf) if suffix == ".xlsx" else pd.read_csv(buf)

    logger.info(
        "data loaded from upload", file=filename, rows=df.shape[0], cols=df.shape[1]
    )

    return df
