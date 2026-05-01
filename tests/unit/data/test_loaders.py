"""Testes para src/data/loaders.py."""

import io
from pathlib import Path

import pandas as pd
import pytest

from src.data.loaders import load_data, load_from_upload

# ════════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    """Cria um arquivo CSV temporário válido para testes."""
    file = tmp_path / "telco.csv"
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
    df.to_csv(file, index=False)
    return file


@pytest.fixture
def xlsx_file(tmp_path: Path) -> Path:
    """Cria um arquivo XLSX temporário válido para testes."""
    file = tmp_path / "telco.xlsx"
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
    df.to_excel(file, index=False)
    return file


# ════════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO BEM-SUCEDIDO
# ════════════════════════════════════════════════════════════════════════════════


class TestLoadDataSuccess:
    """Testes de carregamento bem-sucedido."""

    def test_load_csv(self, csv_file):
        """Deve carregar um CSV e retornar DataFrame."""
        df = load_data(csv_file)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert list(df.columns) == ["col_a", "col_b"]
        assert df["col_a"].to_list() == [1, 2, 3]

    def test_load_xlsx(self, xlsx_file):
        """Deve carregar um XLSX e retornar DataFrame."""
        df = load_data(xlsx_file)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert list(df.columns) == ["col_a", "col_b"]
        assert df["col_a"].to_list() == [1, 2, 3]

    def test_accepts_string_path(self, csv_file):
        """Deve aceitar caminho como string além de Path."""
        df = load_data(str(csv_file))

        assert isinstance(df, pd.DataFrame)

    def test_extension_case_insensitive(self, tmp_path):
        """Deve aceitar extensões em maiúsculo (.CSV, .XLSX)."""
        file = tmp_path / "telco.CSV"
        pd.DataFrame({"a": [1]}).to_csv(file, index=False)

        df = load_data(file)

        assert isinstance(df, pd.DataFrame)


# ════════════════════════════════════════════════════════════════════════════════
# ERROS ESPERADOS
# ════════════════════════════════════════════════════════════════════════════════


class TestLoadDataErrors:
    """Testes de erros esperados."""

    def test_file_not_found(self, tmp_path):
        """Deve levantar FileNotFoundError se o arquivo não existir."""
        with pytest.raises(FileNotFoundError):
            load_data(tmp_path / "nao_existe.csv")

    def test_unsupported_extension(self, tmp_path):
        """Deve levantar ValueError para extensões não suportadas."""
        file = tmp_path / "dados.parquet"
        file.touch()

        with pytest.raises(ValueError):
            load_data(file)


# ════════════════════════════════════════════════════════════════════════════════
# load_from_upload
# ════════════════════════════════════════════════════════════════════════════════


class TestLoadFromUploadSuccess:
    """Testes de carregamento bem-sucedido via bytes de upload."""

    def test_load_csv_bytes(self):
        """Deve carregar CSV a partir de bytes e retornar DataFrame."""
        buf = io.StringIO()
        pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]}).to_csv(buf, index=False)
        content = buf.getvalue().encode()

        df = load_from_upload(content, "telco.csv")

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert df["col_a"].to_list() == [1, 2, 3]

    def test_load_xlsx_bytes(self):
        """Deve carregar XLSX a partir de bytes e retornar DataFrame."""
        buf = io.BytesIO()
        pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]}).to_excel(buf, index=False)
        content = buf.getvalue()

        df = load_from_upload(content, "telco.xlsx")

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert df["col_a"].to_list() == [1, 2, 3]

    def test_extension_case_insensitive(self):
        """Deve aceitar extensões em maiúsculo (.CSV)."""
        buf = io.StringIO()
        pd.DataFrame({"a": [1]}).to_csv(buf, index=False)
        content = buf.getvalue().encode()

        df = load_from_upload(content, "dados.CSV")

        assert isinstance(df, pd.DataFrame)


class TestLoadFromUploadErrors:
    """Testes de erros esperados para upload."""

    def test_unsupported_extension(self):
        """Deve levantar ValueError para extensões não suportadas."""
        with pytest.raises(ValueError):
            load_from_upload(b"data", "dados.parquet")
