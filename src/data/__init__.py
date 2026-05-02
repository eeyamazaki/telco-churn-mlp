"""Pacote de dados — carregamento e limpeza do dataset Telco Customer Churn.

Expõe a API pública do pacote:
    - load_data: carregamento genérico (CSV ou XLSX)
    - load_raw_data: atalho para o arquivo bruto definido em config.py
    - clean: pipeline de limpeza completo para treinamento
    - load_from_upload: carregamento a partir de bytes de upload HTTP (CSV ou XLSX)

Uso típico:
    from src.data import load_raw_data, clean

    df = clean(load_raw_data())
"""

from src.data.cleaning import clean, clean_for_inference
from src.data.loaders import load_data, load_from_upload, load_raw_data

__all__ = [
    "clean",
    "load_data",
    "load_raw_data",
    "load_from_upload",
    "clean_for_inference",
]
