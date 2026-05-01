"""Transformador de feature engineering para o dataset Telco Churn.

Encapsula as 6 features derivadas criadas durante a análise exploratória,
empacotadas como um transformador compatível com scikit-learn para uso em pipelines.

Uso típico:
    from src.features.engineer import FeatureEngineer

    engineer = FeatureEngineer()
    df_engineered = engineer.fit_transform(df_clean)
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import OPTIONAL_SERVICES_COLS
from src.logger import get_logger

logger = get_logger(__name__)

# Mapeamento ordinal de risco por tipo de contrato (EDA: 43% → 11% → 3% churn)
_CONTRACT_RISK_MAP = {"Month-to-month": 3, "One year": 1, "Two year": 0}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Cria 6 features derivadas a partir das colunas brutas do Telco Churn.

    Todas as features são derivadas de padrões identificados no notebook de EDA.
    Este transformador não possui estado aprendido (fit não faz nada) e é seguro
    para uso em validação cruzada e pipelines de inferência.

    Colunas de entrada obrigatórias
    --------------------------------
    Online Security, Online Backup, Device Protection, Tech Support,
    Streaming TV, Streaming Movies, Tenure Months, Monthly Charges,
    Senior Citizen, Partner, Dependents, Contract

    Saída
    -----
    DataFrame original acrescido de 6 novas colunas:
    - services_count       : int, quantidade de serviços opcionais contratados (0–6)
    - tenure_group         : str, grupo de maturidade do contrato (new/growing/loyal)
    - monthly_per_tenure   : float, mensalidade por mês de tenure
    - has_protection       : int (0/1), indica se algum serviço de proteção está ativo
    - is_senior_alone      : int (0/1), idoso sem parceiro e sem dependentes
    - contract_risk_score  : int, score ordinal de risco por tipo de contrato (0/1/3)
    """

    def fit(self, x: pd.DataFrame, y=None) -> "FeatureEngineer":
        """Sem estado aprendido — fit não faz nada, retorna self."""
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Aplica o feature engineering ao DataFrame de entrada.

        Parâmetros
        ----------
        X : pd.DataFrame
            Dados brutos com as colunas obrigatórias.

        Retorno
        -------
        pd.DataFrame
            Cópia de X com 6 colunas adicionais de features engenheiradas.
        """

        logger.info("feature engineering started", rows=x.shape[0], cols=x.shape[1])

        df = x.copy()
        df = self._add_services_count(df)
        df = self._add_tenure_group(df)
        df = self._add_monthly_per_tenure(df)
        df = self._add_has_protection(df)
        df = self._add_is_senior_alone(df)
        df = self._add_contract_risk_score(df)

        logger.info("feature engineering finished", new_cols=df.shape[1] - x.shape[1])

        return df

    def _add_services_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Conta serviços opcionais contratados (0-6).

        Hipótese: mais serviços → maior lock-in → menor churn.
        """
        df["services_count"] = df[OPTIONAL_SERVICES_COLS].apply(
            lambda row: (row == "Yes").sum(), axis=1
        )
        return df

    def _add_tenure_group(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrupa clientes por maturidade do contrato (new/growing/loyal).

        Hipótese: distribuição bimodal no EDA — clientes novos e veteranos
        têm comportamentos de churn distintos.
        """
        df["tenure_group"] = pd.cut(
            df["Tenure Months"],
            bins=[-1, 12, 36, 72],
            labels=["new", "growing", "loyal"],
        ).astype(str)
        return df

    def _add_monthly_per_tenure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula custo mensal normalizado por mês de permanência.

        O +1 evita divisão por zero para clientes com Tenure Months == 0.
        Hipótese: cliente novo pagando muito ainda não justificou o custo percebido.
        """
        df["monthly_per_tenure"] = df["Monthly Charges"] / (df["Tenure Months"] + 1)
        return df

    def _add_has_protection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag: cliente tem Online Security OU Device Protection ativo (0/1).

        Hipótese: adoção de proteção indica engajamento mínimo com a operadora.
        """
        df["has_protection"] = (
            (df["Online Security"] == "Yes") | (df["Device Protection"] == "Yes")
        ).astype(int)
        return df

    def _add_is_senior_alone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag: idoso sem parceiro e sem dependentes (0/1).

        Senior Citizen, Partner e Dependents já foram convertidos para 0/1 no ETL.
        Hipótese: EDA mostrou 42% de churn em idosos — isolamento amplifica o risco.
        """
        df["is_senior_alone"] = (
            (df["Senior Citizen"] == 1) & (df["Dependents"] == 0) & (df["Partner"] == 0)
        ).astype(int)
        return df

    def _add_contract_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score ordinal de risco pelo tipo de contrato (0/1/3).

        Preserva a gradação natural: mensal (43% churn) > anual > bienal (3% churn).
        """
        df["contract_risk_score"] = df["Contract"].map(_CONTRACT_RISK_MAP)
        return df
