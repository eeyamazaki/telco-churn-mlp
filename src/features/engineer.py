"""Transformador de feature engineering para o dataset Telco Churn.

Encapsula as 6 features derivadas criadas durante a análise exploratória,
empacotadas como um transformador compatível com scikit-learn para uso em pipelines.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Serviços opcionais usados para calcular services_count
_OPTIONAL_SERVICES = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]

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

    def fit(self, x, y=None):
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
        df = x.copy()

        # 1. Contagem de serviços opcionais contratados (0–6)
        # Hipótese: mais serviços → maior lock-in → menor churn
        df["services_count"] = df[_OPTIONAL_SERVICES].apply(
            lambda row: (row == "Yes").sum(), axis=1
        )

        # 2. Grupo de maturidade do contrato
        # Distribuição bimodal no EDA: clientes novos e veteranos têm comportamentos distintos
        df["tenure_group"] = pd.cut(
            df["Tenure Months"],
            bins=[-1, 12, 36, 72],
            labels=["new", "growing", "loyal"],
        ).astype(str)

        # 3. Custo mensal por mês de tenure (+1 evita divisão por zero para tenure=0)
        # Hipótese: cliente novo pagando muito ainda não justificou o custo percebido
        df["monthly_per_tenure"] = df["Monthly Charges"] / (df["Tenure Months"] + 1)

        # 4. Flag de serviço de proteção ativo (Online Security OU Device Protection)
        # Hipótese: adoção de proteção indica engajamento mínimo com a operadora
        df["has_protection"] = (
            (df["Online Security"] == "Yes") | (df["Device Protection"] == "Yes")
        ).astype(int)

        # 5. Idoso sem rede de suporte (sem parceiro e sem dependentes)
        # EDA: Senior Citizens têm 42% de churn; isolamento pode amplificar esse risco
        # Senior Citizen, Partner e Dependents já foram convertidos para 0/1 no ETL
        df["is_senior_alone"] = (
            (df["Senior Citizen"] == 1)
            & (df["Partner"] == 0)
            & (df["Dependents"] == 0)
        ).astype(int)

        # 6. Score ordinal de risco por tipo de contrato
        # Preserva a gradação natural: mensal (43% churn) > anual > bienal (3% churn)
        df["contract_risk_score"] = df["Contract"].map(_CONTRACT_RISK_MAP)

        return df
