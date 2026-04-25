"""
Schema de validação para entrada de dados na API de predição.

Este módulo define a estrutura esperada para requisições POST /predict,
garantindo que clientes enviem dados no formato correto e dentro de ranges válidos.

IMPORTANTE: Campos binários (partner, phone_service, etc.) aceitam
valores "Yes" e "No" como no dataset original do IBM Telco.
São automaticamente convertidos para 0/1 internamente.

Pydantic BaseModel
- Validação de tipos automatizada
- Documentação Swagger automática
- Serialização/desserialização JSON
- Validadores customizados para lógica complexa
"""

from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class PredictionInput(BaseModel):
    """
    Schema de validação para predição de churn de cliente.

    Representa os dados brutos do cliente que serão processados pelo pipeline
    e então enviados ao modelo MLP para inferência.

    Todos os campos são obrigatórios (sem defaults).
    Ranges e tipos são validados automaticamente.

    Example:
        >>> data = {
        ...     "senior_citizen": "No",
        ...     "partner": "Yes",
        ...     "dependents": "No",
        ...     "tenure_months": 24,
        ...     "monthly_charges": 65.5,
        ...     "total_charges": 1570.0,
        ...     "phone_service": "Yes",
        ...     "paperless_billing": "No",
        ...     "payment_method": "Electronic check",
        ...     "contract": "One year",
        ...     "internet_service_type": "Fiber optic",
        ...     "online_security": "Yes",
        ...     "online_backup": "No",
        ...     "device_protection": "No",
        ...     "tech_support": "Yes",
        ...     "streaming_tv": "No",
        ...     "streaming_movies": "Yes",
        ... }
        >>> prediction = PredictionInput(**data)
    """

    # ── Demográficos ──────────────────────────────────────────────────
    gender: Literal["Female", "Male"] = Field(
        ...,
        description="Gênero do cliente"
    )

    senior_citizen: Literal["Yes", "No"] = Field(
        ...,
        description="Se o cliente é sênior"
    )

    partner: Literal["Yes", "No"] = Field(
        ...,
        description="Se o cliente tem um parceiro"
    )

    dependents: Literal["Yes", "No"] = Field(
        ...,
        description="Se o cliente tem dependentes"
    )

    # ── Informações de Conta ──────────────────────────────────────────
    tenure_months: int = Field(
        ...,
        ge=1,
        le=72,
        description="Número de meses como cliente. Min: 1 mês. Max: 72 meses (6 anos). Valores > 72 indicam possível data drift.",
    )

    monthly_charges: float = Field(
        ...,
        gt=0,
        le=119,
        description="Custo mensal do serviço (USD).",
    )

    total_charges: float = Field(
        ...,
        ge=0,
        le= 8650,
        description="Custo total acumulado (USD).",
    )

    phone_service: Literal["Yes", "No"] = Field(
        ...,
        description="Se cliente tem serviço de telefone"
    )

    multiple_lines: Literal["No", "No phone service", "Yes"] = Field(
        ...,
        description="Se o cliente tem múltiplas linhas telefônicas"
    )

    paperless_billing: Literal["Yes", "No"] = Field(
        ...,
        description="Se cliente usa fatura eletrônica"
    )

    # ── Pagamento ─────────────────────────────────────────────────────
    payment_method: Literal[
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)'
    ] = Field(
        ...,
        description="Método de pagamento"
    )

    # ── Serviço de Internet ───────────────────────────────────────────
    internet_service_type: Literal["Fiber optic", "DSL", "No"] = Field(
        ...,
        description="Tipo de serviço de internet contratado",
    )

    # ── Contrato ──────────────────────────────────────────────────────
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ...,
        description="Tipo de contrato. Contracts mais longos indicam menor churn.",
    )

    # ── Serviços Opcionais (Yes/No/No internet service) ───────────────
    online_security: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Serviço de segurança online",
    )

    online_backup: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Serviço de backup online",
    )

    device_protection: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Serviço de proteção de dispositivo",
    )

    tech_support: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Serviço de suporte técnico",
    )

    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Serviço de streaming de TV",
    )

    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Serviço de streaming de filmes",
    )

    # ── Configuração Pydantic para documentação Swagger ────────────────
    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "senior_citizen": "No",
                "partner": "Yes",
                "dependents": "No",
                "tenure_months": 24,
                "monthly_charges": 65.5,
                "total_charges": 1570.0,
                "phone_service": "Yes",
                "multiple_lines": "No",
                "paperless_billing": "No",
                "payment_method": "Electronic check",
                "contract": "One year",
                "internet_service_type": "Fiber optic",
                "online_security": "Yes",
                "online_backup": "No",
                "device_protection": "No",
                "tech_support": "Yes",
                "streaming_tv": "No",
                "streaming_movies": "Yes",
            }
        }
    }

    # ── Validadores customizados ──────────────────────────────────────

    @field_validator("total_charges")
    @classmethod
    def validate_total_vs_monthly(cls, value: float, info: ValidationInfo) -> float:
        """
        Validação cruzada: total_charges não deve ser muito menor que
        monthly_charges * tenure_months.

        Racional: Evita inconsistência entre campos correlacionados.
        Permite ~10% de diferença (ajustes, promoções, etc).
        """
        monthly = info.data.get("monthly_charges")
        tenure = info.data.get("tenure_months")

        if monthly is not None and tenure is not None:
            expected_min = monthly * tenure * 0.9  # 10% de desconto permitido
            if value < expected_min and tenure > 1:  # Ignora clientes com apenas 1 mês
                raise ValueError(
                    f"Total charges (${value:.2f}) muito baixo comparado a "
                    f"monthly_charges (${monthly:.2f}) × tenure ({tenure} meses). "
                    f"Esperado mínimo: ${expected_min:.2f}."
                )
        return value

    @model_validator(mode = "after")
    def validate_internet_service_consistency(self):
        """
        Validação cruzada: Se internet_service_type='No', serviços de internet
        não podem estar marcados como 'Yes'.
        """
        if self.internet_service_type == "No":

            internet_service_fields = [
            "streaming_tv", "streaming_movies", "online_security",
            "online_backup", "device_protection", "tech_support"
        ]
            problematic = [feature for feature in internet_service_fields if getattr(self, feature) == 'Yes']

            if problematic:
                raise ValueError(
                    f"Cliente com internet_service_type='No' não pode ter "
                    f"serviços: {', '.join(problematic)}. "
                    f"Esses serviços requerem internet."
                )

        return self


    def to_dict(self) -> dict:
        """
        Converte para dicionário para compatibilidade com sklearn/PyTorch.

        Realiza duas transformações:
        1. Converte campos binários de "Yes"/"No" para 1/0
        2. Renomeia campos de snake_case para Title Case (compatível com preprocessor)

        Returns:
            dict: Dados prontos para o pipeline de processamento
        """
        data = self.model_dump()

        #  ── Transformação 1: Yes/No → 1/0 ──────────────────────────────
        binary_features = [
            'senior_citizen',
            'partner',
            'dependents',
            'phone_service',
            'paperless_billing'
        ]

        for feature in binary_features:
            if feature in data:
                data[feature] = 1 if data[feature] == "Yes" else 0

        # ── Transformação 2: snake_case → Title Case ───────────────────
        # Mapeamento de nomes do schema para nomes do preprocessor
        column_mapping = {
            'gender': 'Gender',
            'senior_citizen': 'Senior Citizen',
            'partner': 'Partner',
            'dependents': 'Dependents',
            'tenure_months': 'Tenure Months',
            'monthly_charges': 'Monthly Charges',
            'total_charges': 'Total Charges',
            'phone_service': 'Phone Service',
            'multiple_lines': 'Multiple Lines',
            'paperless_billing': 'Paperless Billing',
            'payment_method': 'Payment Method',
            'internet_service_type': 'Internet Service',
            'contract': 'Contract',
            'online_security': 'Online Security',
            'online_backup': 'Online Backup',
            'device_protection': 'Device Protection',
            'tech_support': 'Tech Support',
            'streaming_tv': 'Streaming TV',
            'streaming_movies': 'Streaming Movies',
        }

        # Renomear todas as colunas
        data_renamed = {column_mapping.get(key, key): value for key, value in data.items()}

        return data_renamed
