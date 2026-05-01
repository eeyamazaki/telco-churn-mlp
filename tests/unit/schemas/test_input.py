"""
Testes para src/schemas/input.py (PredictionInput - validação de entrada da API).
"""

import pytest
from pydantic import ValidationError

from src.schemas.input import PredictionInput

# ════════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def valid_input_data() -> dict:
    """Dados válidos e realistas para criar PredictionInput."""
    return {
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


# ════════════════════════════════════════════════════════════════════════════════
# CRIAÇÃO E CAMPOS VÁLIDOS
# ════════════════════════════════════════════════════════════════════════════════


class TestPredictionInputCreation:
    """Testes para criação bem-sucedida de PredictionInput."""

    def test_create_with_valid_data(self, valid_input_data):
        """Criar PredictionInput com dados válidos."""
        prediction = PredictionInput(**valid_input_data)

        assert prediction is not None
        assert prediction.senior_citizen == "No"
        assert prediction.tenure_months == 24


# ════════════════════════════════════════════════════════════════════════════════
# VALIDAÇÃO DE CAMPOS INDIVIDUAIS
# ════════════════════════════════════════════════════════════════════════════════


class TestTenureValidation:
    """Testes para campo 'tenure_months' (0-72)."""

    @pytest.mark.parametrize(
        "valid_tenure, total_for_tenure",
        [
            (1, 65.5),  # 1 mês (mínimo válido)
            (12, 786.0),  # 65.5 × 12 × 1.0
            (24, 1570.0),  # 65.5 × 24 × 1.0 (fixture original)
            (60, 3537.0),  # 65.5 × 60 × 0.9 (com 10% desconto)
            (72, 4245.0),  # 65.5 × 72 × 0.9
        ],
    )
    def test_tenure_valid_values(
        self, valid_input_data, valid_tenure, total_for_tenure
    ):
        """Tenure dentro do intervalo válido (0 a 120)."""
        valid_input_data["tenure_months"] = valid_tenure
        valid_input_data["total_charges"] = total_for_tenure
        prediction = PredictionInput(**valid_input_data)
        assert prediction.tenure_months == valid_tenure

    @pytest.mark.parametrize(
        "invalid_value", [0, -1, 73, 121, 150, "doze", "abc", [1, 2], None]
    )
    def test_tenure_invalid_values(self, valid_input_data, invalid_value):
        """Tenure fora do intervalo ou tipo inválido deve lançar ValidationError."""
        valid_input_data["tenure_months"] = invalid_value

        with pytest.raises(ValidationError):
            PredictionInput(**valid_input_data)


class TestMonthlyChargesValidation:
    """Testes para campo 'monthly_charges' (> 0 e <= 119)."""

    @pytest.mark.parametrize(
        "valid_monthly, total_for_monthly",
        [
            (0.01, 0.24),  # 0.01 × 24
            (50.0, 1200.0),  # 50.0 × 24
            (99.99, 2399.76),  # 99.99 × 24
            (119.0, 2856.0),  # 119.0 × 24
        ],
    )
    def test_monthly_charges_valid_values(
        self, valid_input_data, valid_monthly, total_for_monthly
    ):
        """Monthly charges dentro do intervalo válido (> 0 e <= 119)."""
        valid_input_data["monthly_charges"] = valid_monthly
        valid_input_data["total_charges"] = total_for_monthly
        prediction = PredictionInput(**valid_input_data)
        assert prediction.monthly_charges == valid_monthly

    @pytest.mark.parametrize(
        "invalid_value", [0.0, -1.0, 119.01, 200.0, "cinquenta", "abc", [1, 2], None]
    )
    def test_monthly_charges_invalid_values(self, valid_input_data, invalid_value):
        """Monthly charges fora do intervalo ou tipo inválido deve lançar ValidationError."""
        valid_input_data["monthly_charges"] = invalid_value

        with pytest.raises(ValidationError):
            PredictionInput(**valid_input_data)


class TestTotalChargesValidation:
    """Testes para campo 'total_charges' (>= 0)."""

    @pytest.mark.parametrize(
        "tenure, monthly, valid_total",
        [
            (1, 65.5, 65.5),  # 1 mês (mínimo): total = monthly ✓
            (2, 50.0, 100.0),  # 50.0 × 2 = 100.0 ✓
            (24, 65.5, 5000.0),  # bem acima do mínimo ✓
        ],
    )
    def test_total_charges_valid_values(
        self, valid_input_data, tenure, monthly, valid_total
    ):
        """Total charges dentro do intervalo válido (>= 0)."""
        valid_input_data["tenure_months"] = tenure
        valid_input_data["monthly_charges"] = monthly
        valid_input_data["total_charges"] = valid_total
        prediction = PredictionInput(**valid_input_data)
        assert prediction.total_charges == valid_total

    @pytest.mark.parametrize("invalid_value", ["mil", "abc", -1, [1, 2], None])
    def test_total_charges_invalid_type(self, valid_input_data, invalid_value):
        """Tipo inválido para total_charges deve lançar ValidationError."""
        valid_input_data["total_charges"] = invalid_value

        with pytest.raises(ValidationError):
            PredictionInput(**valid_input_data)


# ════════════════════════════════════════════════════════════════════════════════
# VALIDAÇÃO DE CAMPOS CATEGÓRICOS
# ════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    "binary_field",
    ["senior_citizen", "partner", "dependents", "phone_service", "paperless_billing"],
)
class TestBinaryFieldsValidation:
    """Testes para campos binários."""

    @pytest.mark.parametrize("valid_binary", ["Yes", "No"])
    def test_senior_citizen_valid_values(
        self, valid_input_data, binary_field, valid_binary
    ):
        """Senior citizen com valores válidos ('Yes' ou 'No')."""
        valid_input_data[binary_field] = valid_binary
        prediction = PredictionInput(**valid_input_data)
        assert getattr(prediction, binary_field) == valid_binary

    @pytest.mark.parametrize("invalid_value", ["Maybe", "1", "0", "true", "false"])
    def test_senior_citizen_invalid_values(
        self, valid_input_data, binary_field, invalid_value
    ):
        """Senior citizen com valores inválidos deve lançar ValidationError."""
        valid_input_data[binary_field] = invalid_value

        with pytest.raises(ValidationError):
            PredictionInput(**valid_input_data)


class TestContractValidation:
    """Testes para campo 'contract'."""

    @pytest.mark.parametrize(
        "valid_contract", ["Month-to-month", "One year", "Two year"]
    )
    def test_contract_valid_values(self, valid_input_data, valid_contract):
        """Testar todos os valores válidos de contrato."""
        valid_input_data["contract"] = valid_contract
        prediction = PredictionInput(**valid_input_data)
        assert prediction.contract == valid_contract

    @pytest.mark.parametrize(
        "invalid_value", ["Six months", "Month", "One Month", "Three years", "", 1]
    )
    def test_contract_invalid_values(self, valid_input_data, invalid_value):
        """Valores inválidos de contrato devem lançar ValidationError."""
        valid_input_data["contract"] = invalid_value

        with pytest.raises(ValidationError):
            PredictionInput(**valid_input_data)


class TestPaymentMethodValidation:
    """Testes para campo 'payment_method'."""

    @pytest.mark.parametrize(
        "valid_payment",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    def test_payment_method_valid_values(self, valid_input_data, valid_payment):
        """Testar todos os métodos de pagamento válidos."""
        valid_input_data["payment_method"] = valid_payment
        prediction = PredictionInput(**valid_input_data)
        assert prediction.payment_method == valid_payment

    @pytest.mark.parametrize(
        "invalid_value", ["Cash", "Crypto", "Gift Card", "PayPal", "", 1]
    )
    def test_payment_method_invalid_values(self, valid_input_data, invalid_value):
        """Métodos de pagamento inválidos devem lançar ValidationError."""
        valid_input_data["payment_method"] = invalid_value

        with pytest.raises(ValidationError):
            PredictionInput(**valid_input_data)


class TestInternetServiceValidation:
    """Testes para campo 'internet_service_type'."""

    @pytest.mark.parametrize("valid_internet", ["Fiber optic", "DSL", "No"])
    def test_internet_service_valid_values(self, valid_input_data, valid_internet):
        """Testar todos os tipos de internet válidos."""
        valid_input_data["internet_service_type"] = valid_internet
        if valid_internet == "No":
            internet_service_fields = [
                "streaming_tv",
                "streaming_movies",
                "online_security",
                "online_backup",
                "device_protection",
                "tech_support",
            ]
            for field in internet_service_fields:
                valid_input_data[field] = "No internet service"

        prediction = PredictionInput(**valid_input_data)
        assert prediction.internet_service_type == valid_internet

    @pytest.mark.parametrize(
        "invalid_value", ["Cable", "Satellite", "5G", "Ethernet", "", 1]
    )
    def test_internet_service_invalid_values(self, valid_input_data, invalid_value):
        """Tipos de internet inválidos devem lançar ValidationError."""
        valid_input_data["internet_service_type"] = invalid_value

        with pytest.raises(ValidationError):
            PredictionInput(**valid_input_data)


# ════════════════════════════════════════════════════════════════════════════════
# VALIDAÇÕES CRUZADAS (@field_validator)
# ════════════════════════════════════════════════════════════════════════════════


class TestCrossFieldValidations:
    """Testes para validadores envolvendo múltiplos campos."""

    @pytest.mark.parametrize(
        "monthly, tenure, total",
        [
            (100.0, 12, 1200.0),
            (100.0, 12, 1080.0),
            (65.5, 24, 1570.0),
        ],
    )
    def test_total_vs_monthly_charges_valid(
        self, valid_input_data, monthly, tenure, total
    ):
        """Total charges consistente com monthly_charges * tenure."""
        valid_input_data["monthly_charges"] = monthly
        valid_input_data["tenure_months"] = tenure
        valid_input_data["total_charges"] = total

        prediction = PredictionInput(**valid_input_data)
        assert prediction.total_charges == total

    @pytest.mark.parametrize(
        "monthly,tenure,total",
        [
            (100.0, 12, 500.0),
            (100.0, 12, 800.0),
            (50.0, 24, 800.0),
        ],
    )
    def test_total_charges_inconsistent(self, valid_input_data, monthly, tenure, total):
        """Total charges inconsistente devido a regra de 10% max de desconto deve lançar ValidationError."""
        valid_input_data["monthly_charges"] = monthly
        valid_input_data["tenure_months"] = tenure
        valid_input_data["total_charges"] = total

        with pytest.raises(ValidationError):
            PredictionInput(**valid_input_data)

    # def test_total_charges_ignore_validation_for_new_customer(self, valid_input_data):
    #     """Clientes novos (tenure=0) ignoram validação de total_charges."""
    #     valid_input_data["tenure_months"] = 0
    #     valid_input_data["total_charges"] = 0.0

    # prediction = PredictionInput(**valid_input_data)
    # assert prediction.total_charges == 0.0

    @pytest.mark.parametrize(
        "service_field",
        [
            "streaming_tv",
            "streaming_movies",
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
        ],
    )
    def test_internet_service_inconsistency(self, valid_input_data, service_field):
        """Sem internet mas com serviço que requer internet deve lançar erro."""
        valid_input_data["internet_service_type"] = "No"
        valid_input_data[service_field] = "Yes"

        with pytest.raises(ValidationError):
            PredictionInput(**valid_input_data)


# ════════════════════════════════════════════════════════════════════════════════
# TESTES DE CONVERSÃO to_dict()
# ════════════════════════════════════════════════════════════════════════════════


class TestToDictConversion:
    """Testes para método to_dict() (conversão e renaming)."""

    def test_to_dict_with_valid_data(self, valid_input_data):
        """to_dict() retorna dicionário com chaves em Title Case."""
        prediction = PredictionInput(**valid_input_data)
        result_dict = prediction.to_dict()

        assert isinstance(result_dict, dict)
        assert "Senior Citizen" in result_dict
        assert "Partner" in result_dict
        assert "Tenure Months" in result_dict

    @pytest.mark.parametrize(
        "binary_field, input_value, expected_output_key, expected_value",
        [
            ("senior_citizen", "Yes", "Senior Citizen", 1),
            ("senior_citizen", "No", "Senior Citizen", 0),
            ("partner", "Yes", "Partner", 1),
            ("partner", "No", "Partner", 0),
            ("dependents", "Yes", "Dependents", 1),
            ("dependents", "No", "Dependents", 0),
            ("phone_service", "Yes", "Phone Service", 1),
            ("phone_service", "No", "Phone Service", 0),
            ("paperless_billing", "Yes", "Paperless Billing", 1),
            ("paperless_billing", "No", "Paperless Billing", 0),
        ],
    )
    def test_to_dict_binary_conversions(
        self,
        valid_input_data,
        binary_field,
        input_value,
        expected_output_key,
        expected_value,
    ):
        """Campos binários são convertidos de Yes/No para 1/0 com Title Case."""
        valid_input_data[binary_field] = input_value
        prediction = PredictionInput(**valid_input_data)
        result_dict = prediction.to_dict()

        assert result_dict[expected_output_key] == expected_value

    def test_to_dict_no_snake_case_in_output(self, valid_input_data):
        """Nenhuma chave em snake_case no output (todas em Title Case)."""
        prediction = PredictionInput(**valid_input_data)
        result_dict = prediction.to_dict()

        snake_case_keys = [
            "gender",
            "senior_citizen",
            "phone_service",
            "multiple_lines",
            "paperless_billing",
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "internet_service_type",
            "payment_method",
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
        ]

        for snake_key in snake_case_keys:
            assert snake_key not in result_dict

    def test_to_dict_is_compatible_with_sklearn(self, valid_input_data):
        """Output compatível com sklearn (Title Case, tipos int/float/str)."""
        prediction = PredictionInput(**valid_input_data)
        result_dict = prediction.to_dict()

        expected_fields = {
            "Gender": str,
            "Senior Citizen": int,
            "Tenure Months": int,
            "Monthly Charges": float,
            "Total Charges": float,
            "Contract": str,
            "Internet Service": str,
            "Multiple Lines": str,
            "Payment Method": str,
        }

        for field_name, expected_type in expected_fields.items():
            assert field_name in result_dict
            assert isinstance(result_dict[field_name], expected_type)


# ════════════════════════════════════════════════════════════════════════════════
# TRATAMENTO DE ERROS
# ════════════════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Testes para falhas de validação esperadas."""

    @pytest.mark.parametrize(
        "missing_field",
        [
            "contract",
            "tenure_months",
            "monthly_charges",
            "internet_service_type",
            "senior_citizen",
        ],
    )
    def test_missing_required_fields(self, valid_input_data, missing_field):
        """Campos obrigatórios faltando devem lançar ValidationError."""
        del valid_input_data[missing_field]

        with pytest.raises(ValidationError):
            PredictionInput(**valid_input_data)

    def test_extra_field_ignored(self, valid_input_data):
        """Campo extra desconhecido é ignorado."""
        valid_input_data["extra_unknown_field"] = "something"

        prediction = PredictionInput(**valid_input_data)
        result_dict = prediction.to_dict()

        assert prediction is not None
        assert "extra_unknown_field" not in result_dict


# ════════════════════════════════════════════════════════════════════════════════
# FLUXO COMPLETO
# ════════════════════════════════════════════════════════════════════════════════


class TestCompleteWorkflow:
    """Testes de fluxo fim a fim."""

    def test_complete_api_flow(self, valid_input_data):
        """Fluxo completo da API: receber → validar → converter."""
        prediction = PredictionInput(**valid_input_data)
        assert prediction is not None

        pipeline_data = prediction.to_dict()
        assert isinstance(pipeline_data, dict)
        assert len(pipeline_data) == 19
        assert all(
            isinstance(value, (int, float, str)) for value in pipeline_data.values()
        )

    def test_round_trip_conversion(self, valid_input_data):
        """Round trip: original → PredictionInput → to_dict()."""
        original_partner = valid_input_data["partner"]

        prediction = PredictionInput(**valid_input_data)
        result = prediction.to_dict()

        expected_value = 1 if original_partner == "Yes" else 0
        assert result["Partner"] == expected_value
