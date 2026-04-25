"""
Testes unitários para src/models/mlp.py (ChurnMLP).

Verifica a arquitetura isoladamente — sem carregar artefatos de disco,
sem preprocessor, sem dados reais. Usa tensores sintéticos.
"""

import pytest
import torch

from src.models.mlp import ChurnMLP


# ════════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def model() -> ChurnMLP:
    """Arquitetura padrão do projeto: 49 → 64 → 32 → 1."""
    return ChurnMLP(input_dim=49, hidden_dims=[64, 32], dropout_rate=0.3)


@pytest.fixture
def batch() -> torch.Tensor:
    """Batch sintético de 16 amostras com 49 features."""
    torch.manual_seed(42)
    return torch.randn(16, 49)


# ════════════════════════════════════════════════════════════════════════════════
# SHAPE DE SAÍDA
# ════════════════════════════════════════════════════════════════════════════════

class TestOutputShape:
    """O forward() deve retornar logits com shape (batch_size, 1) para qualquer entrada."""

    def test_batch_shape(self, model, batch):
        """Batch de 16 amostras → shape (16, 1)."""
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (16, 1)

    def test_single_sample_shape(self, model):
        """Uma única amostra → shape (1, 1). Exige eval() para o BatchNorm não falhar."""
        model.eval()
        x = torch.randn(1, 49)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)

    def test_large_batch_shape(self, model):
        """Batch maior que o padrão (ex: 256) → shape (256, 1)."""
        model.eval()
        x = torch.randn(256, 49)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (256, 1)

    def test_configurable_input_dim(self):
        """input_dim configurável — modelo menor para verificar flexibilidade."""
        model = ChurnMLP(input_dim=10, hidden_dims=[8, 4], dropout_rate=0.0)
        model.eval()
        x = torch.randn(4, 10)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 1)


# ════════════════════════════════════════════════════════════════════════════════
# SAÍDA COMO LOGITS (sem Sigmoid)
# ════════════════════════════════════════════════════════════════════════════════

class TestLogits:
    """A camada de saída retorna logits — Sigmoid é aplicado apenas na inferência."""

    def test_sigmoid_produz_probabilidades(self, model, batch):
        """torch.sigmoid(logits) deve produzir valores estritamente em [0, 1]."""
        model.eval()
        with torch.no_grad():
            proba = torch.sigmoid(model(batch))
        assert float(proba.min()) >= 0.0
        assert float(proba.max()) <= 1.0

    def test_logits_podem_ser_negativos(self, model):
        """Logits não têm restrição de range — devem poder ser negativos."""
        torch.manual_seed(0)
        model.eval()
        x = torch.randn(64, 49)
        with torch.no_grad():
            out = model(x)
        assert bool((out < 0).any()), "Esperado pelo menos um logit negativo no batch"


# ════════════════════════════════════════════════════════════════════════════════
# DETERMINISMO
# ════════════════════════════════════════════════════════════════════════════════

class TestDeterminism:
    """Em eval mode o modelo deve ser determinístico (dropout desativado)."""

    def test_eval_mode_deterministic(self, model, batch):
        """Duas chamadas com o mesmo input em eval() devem retornar outputs idênticos."""
        model.eval()
        with torch.no_grad():
            out1 = model(batch)
            out2 = model(batch)
        assert torch.allclose(out1, out2)

    def test_train_mode_nondeterministic(self, model, batch):
        """Em train mode o dropout ativo deve produzir outputs diferentes a cada chamada."""
        torch.manual_seed(0)
        model.train()
        out1 = model(batch)
        out2 = model(batch)
        assert not torch.allclose(out1, out2), "Dropout deveria tornar as saídas diferentes"


# ════════════════════════════════════════════════════════════════════════════════
# ARQUITETURA
# ════════════════════════════════════════════════════════════════════════════════

class TestArchitecture:
    """Verifica que a arquitetura configurada foi construída corretamente."""

    def test_parametros_arquitetura_padrao(self, model):
        """Arquitetura 49→64→32→1 deve ter exatamente 5.505 parâmetros treináveis.

        Cálculo:
            Linear(49, 64)   : 49×64 + 64   = 3.200
            BatchNorm1d(64)  : 64×2          =   128
            Linear(64, 32)   : 64×32 + 32   = 2.080
            BatchNorm1d(32)  : 32×2          =    64
            Linear(32, 1)    : 32×1 + 1     =    33
            Total                            = 5.505
        """
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == 5505

    def test_camada_de_saida_tem_dimensao_1(self, model):
        """A última camada deve ter exatamente 1 neurônio de saída (classificação binária)."""
        last_layer = list(model.net.children())[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        assert last_layer.out_features == 1

    @pytest.mark.parametrize("hidden_dims", [
        [32],
        [64, 32],
        [128, 64, 32],
    ])
    def test_numero_de_camadas_lineares(self, hidden_dims):
        """Número de camadas Linear deve ser len(hidden_dims) + 1 (saída)."""
        m = ChurnMLP(input_dim=10, hidden_dims=hidden_dims, dropout_rate=0.0)
        linear_layers = [l for l in m.net.children() if isinstance(l, torch.nn.Linear)]
        assert len(linear_layers) == len(hidden_dims) + 1
