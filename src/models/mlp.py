"""Arquitetura MLP para classificação binária de churn.

Define a classe ChurnMLP — uma rede neural feed-forward configurável,
implementada em PyTorch, utilizada para previsão de churn no dataset Telco.

Uso típico:
    from src.models import ChurnMLP

    model = ChurnMLP(input_dim=49, hidden_dims=[64, 32], dropout_rate=0.3)
    logits = model(features_tensor)

"""

import torch
import torch.nn as nn


class ChurnMLP(nn.Module):
    """MLP configurável para classificação binária de churn.

    A camada de saída retorna logits (sem Sigmoid), compatível com
    BCEWithLogitsLoss no treino e torch.sigmoid() na inferência.

    Parâmetros
    ----------
    input_dim : int
        Número de features após encoding (49 no pipeline atual).
    hidden_dims : list[int]
        Tamanho de cada camada oculta. Ex: [64, 32].
    dropout_rate : float
        Taxa de dropout aplicada após cada camada oculta.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev = input_dim

        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Propaga os dados pela rede e retorna logits.

        Parâmetros
        ----------
        features : torch.Tensor
            Tensor de shape (batch_size, input_dim) com as features encodadas.

        Retorno
        -------
        torch.Tensor
            Tensor de shape (batch_size, 1) com logits (não probabilidades).
            Para obter probabilidades, aplique torch.sigmoid() na inferência.
        """
        return self.net(features)
