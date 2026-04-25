"""Arquitetura MLP para classificação binária de churn."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
