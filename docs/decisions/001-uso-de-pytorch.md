# ADR 001: Rede Neural MLP com PyTorch

## Status
Aceita — Maio/2026

## Contexto

O projeto requer um modelo de **rede neural** para previsão de churn. A questão de design era:
qual framework usar para implementar o MLP, e por que MLP e não apenas os modelos de ensemble
que apresentaram boa performance na fase de model engineering?

O dataset tem 7.010 linhas, 49 features após pré-processamento — escala pequena/média, onde
Gradient Boosting (scikit-learn) também é competitivo.

## Decisão

Adotamos **PyTorch** para implementar o MLP, mantendo scikit-learn para pré-processamento
(ColumnTransformer), modelos baseline e comparação de ensemble.

**Arquitetura do MLP:**
- Input: 49 features (StandardScaler + OneHotEncoder via ColumnTransformer)
- Camadas ocultas: [64, 32] — configurável via `mlp_config.json`
- Por camada: `Linear → BatchNorm1d → ReLU → Dropout(0.3)`
- Output: `Linear(32 → 1)` + sigmoid implícita na loss (`BCEWithLogitsLoss`)

## Justificativas

**Por que PyTorch e não sklearn MLP (`MLPClassifier`)?**

| Critério | PyTorch | sklearn MLPClassifier |
|---|---|---|
| Controle do loop de treino | Total (loss por época, early stopping customizado) | Limitado |
| `pos_weight` na loss | Nativo (`BCEWithLogitsLoss(pos_weight=...)`) | Não disponível |
| `ReduceLROnPlateau` | Nativo (`torch.optim.lr_scheduler`) | Não disponível |
| Rastreamento MLflow | `mlflow.pytorch.log_model` (integração nativa) | Requer workaround |
| Extensibilidade | Arquitetura configurável via JSON | Fixa |
| Extensão futura (TorchScript) | Pronto para servir sem sklearn em runtime | Não disponível |

**Por que MLP e não apenas Gradient Boosting (que também teve boa performance)?**

- O GB atingiu ROC-AUC de ~0,86 na fase de model engineering, similar ao MLP.
- O MLP foi escolhido para compor o stack de produção — neural networks lidam melhor com interações não-lineares entre features de comportamento do que árvores boosted em dados de serviços.
- O MLP final (0,8611) superou o GB em F1 e Recall na comparação final com os mesmos dados.
- A arquitetura PyTorch permite servir via `torch.jit.script` em produção de alta escala, sem dependência do scikit-learn em runtime (roadmap v2).

## Consequências

**Positivas:**
- Loop de treino completamente auditável (loss por época registrada no MLflow)
- Controle total sobre o tratamento do desbalanceamento de classes (`pos_weight`)
- Arquitetura e threshold armazenados em `mlp_config.json` — não há "magia" no código de inferência
- Separa claramente "treinamento" (PyTorch) de "pré-processamento" (scikit-learn)

**Negativas:**
- Maior verbosidade de código vs. `model.fit()` do scikit-learn
- Dependência de `torch` (~700MB) em produção
- Reprodutibilidade em GPU requer `cudnn.deterministic=True` e `cudnn.benchmark=False`

## Alternativas Consideradas

| Alternativa | Razão de Rejeição |
|---|---|
| `sklearn.neural_network.MLPClassifier` | Sem `pos_weight`; não atende requisito de PyTorch |
| XGBoost | Não atende requisito de rede neural |
| PyTorch Lightning | Overhead desnecessário para dataset pequeno |
| TensorFlow/Keras | Preferência da equipe por PyTorch; ecossistema mais consistente com requisito |

## Referências

- `src/models/mlp.py` — implementação do `ChurnMLP`
- `src/training/train.py` — loop de treino com early stopping e `pos_weight`
- `mlp_config.json` — configuração da arquitetura salva junto com os pesos
- `notebooks/04_mlp_pytorch.ipynb` — experimentos e comparação final
- MLflow experiment: `telco-churn-mlp` (ver `docs/results.md`)
