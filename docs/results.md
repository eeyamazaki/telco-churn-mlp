# Resultados Comparativos — Experimentos MLflow

> Tabela exportada do experimento `telco-churn-mlp` no MLflow.
> Todos os modelos avaliados no **mesmo test set** (n=1.052, 15% estratificado, `random_state=42`).
> Para detalhes completos: `mlflow ui` na raiz do projeto.

---

## Comparação Final: 5 Modelos

| Modelo | ROC-AUC | PR-AUC | F1 | Recall | Precisão | Threshold |
|---|---|---|---|---|---|---|
| DummyClassifier (most_frequent) | 0,500 | ~0,266 | — | 0,000 | — | 0,5 |
| Logistic Regression (baseline) | 0,8425 | — | — | 0,710 | — | 0,5 |
| Gradient Boosting (tuned) | — | — | — | — | — | 0,5 |
| Random Forest | — | — | — | — | — | 0,5 |
| **MLP PyTorch (final)** | **0,8611** | **0,6748** | **0,6537** | **0,7204** | **0,5982** | **0,5996** |

> Campos `—` devem ser preenchidos a partir do MLflow UI (`mlflow ui`, aba "Experiments → telco-churn-mlp").
> Comando para exportar: `mlflow runs list --experiment-name telco-churn-mlp --view_type ALL`.

---

## Evolução do MLP — Ablation Study

| Configuração | ROC-AUC | F1 | Recall | Nota |
|---|---|---|---|---|
| MLP [64] sem BatchNorm | — | — | — | experimento inicial |
| MLP [64, 32] sem Dropout | — | — | — | overfitting observado |
| MLP [64, 32] + Dropout(0.3) | — | — | — | melhor val_loss |
| **MLP [64, 32] + BN + Dropout + pos_weight** | **0,8611** | **0,6537** | **0,7204** | **modelo final** |

---

## Análise de Threshold — MLP Final

| Threshold | Recall | Precisão | F1 | Custo Total Estimado |
|---|---|---|---|---|
| 0,05 | ~97% | ~35% | — | a calcular |
| **0,10 (custo-ótimo)** | **94,6%** | **41,8%** | — | **R$78.804 (mínimo)** |
| 0,30 | — | — | — | — |
| **0,5996 (F1-ótimo)** | **72,0%** | **59,9%** | **0,6537** | intermediário |
| 0,70 | — | — | — | — |

---

## Parâmetros do Modelo Final

| Parâmetro | Valor |
|---|---|
| `hidden_dims` | [64, 32] |
| `dropout_rate` | 0,3 |
| `batch_size` | 64 |
| `learning_rate` | 1e-3 |
| `weight_decay` | 1e-4 |
| `lr_patience` | 8 |
| `early_stopping_patience` | 15 |
| `max_epochs` | 150 |
| `convergiu_epoch` | 43 |
| `pos_weight` | ~2,76 (n_neg/n_pos no X_train) |
| `random_seed` | 42 |

---

## Artefatos Salvos

| Artefato | Caminho | Descrição |
|---|---|---|
| Pré-processador | `models/preprocessor_mlp.pkl` | ColumnTransformer treinado (StandardScaler + OHE) |
| Pesos do modelo | `models/mlp_weights.pt` | `state_dict` do ChurnMLP |
| Configuração | `models/mlp_config.json` | Arquitetura, dropout, threshold |
| Experimento MLflow | `mlruns/` (local) | Parâmetros, métricas, curvas de loss, model snapshot |

---

## Como Reproduzir

```bash
# 1. Limpeza dos dados
make clean-data

# 2. Treinamento e registro no MLflow
make train

# 3. Verificar resultados
mlflow ui
# Abrir http://localhost:5000 → Experiments → telco-churn-mlp
```
