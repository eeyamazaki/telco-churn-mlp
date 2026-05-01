# ADR 003: pos_weight para tratamento de desbalanceamento de classes

## Status
Aceita — Maio/2026

## Contexto

O dataset de churn é desbalanceado: **26,58% positivos (churn)** vs. 73,42% negativos.
Sem correção, um classificador pode atingir 73,4% de acurácia simplesmente prevendo
"Não churn" para todos os registros — e essa acurácia alta é enganosa para o problema de negócio.

As principais estratégias disponíveis eram:

1. **Nada** — treinar sem correção
2. **Oversampling** (SMOTE, random oversampling) — criar instâncias sintéticas da classe minoritária
3. **Undersampling** — remover instâncias da classe majoritária
4. **`class_weight='balanced'`** — abordagem scikit-learn
5. **`pos_weight` na loss** — abordagem nativa PyTorch
6. **Ajuste de threshold** — pós-treino, sem alterar o treinamento

## Decisão

Adotamos **`pos_weight` na `BCEWithLogitsLoss`** do PyTorch como estratégia primária,
complementada pelo ajuste de threshold em pós-processamento (ADR 002).

```python
# src/training/train.py
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)  # ≈ 2.76

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

Isso penaliza o modelo **2,76× mais** por errar um caso positivo (churn real) do que um negativo.

## Justificativas

**Por que `pos_weight` e não SMOTE?**

| Critério | `pos_weight` | SMOTE |
|---|---|---|
| Gera dados sintéticos | Não | Sim — risco de artefatos |
| Mantém o dataset original intacto | Sim | Não |
| Compatível com ColumnTransformer | Diretamente | Requer ajuste no pipeline |
| Risco de overfitting em features categóricas | Baixo | Alto (interpolação não faz sentido) |
| Reprodutibilidade | Determinística com seed | Depende de seed do SMOTE |
| Implementação | 2 linhas | Pipeline adicional |

> SMOTE em dados tabulares com muitas features categóricas gera instâncias sintéticas por
> interpolação linear, o que não é semanticamente válido para variáveis como `Contract` ou
> `Internet Service Type`. Evitamos SMOTE por essa razão.

**Por que `pos_weight` e não `class_weight='balanced'` do sklearn?**

`class_weight='balanced'` é a solução equivalente para modelos sklearn. No contexto de PyTorch
com `BCEWithLogitsLoss`, o equivalente é `pos_weight`. São matematicamente equivalentes — a
escolha foi pelo idioma nativo do framework.

**Por que não undersampling?**

Com apenas 7.010 registros, remover amostras da classe majoritária reduziria o dataset de treino
para ~3.700 linhas — muito pequeno para uma rede neural com 49 features. Descartado.

**Por que manter ajuste de threshold como complemento?**

O `pos_weight` desloca a distribuição de probabilidades do modelo (scores tendem a ser maiores
que 0,5 para a maioria dos registros). Isso torna o threshold de 0,5 subótimo mesmo para F1.
O ajuste de threshold (ADR 002) é o complemento natural: `pos_weight` corrige o treinamento,
threshold corrige a decisão em pós-processamento.

## Consequências

**Positivas:**
+ Dataset de treino preservado (sem amostras sintéticas ou removidas)
+ Treinamento determinístico
+ Implementação mínima (2 linhas em `train.py`)
+ Recall do modelo aumenta significativamente vs. baseline sem correção

**Negativas:**
- O `pos_weight` desloca a distribuição de probabilidades: scores > 0,3 devem ser
  interpretados como "alto risco", não como probabilidade real de 30%
- Para calibração probabilística real, seria necessário Platt scaling em pós-processamento
  (deferida para v2 — ver `docs/model-card.md`, Seção 7)
- O valor de `pos_weight` deve ser recalculado a cada retreinamento se a proporção de classes mudar

## Referências

- `src/training/train.py` — cálculo e aplicação do `pos_weight`
- `notebooks/04_mlp_pytorch.ipynb` — análise do impacto no treinamento
- `docs/decisions/002-threshold-cost-sensitive.md` — complemento: ajuste de threshold
- `docs/model-card.md` — seção 7: calibração e seção 9: ressalva sobre probabilidades
