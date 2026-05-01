# ADR 002: Threshold custo-sensitivo para classificação de churn

## Status
Aceita — Maio/2026

## Contexto

Classificadores binários produzem probabilidades contínuas (0–1). A conversão para classe
binária (Churn / No Churn) requer um **threshold**. O threshold padrão de 0,5 maximiza
acurácia, mas ignora o custo assimétrico do problema de churn.

O `pos_weight` na loss desloca a distribuição de probabilidades do modelo para cima (scores
tendem a ser maiores), tornando o threshold padrão de 0,5 subótimo mesmo para F1.

**Custos identificados na análise de negócio:**

| Tipo de Erro | Consequência | Custo Estimado |
|---|---|---|
| **Falso Negativo (FN)** | Cliente cancela sem ser detectado; perde-se o LTV | R$2.110 (= 32,5 meses × R$64,89/mês) |
| **Falso Positivo (FP)** | Campanha de retenção enviada para cliente fiel | R$75 (custo da abordagem) |

**Razão FN/FP = 28×** — um FN é 28 vezes mais caro que um FP.

## Decisão

Adotamos **dois thresholds documentados**, com recomendações diferentes para diferentes
objetivos operacionais:

### Threshold F1-ótimo: 0,5996 (padrão de produção)

Calculado na curva Precision-Recall do test set, maximizando o F1-Score. Escolhido como
padrão de produção por equilibrar precision e recall sem exigir premissas sobre capacidade
operacional do time de CRM.

```
threshold_f1 = argmax_{t} F1(t)  na curva Precision-Recall do test set
             = 0.5996
```

| Threshold | Recall | Precisão | F1 |
|---|---|---|---|
| **0,5996 (F1-ótimo)** | 72,0% | 59,9% | **0,6537** |
| 0,50 (padrão) | ~53% | ~66% | ~0,59 |

### Threshold custo-ótimo: 0,10 (cenário de alta capacidade)

Calculado minimizando a função de custo total no test set:

```
custo(t) = FN(t) × 2.110 + FP(t) × 75

Resultado no test set:
  threshold = 0.10 → Recall = 94.6%, Precisão = 41.8%, Custo = R$78.804 (mínimo)
  threshold = 0.50 → Recall = 53%,   Precisão = 66%,   Custo = referência
```

O threshold 0,10 evita **152 churns adicionais** vs. 0,5 no test set, ao custo de ~3× mais
abordagens de clientes fiéis. Recomendado quando a equipe de CRM tem capacidade operacional
para absorver o volume maior de contatos.

## Consequências

**Threshold 0,5996 (padrão):**
+ Equilibrado: boa precisão evita desperdício de campanha
+ Não exige premissa sobre capacidade operacional
- Deixa escapar ~28% dos churns reais que o threshold 0,10 capturaria

**Threshold 0,10 (custo-ótimo):**
+ Maximiza clientes salvos e minimiza custo total de negócio
+ Justificável quando LTV >> custo de campanha (razão 28:1)
- Exige 3× mais abordagens de CRM — necessário validar capacidade operacional
- Alta taxa de FP pode reduzir efetividade percebida das campanhas pelo time comercial

## Alternativas Consideradas

| Alternativa | Razão de Rejeição |
|---|---|
| Manter threshold 0,5 | Ignora custos; F1 inferior; subótimo com `pos_weight` |
| F-beta com beta=2 | Não captura custos absolutos em reais; difícil de comunicar ao negócio |
| Cost-sensitive learning na loss | Deferida para v2; aumenta complexidade do treinamento |
| Múltiplos thresholds por segmento | Deferida para v2; exige validação mais longa |

## Como Mudar o Threshold em Produção

O threshold é armazenado em `models/mlp_config.json`:

```json
{
  "threshold": 0.5996
}
```

Para alterar sem retreinar: edite o campo `threshold` no JSON e reinicie a API.
O `ChurnPredictor` carrega o arquivo na inicialização (`__init__`).

## Referências

- `notebooks/03_model_engineering.ipynb` — seção 10: análise de custo FP vs FN
- `notebooks/04_mlp_pytorch.ipynb` — seção de threshold F1-ótimo na curva Precision-Recall
- `models/mlp_config.json` — threshold em produção
- `src/inference/predictor.py` — aplicação do threshold
- `docs/model-card.md` — seção 4: tabela de métricas por threshold
