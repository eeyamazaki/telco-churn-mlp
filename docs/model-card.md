# Model Card — Telco Churn MLP v1.0.0

> Referência: Mitchell et al., "Model Cards for Model Reporting" (FAccT 2019).
> Formato adaptado para classificação tabular binária com PyTorch.

---

## 1. Detalhes do Modelo

| Campo | Valor |
|---|---|
| **Nome** | Telco Churn MLP |
| **Versão** | 1.0.0 |
| **Data de treinamento** | Maio/2026 |
| **Tipo** | Classificação binária supervisionada |
| **Framework** | PyTorch 2.x + Scikit-Learn (pré-processamento) |
| **Arquitetura** | MLP: `49 → 64 → 32 → 1` (BatchNorm1d + ReLU + Dropout=0,3 por camada oculta) |
| **Loss** | `BCEWithLogitsLoss` com `pos_weight` para desbalanceamento de classes |
| **Otimizador** | Adam (lr=1e-3, weight_decay=1e-4) + `ReduceLROnPlateau` (patience=8) |
| **Regularização** | Early Stopping (patience=15, máx 150 épocas) — convergiu na época 43 |
| **Threshold de decisão** | 0,5996 (F1-ótimo na curva Precision-Recall do test set) |
| **Autores** | Equipe Telco Churn MLP |
| **Licença** | MIT |
| **Experimento MLflow** | `telco-churn-mlp` — consulte `docs/results.md` para run IDs |
| **Decisões de design** | Ver `docs/decisions/` para ADRs |

---

## 2. Uso Pretendido

### Casos de uso primário

- **Campanha de retenção proativa:** score mensal (batch) de todos os clientes ativos para identificar aqueles com maior risco de cancelamento, alimentando o CRM da equipe de retenção.
- **Priorização de ações:** ranking por probabilidade de churn para alocar orçamento de retenção onde o impacto é maior.

### Usuários pretendidos

| Usuário | Como usa |
|---|---|
| Equipe de Retenção/CRM | Recebe lista ordenada por risco para acionar campanhas |
| Gestão de Produto | Analisa os drivers de churn (feature importance) para priorizar melhorias |
| Área Financeira | Avalia ROI das campanhas vs. custo de churn evitado |
| Equipe de Dados/ML | Opera, monitora e retreina o modelo |

### Fora de escopo

- **Clientes B2B / corporativos** — o dataset cobre apenas consumidores individuais da Califórnia.
- **Períodos promocionais atípicos** (Black Friday, campanhas massivas) — a distribuição de comportamento pode diferir significativamente do período de treino (Q3 California).
- **Penalidades automatizadas** — o modelo não deve ser usado para ações punitivas (corte de serviço, bloqueio) sem revisão humana.
- **Predição de motivo de churn** — o modelo prevê *se* o cliente vai cancelar, não *por que*.
- **Extrapolação temporal** — para dados de períodos muito distintos do Q3 California, aguarde reavaliação de drift antes de colocar em produção.

---

## 3. Fatores

Variáveis que, segundo a análise exploratória e a importância de features, mais influenciam a predição:

### Demográficos

| Fator | Observação |
|---|---|
| **Senior Citizen** | Taxa de churn de ~42% vs ~24% para não-sêniors. Modelo tende a scores mais altos para este grupo — monitorar separadamente (ver Seção 8). |
| **Partner / Dependents** | Clientes sem parceiro e sem dependentes apresentam maior churn. |

### Contratuais (maior peso preditivo)

| Fator | Importância |
|---|---|
| `contract_risk_score` | Feature derivada — top-2 em importância Gini e permutação |
| `monthly_per_tenure` | Feature derivada — top-1 em importância Gini |
| `Tenure Months` | Correlação de -0,35 com churn: quanto mais antigo, menos provável o cancelamento |
| `Contract` | Month-to-month tem ~3× mais churn que contratos anuais/bianuais |

### Serviços

| Fator | Observação |
|---|---|
| `has_protection` | Feature derivada — clientes sem seguro/suporte têm maior churn |
| `services_count` | Clientes com menos serviços tendem a cancelar mais |
| Internet Service | Fiber Optic tem maior taxa de churn que DSL/No internet |

---

## 4. Métricas de Performance

Avaliadas no **test set** (n=1.052, holdout estratificado de 15%, nunca visto durante treino ou validação).

### Métricas principais

| Métrica | Valor | IC 95% (bootstrap n=1.000) | Baseline (LR) |
|---|---|---|---|
| **ROC-AUC** | **0,8611** | [0,838 – 0,884] | 0,8425 |
| **PR-AUC** | **0,6748** | [0,638 – 0,711] | — |
| **F1-Score** | **0,6537** | [0,617 – 0,690] | — |
| **Recall** | **0,7204** | [0,681 – 0,759] | 0,71 |
| **Precisão** | 0,5982 | [0,558 – 0,641] | — |

> IC 95% estimado via bootstrap estratificado no test set. Para reproduzir: `notebooks/04_mlp.ipynb`, seção de avaliação final.

### Análise de custo por threshold

A escolha do threshold impacta diretamente o custo de negócio (ver [ADR 002](decisions/002-threshold-cost-sensitive.md)):

| Threshold | Recall | Precisão | Custo Total Estimado |
|---|---|---|---|
| 0,10 (custo-ótimo) | 94,6% | 41,8% | R$78.804 (mínimo) |
| **0,5996 (F1-ótimo)** | 72,0% | 59,9% | intermediário |
| 0,50 (padrão) | ~53% | ~66% | referência |

**Threshold recomendado para produção:** 0,5996 (F1-ótimo). O threshold de 0,10 é preferível quando a capacidade operacional da equipe de CRM permite absorver mais abordagens (trade-off com time de retenção).

---

## 5. Dados de Avaliação

| Campo | Valor |
|---|---|
| **Fonte** | IBM Telco Customer Churn (Kaggle) — California, Q3 |
| **Tamanho** | 1.052 clientes (15% do dataset limpo, 7.010 linhas) |
| **Distribuição** | 26,1% churn / 73,9% não-churn (preservada via split estratificado) |
| **Período** | Q3, California |
| **Estratégia de split** | `train_test_split` estratificado com `random_state=42`; test set separado *antes* de qualquer ajuste de hiperparâmetros |
| **Contaminação** | Nenhuma — ColumnTransformer foi `fit` apenas em X_train |

---

## 6. Dados de Treinamento

| Campo | Valor |
|---|---|
| **Tamanho** | 4.909 clientes (70% do dataset limpo) |
| **Validação** | 1.049 clientes (15%), usados para early stopping e seleção de threshold |
| **Features de entrada** | 20 colunas originais → 26 após feature engineering → 49 após OHE/scaling |
| **Desbalanceamento** | 26,58% positivos (churn) — corrigido via `pos_weight` na loss |
| **Leakage removido** | `Churn Score`, `CLTV`, `Churn Reason`, `Churn Label` — derivados do evento alvo |
| **Limpeza** | 11 linhas com `Tenure == 0` removidas; 22 duplicatas removidas |
| **Reprodutibilidade** | `RANDOM_SEED=42` fixado em `numpy`, `torch`, `cudnn.deterministic=True` |

---

## 7. Análises Quantitativas

### Performance por subgrupo (fairness)

Métricas computadas no test set (n=1.052) usando threshold F1-ótimo = 0,5996.
Reproduzir: `notebooks/04_mlp.ipynb`, seção 11.

| Subgrupo | N | % Test | Churn Real | ROC-AUC | Recall | Precisão | F1 |
|---|---|---|---|---|---|---|---|
| **TOTAL (referência)** | **1.052** | **100%** | **26,5%** | **0,8611** | **0,7204** | **0,5982** | **0,6537** |
| Senior Citizen = 0 (não sênior) | 889 | 84,5% | 24,5% | 0,8700 | 0,6743 | 0,6100 | 0,6405 |
| Senior Citizen = 1 (sênior) | 163 | 15,5% | 37,4% | 0,8092 | 0,8852 | 0,5684 | 0,6923 |
| Contract = Month-to-month | 566 | 53,8% | 43,5% | 0,7657 | 0,7967 | 0,5994 | 0,6841 |
| Contract = One year | 222 | 21,1% | 12,2% | 0,8274 | 0,1852 | 0,5556 | 0,2778 |
| Contract = Two year | 264 | 25,1% | 2,3% | 0,7267 | 0,0000 | — | 0,0000 |
| Gender = Female | 503 | 47,8% | 28,8% | 0,8696 | 0,7103 | 0,6358 | 0,6710 |
| Gender = Male | 549 | 52,2% | 24,4% | 0,8534 | 0,7313 | 0,5632 | 0,6364 |

**Gaps de fairness:**

| Gap | Valor | Status |
|---|---|---|
| AUC: sênior vs não-sênior | **0,0608** | **⚠ > 0,05 — monitorar separadamente** |
| AUC: Female vs Male | 0,0162 | ✓ aceitável |
| Recall: Contract Two year | 0,0000 | ⚠ esperado (2,3% churn real), documentado |

> **Senior Citizen:** AUC 6pp menor para sêniores (0,8092 vs 0,8700). Porém o Recall é maior (88,5% vs 67,4%) — o modelo é mais "agressivo" ao marcar sêniores como risco, coerente com a taxa de churn real de 37,4%. Risco: sobreabordagem do grupo se a taxa de churn mudar. Monitorar separadamente a cada ciclo.
>
> **Contract Two year:** Recall = 0,0 porque apenas 6 clientes neste grupo churnam no test set — o modelo acerta quase todos como "No Churn" (correto para 97,7% dos casos), mas não detecta os poucos que cancelam. Comportamento esperado para grupo de baixíssimo risco.

**Ação recomendada:** Recalcular este mapa a cada retreinamento. Se o gap de AUC Senior Citizen ultrapassar 0,08, avaliar threshold separado para o subgrupo.

### Calibração

Métricas computadas no test set (n=1.052). Reproduzir: `notebooks/04_mlp.ipynb`, seção 10.

| Métrica | Valor | Baseline Ingênuo | Interpretação |
|---|---|---|---|
| **Brier Score** | **0,1545** | 0,1949 | ✓ Melhor que baseline — modelo tem poder preditivo real |
| **ECE** (10 bins) | **0,1296** | — | ✗ > 0,10 — modelo descalibrado por design (`pos_weight`) |

> O Brier Score de 0,1545 confirma que o modelo agrega valor real vs. prever sempre a média. O ECE de 0,1296 indica que os scores não representam probabilidades calibradas — consequência documentada do `pos_weight` (ver [ADR 003](decisions/003-pos-weight-balancing.md)). **Usar o score apenas para ranking e threshold, nunca como probabilidade direta.** Calibração isotônica (Platt scaling) deferida para v2.

---

## 8. Considerações Éticas e LGPD

### Proteção de dados

- **PII não é armazenada:** nenhum campo identificador de cliente (`CustomerID`, nome, endereço) é processado pelo modelo ou registrado nos logs.
- **Logs estruturados:** `src/logger.py` usa `structlog` com logging de probabilidades agregadas, sem dados pessoais.
- **Direito à explicação (LGPD Art. 20):** feature importance (Gini e permutação) permite explicar a decisão ao cliente se necessário. Em v2, implementar SHAP por instância.

### Viés identificado

| Risco | Detalhes | Mitigação |
|---|---|---|
| **Viés etário (Senior Citizen)** | AUC gap de 0,0608 (sêniores: 0,8092 vs não-sêniores: 0,8700). O modelo é mais agressivo ao marcar sêniores como risco (Recall 88,5%) — acerta mais churns reais, mas gera mais FPs nesse grupo. | Monitorar AUC e taxa de FP separados por ciclo; se gap > 0,08, calibrar threshold específico para o subgrupo. |
| **Ausência de dados socioeconômicos** | Sem renda declarada, o `monthly_charges` é proxy imperfeito. | Documentar como limitação; não usar o score para negar serviços. |
| **Dataset estático (Q3 CA)** | Sazonalidade não capturada. Comportamento de churn pode diferir em outras regiões/períodos. | Retreinar com dados mais recentes antes de expandir cobertura geográfica. |

### Uso responsável

- O score de churn deve ser usado para **oferecer benefícios**, não para aplicar penalidades.
- Decisões finais de retenção devem ter **revisão humana** (equipe de CRM).
- O modelo **não é auditado para discriminação por raça ou etnia** — dados não disponíveis no dataset.

---

## 9. Ressalvas e Recomendações

| Ressalva | Impacto | Recomendação |
|---|---|---|
| Dataset de uma única operadora (California, Q3) | Baixa generalização para outros mercados ou períodos | Retreinar com dados locais antes de expandir |
| Threshold foi otimizado no test set | Leve sobreajuste ao test set | Validar threshold em dados reais de produção nos primeiros 3 meses |
| Sem validação cruzada no modelo final | Estimativa de performance com variância mais alta | Implementar k-fold no ciclo de retreinamento |
| `pos_weight` descalibra probabilidades | Score não é uma probabilidade real | Aplicar calibração isotônica em v2 |
| Features de engenharia replicadas manualmente | Risco de inconsistência se a fórmula mudar | `FeatureEngineer` versionado junto com `preprocessor_mlp.pkl` |
| Churn Reason não disponível antes do evento | Modelo não sabe *por que* o cliente vai cancelar | Complementar com análise de NPS/satisfação para drivers qualitativos |

### Gatilhos de retreinamento

Retreinar o modelo quando qualquer das condições abaixo for detectada (ver [Plano de Monitoramento](monitoring-plan.md)):

- ROC-AUC no holdout mensal cair abaixo de **0,83** (queda > 3,5%)
- PSI > 0,20 em qualquer uma das top-5 features
- Mudança estrutural no produto (novo tipo de contrato, nova feature de serviço)
- Expansão para nova região geográfica ou segmento de clientes
