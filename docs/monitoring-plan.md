# Plano de Monitoramento — Telco Churn MLP v1.0.0

> Baseado nas SLOs definidos no projeto e nas métricas de baseline da v1.0.0.
> Revisão recomendada a cada ciclo de retreinamento.

---

## 1. Visão Geral

O modelo é servido via API FastAPI em modo **batch mensal**: a equipe de dados executa inferência sobre a base de clientes ativos e entrega um ranking de risco para o CRM. O monitoramento tem dois objetivos:

1. **Detectar degradação de performance** antes que ela afete campanhas de retenção.
2. **Detectar drift de dados** que indique que o modelo foi treinado em uma distribuição diferente da produção atual.

---

## 2. Métricas Monitoradas

### 2.1 Performance do Modelo

| Métrica | Baseline (v1.0.0) | Limiar de Alerta | Limiar de Retreinamento | Frequência |
|---|---|---|---|---|
| **ROC-AUC** | 0,8611 | < 0,845 (–2%) | < 0,830 (–3,5%) | Mensal |
| **F1-Score** | 0,6537 | < 0,630 | < 0,610 | Mensal |
| **Recall** | 0,7204 | < 0,690 | < 0,660 | Mensal |
| **Precisão** | 0,5994 | < 0,560 | < 0,530 | Mensal |

> **Como medir:** Rótulos reais de churn do mês M são conhecidos no fechamento do mês M+1. Calcular métricas comparando a predição feita no início de M com os rótulos de M+1.

### 2.2 Drift de Dados (PSI — Population Stability Index)

Calculado mensalmente comparando a distribuição de cada feature do batch atual com a distribuição do dataset de treino (referência).

**Interpretação do PSI:**

| PSI | Classificação | Ação |
|---|---|---|
| < 0,10 | Estável | Nenhuma |
| 0,10 – 0,20 | Mudança moderada | Investigar + monitorar mais de perto |
| > 0,20 | Drift significativo | Acionar playbook de retreinamento |

**Features prioritárias para monitorar (top-5 por importância):**

| Feature | Tipo | PSI Alerta | PSI Retreinamento |
|---|---|---|---|
| `monthly_per_tenure` | derivada | 0,15 | 0,20 |
| `contract_risk_score` | derivada | 0,15 | 0,20 |
| `Tenure Months` | numérica | 0,15 | 0,20 |
| `Monthly Charges` | numérica | 0,15 | 0,20 |
| `Contract` | categórica | 0,10 | 0,20 |

> Features derivadas (`monthly_per_tenure`, `contract_risk_score`) dependem de features base. Se o PSI de uma feature derivada disparar, verifique as features componentes primeiro.

### 2.3 SLOs da API

| Objetivo | Meta | Limiar de Alerta | Medição |
|---|---|---|---|
| **Latência p99** | ≤ 200ms | > 250ms | Middleware de latência em `src/api/main.py` (structlog) |
| **Disponibilidade** | ≥ 99,5% | < 99,0% | Health check `GET /health` |
| **Taxa de erro (4xx/5xx)** | < 1% das requisições | > 2% | Logs da API |
| **Cobertura do modelo** | ≥ 99% dos clientes ativos | < 97% | Contagem de registros processados vs. total |

### 2.4 Qualidade dos Dados de Entrada

| Verificação | Limiar | Ação |
|---|---|---|
| Campos ausentes (null rate) | > 5% em qualquer coluna | Bloquear batch, acionar time de dados |
| Valores fora do domínio (Pandera) | Qualquer SchemaError | HTTP 422; investigar fonte do dado |
| Volume de registros | < 80% do batch histórico | Suspeita de falha no pipeline upstream |
| `Tenure Months = 0` | > 0 casos | Filtrar; podem representar clientes novos sem histórico |

---

## 3. Frequência e Responsáveis

| Atividade | Frequência | Responsável |
|---|---|---|
| Cálculo de PSI das features | Mensal (D+1 após batch) | Engenheiro de Dados |
| Avaliação de performance (ROC-AUC, F1) | Mensal (D+30 após batch, quando rótulos disponíveis) | Cientista de Dados |
| Verificação de SLOs da API | Semanal | Engenheiro de ML |
| Análise de qualidade dos dados de entrada | A cada batch | Automático (Pandera + logs) |
| Revisão completa do plano de monitoramento | Trimestral ou a cada retreinamento | Tech Lead de ML |
| Avaliação de fairness (subgrupos) | A cada retreinamento | Cientista de Dados |

---

## 4. Playbook de Resposta a Incidentes

### Cenário A — Queda de Performance (ROC-AUC < 0,83)

```
1. Verificar se o problema é de dados ou de modelo:
   a. Inspecionar PSI das top-5 features no período afetado.
   b. Se PSI > 0,20 → o problema é drift de dados → ir para Cenário B.
   c. Se PSI < 0,10 → investigar mudança na distribuição do target (taxa de churn real aumentou?).

2. Se o problema for mudança na taxa de churn base:
   a. Confirmar com equipe de negócio se houve evento externo (crise, concorrente, promoção).
   b. Se mudança estrutural confirmada → retreinar com dados mais recentes (últimos 12-18 meses).
   c. Se evento pontual → aguardar normalização; não retreinar.

3. Retreinamento:
   a. Executar: make train (garante reprodutibilidade via RANDOM_SEED=42).
   b. Comparar novo modelo com champion em holdout.
   c. Se novo ROC-AUC > champion em ≥ 0,005 → promover para produção.
   d. Registrar no MLflow com tag model_version e git_sha.

4. Comunicar ao time de CRM a eventual pausa ou ajuste no ranking de risco.
```

### Cenário B — Drift de Dados (PSI > 0,20)

```
1. Identificar a feature com maior PSI.

2. Para features derivadas (monthly_per_tenure, contract_risk_score):
   a. Verificar PSI das features componentes (Monthly Charges, Tenure, Contract).
   b. O problema pode ser upstream no pipeline de dados.

3. Investigar causa raiz:
   a. Mudança de produto: novo plano de contrato, reajuste de preços?
   b. Mudança de processo: coluna renomeada no sistema fonte?
   c. Sazonalidade esperada (ex: promoções de final de ano)?

4. Ações por causa:
   - Mudança de produto → retreinar com dados do novo período.
   - Bug de pipeline → corrigir extração; reprocessar batch afetado.
   - Sazonalidade pontual → documentar; monitorar nos próximos 2 meses.

5. Se PSI > 0,20 em ≥ 3 features simultaneamente → acionar retreinamento imediato.
```

### Cenário C — Degradação de Latência (p99 > 200ms)

```
1. Verificar logs do middleware (structlog) para identificar requests lentos.
2. Verificar se o problema é na inferência do modelo ou no I/O de dados.
3. Os artefatos são carregados UMA VEZ na inicialização (ChurnPredictor.__init__).
   Se o problema aparecer depois de horas em produção, verificar memory leak.
4. Reiniciar o serviço como medida paliativa imediata.
5. Investigar profiling da chamada ao modelo (forward pass + sigmoid).
6. Para latência sistemática > 300ms, avaliar otimização do modelo (quantização, ONNX).
```

### Cenário D — Alta Taxa de Erros Pandera (> 5% dos registros)

```
1. Inspecionar mensagem de erro nos logs (campo schema_error).
2. Verificar se o range ou categoria inválida é esperada (novo produto?) ou bug.
3. Se novo produto/serviço foi lançado:
   a. Atualizar src/schemas/common.py com os novos domínios.
   b. Avaliar impacto na feature engineering.
   c. Se impacto em features → retreinar modelo.
4. Se bug de pipeline upstream:
   a. Corrigir na fonte.
   b. Reprocessar registros rejeitados.
```

---

## 5. Baseline de Referência (v1.0.0)

Distribuição das features no dataset de treino — usar como referência para cálculo de PSI:

| Feature | Tipo | Min | Mediana | Máx | % Missing |
|---|---|---|---|---|---|
| `Tenure Months` | int | 1 | 29 | 72 | 0% |
| `Monthly Charges` | float | 18,25 | 64,44 | 118,75 | 0% |
| `Total Charges` | float | 18,85 | 1.397 | 8.684 | 0% |
| `Contract` | cat | — | Month-to-month (55%) | — | 0% |
| `Internet Service` | cat | — | Fiber optic (44%) | — | 0% |
| `monthly_per_tenure` | float derivada | 0,26 | 2,69 | 118,75 | 0% |
| `services_count` | int derivada | 0 | 3 | 6 | 0% |

> Distribuições completas disponíveis em `notebooks/01_eda.ipynb`.

---

## 6. Links e Referências

| Recurso | Localização |
|---|---|
| Model Card (baseline de métricas) | [docs/model-card.md](model-card.md) |
| Diagramas de arquitetura (treino, inferência, sequência da API) | [docs/architecture_diagrams.md](architecture_diagrams.md) |
| Schemas Pandera (domínios válidos) | `src/schemas/common.py` |
| ADR threshold (custo-sensitivo) | [docs/decisions/002-threshold-cost-sensitive.md](decisions/002-threshold-cost-sensitive.md) |
| Resultados comparativos MLflow | [docs/results.md](results.md) |
