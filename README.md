# Telco Customer Churn — Previsão de Cancelamento de Clientes

## Descrição do Projeto

Uma empresa de telecomunicações enfrenta alta taxa de **churn** (cancelamento de clientes).
Adquirir um novo cliente custa de 5 a 25 vezes mais do que reter um existente, o que torna
a retenção proativa uma alavanca de negócio crítica.

Este projeto implementa um sistema **end-to-end de Machine Learning** que prevê quais clientes
têm maior risco de cancelar o serviço, permitindo que a empresa atue preventivamente com
campanhas de retenção direcionadas.

O modelo central é uma **rede neural (MLP)** construída com PyTorch, complementada por
modelos baseline (Logistic Regression, DummyClassifier) e ensemble (Gradient Boosting,
Random Forest) para referência de performance, todos rastreados com MLflow.

---

## ML Canvas

### 1. Proposta de Valor

**Problema:** Uma empresa de telecom da Califórnia perde clientes sem conseguir agir preventivamente.

**Solução:** Um modelo que identifica, com antecedência, quais clientes têm alta probabilidade de cancelar — permitindo que a equipe de retenção/CRM aja *antes* do cancelamento acontecer.

---

### 2. Stakeholders

| Papel | Interesse |
|---|---|
| **Equipe de Retenção/CRM** | Recebe a lista de clientes em risco para acionar campanhas |
| **Gestão de Produto** | Usa os drivers de churn para priorizar melhorias no serviço |
| **Área Financeira** | Avalia o ROI das campanhas vs. custo do churn |
| **Equipe de Dados/ML** | Constrói, monitora e retreina o modelo |
| **Área de Atendimento** | ~25% dos churns são por suporte — impacto operacional direto |

---

### 3. Dados Disponíveis

**Fonte:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) — 7.032 clientes após limpeza (original: 7.043), California (Q3), 33 colunas originais.

| Grupo | Variáveis-chave | Uso no Modelo |
|---|---|---|
| Demográficas | Gender, Senior Citizen, Partner, Dependents | Feature |
| Contrato | Tenure Months, Contract, Payment Method, Paperless Billing | Feature (mais preditivas) |
| Serviços | Phone, Internet, Security, Backup, Tech Support, Streaming | Feature |
| Financeiro | Monthly Charges, Total Charges | Feature |
| **Engenharia** | monthly_per_tenure, contract_risk_score, services_count, has_protection, is_senior_alone, tenure_group | Feature (top preditoras — #1 e #2 em feature importance) |
| **Leakage** | Churn Score, CLTV | Excluir (derivados do evento) |
| **Target** | Churn Value (0/1) | Variável alvo |
| Pós-evento | Churn Reason | Excluir (só existe após o churn) |
| Geolocalização | City, Zip Code, Lat/Long | Excluir (sem variância útil no escopo) |

**Desbalanceamento:** 26.58% churn / 73.42% não-churn (ratio ≈ 1:2.8)

---

### 4. Definição da Tarefa de ML

| Item | Decisão |
|---|---|
| **Tipo de tarefa** | Classificação binária supervisionada |
| **Output do modelo** | Probabilidade de churn (0.0 – 1.0) |
| **Decisão de negócio** | Threshold ajustável (padrão 0.5, tunable) |
| **Granularidade** | Por cliente individual |
| **Frequência** | Batch mensal (ou semanal no futuro) |

---

### 5. Métricas de Sucesso

#### Métricas Técnicas (ML)

| Métrica | Por que usar | Meta |
|---|---|---|
| **ROC-AUC** | Mede poder discriminativo independente do threshold | ≥ 0.85 |
| **F1-Score** | Equilíbrio entre Precision e Recall | ≥ 0.65 |
| **Recall** | Prioridade: não deixar escapar churns reais | ≥ 0.75 |

#### Métricas de Negócio (KPIs)

| KPI | Definição | Meta |
|---|---|---|
| **Custo de churn evitado** | (FN × LTV_médio) − (FP × custo_campanha) | Maximizar |
| **Taxa de churn mensal** | % clientes que cancelaram no mês | Redução ≥ 15% vs. baseline histórico |
| **Receita retida** | Receita dos clientes salvos pelas campanhas | Medido por grupo de controle |
| **ROI da campanha** | Receita retida / custo da campanha de retenção | > 3× |

---

### 6. Trade-off FP vs FN (Análise de Custo)

| Erro | O que acontece | Custo |
|---|---|---|
| **Falso Negativo (FN)** | Cliente cancela sem ser detectado | Alto — perda do LTV médio (~R$2.110 por cliente) |
| **Falso Positivo (FP)** | Campanha desnecessária para cliente fiel | Baixo — custo da abordagem de retenção (~R$75) |

#### Fórmula de Custo

```
Custo_total = (FN × LTV_médio) + (FP × custo_campanha)

Onde:
  LTV_médio      = Tenure_médio × Monthly_Charges_médio
                 = 32,5 meses × R$64,89 = R$2.110,21 por cliente perdido

  custo_campanha = R$75,00 por abordagem desnecessária

Razão FN/FP    = 28x — FN é 28 vezes mais caro que FP
```

**Resultado da análise (03_model_engineering.ipynb, seção 10):**

| Threshold | Recall | Precisão | Custo Total |
|---|---|---|---|
| 0,50 (padrão) | ~53% | ~66% | referência |
| **0,10 (ótimo)** | **94,6%** | 41,8% | **R$78.804** (mínimo) |

**Estratégia:** Maximizar **Recall**, aceitando mais FPs. O custo de perder um cliente real (R$2.110) é **28× maior** que o custo de uma campanha desnecessária (R$75). O threshold ótimo de **0,10** evita 152 churns a mais do que o threshold padrão de 0,5. Threshold final deve ser validado com a equipe de CRM considerando capacidade operacional.

---

### 7. SLOs — Service Level Objectives

| Objetivo | Meta |
|---|---|
| **Latência de inferência (API)** | ≤ 200ms por requisição (p99) |
| **Disponibilidade da API** | ≥ 99.5% uptime |
| **Frequência de retreinamento** | Trimestral, ou se ROC-AUC cair > 5% |
| **Drift de dados** | Alerta se distribuição das features mudar > 10% (PSI) |
| **Cobertura do modelo** | ≥ 99% dos clientes ativos devem receber score |

---

### 8. Riscos e Limitações

| Risco | Mitigação |
|---|---|
| **Dataset estático (snapshot Q3 CA)** | Resultados podem não generalizar para outros estados/períodos |
| **Data leakage** | `Churn Score` e `CLTV` excluídos explicitamente |
| **Desbalanceamento de classes** | `class_weight='balanced'` na LR; `pos_weight` no MLP |
| **Viés demográfico** | Monitorar performance separada para Senior Citizens |
| **Ausência de dados temporais** | `Tenure Months` é proxy — sem histórico de transações |
| **Features de engenharia em produção** | Transformações devem ser replicadas no pipeline de inferência |

---

### 9. Pipeline de Produção

```
Dados novos (mensal)
      ↓
Validação de Schema (Pandera)
      ↓
Feature Engineering
(monthly_per_tenure, contract_risk_score, services_count,
 has_protection, is_senior_alone, tenure_group)
      ↓
Pré-processamento
(StandardScaler + OneHotEncoder — ColumnTransformer)
      ↓
Seleção de Features (SelectKBest — top 30)
      ↓
Inferência (MLP PyTorch / Gradient Boosting / FastAPI)
      ↓
Lista de risco → CRM
      ↓
Monitoramento (drift, métricas, latência)
      ↓
[Gatilho de retreinamento se necessário]
```

---

## Arquitetura

```
project-root/
├── src/
│   ├── data/            # Carregamento e validação de dados
│   ├── features/        # Transformações e feature engineering
│   ├── models/          # Definição dos modelos (MLP, baselines)
│   ├── training/        # Loop de treino e avaliação
│   ├── inference/       # Pipeline de inferência
│   └── api/             # Serviço FastAPI
├── data/
│   ├── raw/             # Dados originais (imutáveis)
│   └── processed/       # Dados processados (telco_churn_cleaned.csv)
├── models/              # Modelos treinados serializados
├── notebooks/           # Análises exploratórias e experimentos
├── tests/               # Testes automatizados (pytest)
├── docs/                # Documentação e Model Card
├── pyproject.toml
├── Makefile
└── README.md
```

---

## Como Usar

### Instalação

```bash
# Clonar o repositório
git clone <repo-url>
cd telco-churn-mlp

# Instalar dependências
pip install -e ".[dev]"
```

### Treinar o modelo

```bash
make train
```

### Rodar a API

```bash
make run-api
```

### Rodar os testes

```bash
make test
```

### Linting

```bash
make lint
```

---

## Tecnologias

- **PyTorch** — rede neural (MLP)
- **Scikit-Learn** — preprocessing pipelines e modelos baseline/ensemble
- **Scipy** — distribuições para busca de hiperparâmetros
- **MLflow** — rastreamento de experimentos
- **FastAPI** — API de inferência
- **pytest** — testes automatizados
- **pandera** — validação de schema dos dados
- **ruff** — linting

---

## Roadmap

**Estágio 1 — Entendimento e Preparação**
- [x] Definição do problema de negócio e ML Canvas
- [x] EDA completa (distribuições, correlações, missing values, análise de churn por segmento)
- [x] Modelos baseline (Dummy + Logistic Regression) com PR-AUC + MLflow

**Estágio 2 — Modelagem com Redes Neurais**
- [x] Feature Engineering (6 novas features — top preditoras em importância Gini e permutação)
- [x] Model Engineering (Decision Tree, Random Forest, SVM, Gradient Boosting + tuning com StratifiedKFold)
- [x] Rede neural MLP com PyTorch — arquitetura [64, 32], BatchNorm, Dropout, early stopping (epoch 28)
- [x] Análise de custo FP vs FN — threshold ótimo 0,10 (Recall 94,6%, custo mínimo R$78.804)
- [x] Comparação final: 6 métricas, 5 modelos — MLP com maior F1 (0,6495) e Recall (0,8136)

**Estágio 3 — Engenharia e API**
- [ ] Refatoração em módulos (`src/features`, `src/inference`, `src/api`)
- [ ] Pipeline reprodutível (sklearn + transformadores custom)
- [ ] API de inferência (FastAPI + Pydantic + logging estruturado + middleware de latência)
- [ ] Testes automatizados (smoke, schema Pandera, unitários)

**Estágio 4 — Documentação e Entrega**
- [ ] Model Card (performance, limitações, vieses, cenários de falha)
- [ ] Plano de monitoramento (métricas, alertas, playbook)
- [ ] STAR Video (5 min)
