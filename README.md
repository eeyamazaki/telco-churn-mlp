# Telco Customer Churn — Predição de Cancelamento de Clientes

## Descrição do Projeto

Uma empresa de telecomunicações enfrenta alta taxa de **churn** (cancelamento de clientes).
Adquirir um novo cliente custa de 5 a 25 vezes mais do que reter um existente, o que torna
a retenção proativa uma alavanca de negócio crítica.

Este projeto implementa um sistema **end-to-end de Machine Learning** que prevê quais clientes
têm maior risco de cancelar o serviço, permitindo que a empresa atue preventivamente com
campanhas de retenção direcionadas.

O modelo principal é uma **rede neural (MLP)** construída com PyTorch, complementada por
modelos baseline (Logistic Regression, DummyClassifier) para referência de performance.

## Problema de Negócio

### Contexto

O churn em telecomunicações é um dos problemas mais impactantes do setor. Quando um cliente
cancela, a empresa perde não apenas a receita recorrente, mas também o investimento feito
para adquiri-lo. Identificar sinais de insatisfação antes do cancelamento permite intervenções
como ofertas personalizadas, descontos ou contato proativo do time de retenção.

### Stakeholders

| Stakeholder | Papel | Como usa o modelo |
|---|---|---|
| **Equipe de Retenção/CRM** | Principal consumidor | Recebe a lista de clientes em risco e executa campanhas de retenção |
| **Gestão de Produto** | Consumidor estratégico | Usa insights para entender quais aspectos do serviço influenciam o churn |
| **Área Financeira** | Avaliação de ROI | Mede o retorno das ações de retenção baseadas no modelo |
| **Equipe de Dados/ML** | Mantenedor | Constrói, monitora e atualiza o modelo em produção |

### Métricas de Sucesso

**Métricas técnicas:**

- **ROC-AUC** — capacidade geral de discriminação entre clientes que vão ou não cancelar
- **PR-AUC** — performance em cenário desbalanceado (churn é tipicamente a classe minoritária)
- **F1-Score** — equilíbrio entre precision e recall
- **Precision** — dentre os clientes sinalizados como risco, quantos realmente cancelariam
- **Recall** — dentre os clientes que cancelaram, quantos o modelo conseguiu identificar

**KPI de negócio (critério de sucesso do projeto):**

O indicador-chave para avaliar se o projeto gerou resultado real é a **redução da taxa de
churn** após a implementação de ações preventivas baseadas no modelo. Concretamente:

- **Meta:** reduzir a taxa de churn mensal em pelo menos **15%** em relação à taxa atual,
  dentro de um período de avaliação de 3 meses após o deploy do modelo.
- **Como medir:** comparar a taxa de churn no grupo de clientes que receberam ações de
  retenção (identificados pelo modelo) versus a taxa de churn histórica ou de um grupo
  de controle sem intervenção.
- **Métrica complementar:** **custo de churn evitado** — valor de receita retida ao intervir
  nos clientes corretamente identificados, descontando o custo das campanhas de retenção
  (incluindo falsos positivos).

Esse KPI conecta a performance técnica do modelo ao impacto real no negócio: não basta o
modelo ter um bom ROC-AUC — ele precisa gerar uma redução mensurável na perda de clientes.

### Trade-off Central

| Tipo de Erro | O que acontece | Impacto |
|---|---|---|
| **Falso Negativo** | Cliente ia cancelar, modelo não detectou | Perda de receita recorrente |
| **Falso Positivo** | Cliente não ia cancelar, modelo sinalizou como risco | Custo desnecessário de campanha de retenção |

A calibração desse equilíbrio depende da estratégia da empresa: se o custo de perder um
cliente for muito maior que o custo de uma campanha de retenção, faz sentido priorizar
**recall** (capturar o máximo de churns possível, mesmo com mais falsos alarmes).

## Dataset

**Telco Customer Churn (IBM)** — dataset com informações de clientes de uma empresa de
telecomunicações, incluindo dados demográficos, serviços contratados, informações de conta
e se o cliente cancelou ou não (variável alvo: `Churn`).

## Arquitetura

<!-- TODO: Será detalhado no Estágio 3 -->

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
│   └── processed/       # Dados processados
├── models/              # Modelos treinados serializados
├── notebooks/           # Análises exploratórias e experimentos
├── tests/               # Testes automatizados (pytest)
├── docs/                # Documentação e Model Card
├── pyproject.toml
├── Makefile
└── README.md
```

## Como Usar

### Instalação

<!-- TODO: Será detalhado no Estágio 3 -->

```bash
# Clonar o repositório
git clone <repo-url>
cd telco-churn

# Instalar dependências
pip install -e .
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

## Tecnologias

- **PyTorch** — rede neural (MLP)
- **Scikit-Learn** — preprocessing pipelines e modelos baseline
- **MLflow** — rastreamento de experimentos
- **FastAPI** — API de inferência
- **pytest** — testes automatizados
- **pandera** — validação de schema dos dados
- **ruff** — linting

## Roadmap

- [x] Definição do problema de negócio
- [ ] EDA (Análise Exploratória de Dados)
- [ ] Modelos baseline + MLflow
- [ ] Rede neural (MLP) com PyTorch
- [ ] Refatoração em módulos (`src/`)
- [ ] Pipeline reprodutível
- [ ] API de inferência (FastAPI)
- [ ] Testes automatizados
- [ ] Model Card e documentação final
