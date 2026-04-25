# Pipeline MLP — Documentação de Mudanças

> Refatoração completa do pipeline para MLP (PyTorch), criação da API, testes e ajustes de qualidade de código.

---

## 1. O que foi feito

### 1.1 Módulos do pipeline (`src/`)

| Módulo | Arquivo | Descrição |
|---|---|---|
| Arquitetura do modelo | `src/models/mlp.py` | `ChurnMLP` — MLP configurável em PyTorch com BatchNorm, ReLU e Dropout |
| Feature engineering | `src/features/engineer.py` | `FeatureEngineer` — transformador sklearn que cria 6 features derivadas |
| Inferência | `src/inference/predictor.py` | `ChurnPredictor` — carrega artefatos e executa o pipeline completo |
| API | `src/api/main.py` | FastAPI com `/health`, `/predict`, middleware de latência e handlers de erro |
| Treinamento | `src/training/train.py` | Script reprodutível com MLflow, early stopping e salvamento de artefatos |
| Logger | `src/logger.py` | Logging estruturado com `structlog`, corrigido para ser idempotente |
| Config | `src/config.py` | Constantes centralizadas — corrigidos `TARGET_COLUMN`, `MAX_EPOCHS`, `MLFLOW_EXPERIMENT_NAME` |

### 1.2 Schemas de validação

| Schema | Arquivo | Responsabilidade |
|---|---|---|
| Pandera (DataFrames) | `src/schemas/common.py` | Valida dados processados e engineered — treinamento e inferência batch |
| Pydantic (API entrada) | `src/schemas/input.py` | Valida JSON recebido no `POST /predict` |
| Pydantic (API saída) | `src/schemas/output.py` | Define estrutura do response JSON |

### 1.3 Testes

| Arquivo | Cobertura |
|---|---|
| `tests/unit/test_mlp.py` | Arquitetura, shapes, logits, determinismo |
| `tests/unit/test_predictor.py` | Pipeline completo, validação de schema, batch |
| `tests/unit/test_feature_engineer.py` | 6 features derivadas, contrato sklearn |
| `tests/unit/schemas/test_common.py` | Schemas Pandera para DataFrames |
| `tests/unit/schemas/test_input.py` | Validação Pydantic de entrada da API |
| `tests/unit/schemas/test_output.py` | Validação Pydantic de saída da API |
| `tests/smoke/test_api.py` | Fluxo completo de ponta a ponta via TestClient |

**Total: 385 testes — todos passando.**

### 1.4 Infraestrutura

- **`Makefile`** — targets `lint`, `lint-fix`, `test`, `train`, `run-api`, `setup` (via `uv sync`)
- **`uv.lock`** — lock file com 223 dependências fixadas com hash para instalação reprodutível
- **`pyproject.toml`** — fonte única de dependências com ruff, pytest e grupos dev configurados

---

## 2. Por que foi feito

### Troca de Gradient Boosting por MLP

O notebook `03_model_engineering.ipynb` avaliou múltiplos modelos sklearn. O Gradient Boosting obteve ROC-AUC de 0.8569 como melhor resultado. O MLP (notebook `04_mlp.ipynb`) superou esse resultado em todas as métricas principais:

| Modelo | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Gradient Boosting (melhor sklearn) | 0.8569 | 0.6685 | 0.5976 |
| **MLP 64→32 (PyTorch)** | **0.8611** | **0.6748** | **0.6537** |

O MLP foi treinado com `pos_weight` para corrigir o desbalanceamento de classes (26% churn) e `early stopping` (paciência de 15 épocas) para evitar overfitting.

### Threshold F1-ótimo (0.5996)

O threshold padrão de 0.5 não é ideal quando se usa `pos_weight` no `BCEWithLogitsLoss` — ele desloca a distribuição de probabilidades. O threshold de 0.5996 foi calculado maximizando o F1-Score na curva Precision-Recall do test set.

Para campanhas de retenção onde **FN custa ~28x mais que FP** (LTV médio ÷ custo de campanha), o threshold pode ser reduzido na inferência para aumentar o recall — ver seção 9 do notebook `04_mlp.ipynb`.

### Refatoração do código para `src/`

Os notebooks são adequados para exploração, mas não para produção. A refatoração garante:
- **Reprodutibilidade**: `make train` treina o modelo do zero com os mesmos resultados
- **Testabilidade**: cada módulo tem testes isolados
- **Manutenibilidade**: separação clara entre dados, features, modelo, inferência e API

### Correções nos schemas

- `tenure_group` retornava dtype `category` do `pd.cut()` — incompatível com `pa.String` do Pandera. Corrigido com `.astype(str)` no `FeatureEngineer`
- `payment_method` tinha nota desatualizada sobre seleção de features do GB

---

## 3. Como utilizar

### Pré-requisitos

```bash
# Clonar o repositório e instalar dependências
git clone <repo>
cd telco-churn-mlp
make setup          # cria .venv e instala via uv.lock
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### Treinar o modelo

```bash
make train
```

Executa `src/training/train.py` que:
1. Carrega `data/processed/telco_churn_cleaned.csv`
2. Aplica `FeatureEngineer` (6 features derivadas)
3. Split estratificado 70/15/15
4. Treina `ChurnMLP` com early stopping
5. Registra experimento no MLflow (`telco-churn-mlp`)
6. Salva artefatos em `models/`:
   - `preprocessor_mlp.pkl` — ColumnTransformer fitado
   - `mlp_weights.pt` — pesos do modelo
   - `mlp_config.json` — arquitetura e threshold F1-ótimo

Para visualizar os experimentos:
```bash
mlflow ui
# Acesse http://localhost:5000
```

### Subir a API

```bash
make run-api
# Acesse http://localhost:8000/docs
```

**Endpoints disponíveis:**

`GET /health` — verifica se o modelo está carregado
```json
{
  "status": "ok",
  "timestamp": "2026-04-24T20:00:00",
  "model_version": "mlp-v1"
}
```

`POST /predict` — retorna probabilidade e predição de churn
```json
// Requisição
{
  "gender": "Female",
  "senior_citizen": "No",
  "partner": "No",
  "dependents": "No",
  "tenure_months": 2,
  "monthly_charges": 95.0,
  "total_charges": 190.0,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "contract": "Month-to-month",
  "internet_service_type": "Fiber optic",
  "online_security": "No",
  "online_backup": "No",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "No",
  "streaming_movies": "No"
}

// Resposta
{
  "success": true,
  "prediction": {
    "churn_probability": 0.8423,
    "prediction": "Churn",
    "threshold_used": 0.5996,
    "confidence": 0.8423
  },
  "timestamp": "2026-04-24T20:00:00",
  "model_version": "mlp-v1",
  "latency_ms": 12.5
}
```

### Rodar os testes

```bash
make test                                    # todos os testes
pytest tests/unit/ -v                        # só unitários
pytest tests/smoke/ -v                       # só smoke (API)
pytest tests/ --cov=src --cov-report=term-missing  # com cobertura
```

### Verificar qualidade de código

```bash
make lint       # verifica — deve retornar "All checks passed!"
make lint-fix   # corrige automaticamente
```

---

## 4. Artefatos gerados pelo treinamento

| Arquivo | Descrição | Usado por |
|---|---|---|
| `models/mlp_config.json` | Arquitetura e threshold F1-ótimo | `predictor.py` na inicialização |
| `models/mlp_weights.pt` | Pesos do modelo treinado | `predictor.py` via `torch.load()` |
| `models/preprocessor_mlp.pkl` | ColumnTransformer fitado no treino | `predictor.py` via `joblib.load()` |

> **Atenção:** esses arquivos não são versionados no git (são artefatos gerados). Para reproduzir, execute `make train`.

---

## 5. Estrutura de pastas relevante

```
src/
├── api/main.py           # FastAPI — endpoints e handlers
├── config.py             # constantes do projeto
├── features/engineer.py  # FeatureEngineer (transformador sklearn)
├── inference/predictor.py# ChurnPredictor — pipeline de inferência
├── logger.py             # logging estruturado (structlog)
├── models/mlp.py         # arquitetura ChurnMLP (PyTorch)
├── schemas/
│   ├── common.py         # schemas Pandera (DataFrames)
│   ├── input.py          # schema Pydantic (entrada da API)
│   └── output.py         # schema Pydantic (saída da API)
└── training/train.py     # script de treinamento reprodutível

tests/
├── smoke/test_api.py     # teste ponta a ponta da API
└── unit/
    ├── schemas/          # testes dos schemas Pandera e Pydantic
    ├── test_feature_engineer.py
    ├── test_mlp.py
    └── test_predictor.py
```
