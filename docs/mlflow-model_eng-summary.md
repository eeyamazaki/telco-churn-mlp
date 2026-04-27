# MLflow - Model Engineering

## Experimento

`churn-model-engineering`

## Objetivo

Comparar diferentes modelos clássicos de Machine Learning para previsão de churn, avaliando desempenho, parâmetros e capacidade de generalização.

## Modelos avaliados

- Decision Tree
- Random Forest
- SVM (RBF)
- Gradient Boosting
- Gradient Boosting Tunado (modelo final)

## Principais resultados

Os modelos baseados em ensemble apresentaram desempenho superior aos modelos mais simples.

O Gradient Boosting Tunado foi escolhido como modelo final da etapa de Model Engineering por apresentar o melhor equilíbrio entre:

- ROC-AUC
- Precision
- Recall
- F1-score

O SVM também apresentou desempenho competitivo, especialmente em accuracy e F1-score.

## Métricas monitoradas

Os experimentos foram registrados utilizando métricas padronizadas no MLflow:

- `test.accuracy`
- `test.precision`
- `test.recall`
- `test.f1_score`
- `test.roc_auc`
- `test.pr_auc`

## Informações registradas no MLflow

Cada run armazenou:

- hiperparâmetros do modelo
- métricas de avaliação
- dataset utilizado
- feature selection aplicada
- tags descritivas
- descrição do experimento (`mlflow.note.content`)
- artifacts do modelo

## Tags utilizadas

Exemplos de tags registradas:

- `project: telco-churn`
- `experiment_phase: model-engineering`
- `model_type`
- `dataset_name`
- `feature_selection`
- `feature_selection.k`
- `task: binary-classification`
- `target: churn`

## Artifacts registrados

- Matrizes de confusão
- Modelo serializado
- Feature importance (Gini)
- Feature importance (Permutation Importance)
- Métricas
- Parâmetros
- Tags do experimento

## Dataset tracking

Foi utilizado `mlflow.log_input()` para registrar o dataset processado utilizado no treinamento dos modelos.

## Estrutura de documentação

```text
docs/
├── mlflow-artifacts/
│   └── churn-model-engineering/
└── mlflow-screenshots/
    └── churn-model-engineering/