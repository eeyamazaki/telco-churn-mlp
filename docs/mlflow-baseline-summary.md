# MLflow - Baseline

## Experimento

`churn-baseline`

## Objetivo

Registrar os primeiros modelos baseline para previsão de churn.

## Modelos avaliados

- DummyClassifier
- Logistic Regression

## Principais resultados

A Regressão Logística apresentou desempenho significativamente superior ao DummyClassifier em todas as métricas principais:

- ROC-AUC
- Recall
- Precision
- F1-score

## Artifacts registrados

- Modelo serializado
- Matrizes de confusão
- Métricas
- Parâmetros
- Tags do experimento

## Observações

O DummyClassifier foi utilizado apenas como referência mínima para comparação dos modelos posteriores.



