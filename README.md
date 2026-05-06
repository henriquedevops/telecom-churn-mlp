# telecom-churn-mlp

Pipeline end-to-end de machine learning para predição de churn em telecom, com rede neural MLP (PyTorch), API de inferência (FastAPI) e rastreamento de experimentos (MLflow).

## Visão Geral

| Item | Detalhe |
|---|---|
| Problema | Classificação binária — predizer cancelamento de cliente |
| Dataset | Telco Customer Churn (IBM/Kaggle) — 7.043 registros, 19 features |
| Modelo | MLP: 30 → 128 → 64 → 1 (BatchNorm + ReLU + Dropout 0.3) |
| Threshold ótimo | 0.34 (minimização de custo FP/FN) |
| AUC-ROC (teste holdout) | 0.846 |
| Recall churn (threshold 0.34) | 95.4% |

---

## Setup

### Pré-requisitos

- Python 3.10+
- [Kaggle CLI](https://github.com/Kaggle/kaggle-api) configurada em `~/.kaggle/kaggle.json`

### Instalação

```bash
pip install -e ".[dev]"
```

### Dataset

```bash
make download-data
```

Ou baixar manualmente em [kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) e salvar como `data/raw/telco_churn.csv`.

---

## Reprodução dos Experimentos

Execute os notebooks **em ordem** no diretório `notebooks/`:

| Notebook | Descrição |
|---|---|
| `01_eda.ipynb` | Análise exploratória e qualidade dos dados |
| `02_baselines.ipynb` | DummyClassifier + LogisticRegression com MLflow |
| `03_training.ipynb` | Treinamento do MLP com early stopping |
| `04_model_comparison.ipynb` | Comparação de modelos e análise de threshold |

Os artefatos treinados são salvos em `models/` (não versionados).

---

## API de Inferência

### Iniciar o servidor

```bash
make run
```

Requer que `models/churn_mlp.pt` e `models/preprocessor.joblib` existam (gerados pelo `03_training.ipynb`).

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Status do servidor e verificação do modelo |
| `POST` | `/predict` | Predição de churn para um cliente |
| `GET` | `/docs` | Swagger UI com schema e exemplo |

### Exemplo

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "SeniorCitizen": 0, "tenure": 12, "MonthlyCharges": 65.0, "TotalCharges": 780.0,
    "gender": "Female", "Partner": "Yes", "Dependents": "No", "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check"
  }'
```

```json
{
  "churn_probability": 0.7432,
  "churn_prediction": 1,
  "threshold": 0.34
}
```

---

## Testes

```bash
make test
```

10 testes automatizados cobrindo: arquitetura do modelo, validação de schema (pandera) e endpoints da API.

---

## Comandos Disponíveis

```bash
make install       # instala dependências
make download-data # baixa o dataset via Kaggle CLI
make lint          # verifica estilo com ruff
make format        # formata código com ruff
make test          # executa pytest com cobertura
make run           # sobe a API na porta 8000
```

---

## Estrutura do Projeto

```
src/
├── config.py              seeds, paths, threshold (0.34)
├── data/
│   └── loader.py          carregamento, limpeza e validação pandera
├── features/
│   └── pipeline.py        ColumnTransformer (StandardScaler + OHE)
├── models/
│   ├── mlp.py             ChurnMLP — arquitetura PyTorch
│   ├── trainer.py         loop de treinamento com early stopping
│   ├── baseline.py        DummyClassifier e LogisticRegression
│   └── predictor.py       load_artifacts() + predict() para a API
└── api/
    ├── app.py             FastAPI — lifespan, /health, /predict
    └── schemas.py         Pydantic — CustomerFeatures, PredictionResponse
tests/
├── test_model.py          smoke tests da rede neural
├── test_schema.py         validação pandera com dados sintéticos
└── test_api.py            testes de integração dos endpoints
notebooks/
├── 01_eda.ipynb
├── 02_baselines.ipynb
├── 03_training.ipynb
└── 04_model_comparison.ipynb
docs/
└── model_card.md          performance, limitações, vieses e monitoramento
```

---

## API em Produção

Endpoint público hospedado no **GCP Cloud Run**:

| | URL |
|---|---|
| Health | `GET https://telecom-churn-mlp-1016158003629.us-central1.run.app/health` |
| Predição | `POST https://telecom-churn-mlp-1016158003629.us-central1.run.app/predict` |
| Docs (Swagger) | `https://telecom-churn-mlp-1016158003629.us-central1.run.app/docs` |

---

## Resultados

| Modelo | AUC-ROC | PR-AUC | F1 | Accuracy |
|---|---|---|---|---|
| DummyClassifier | 0.500 | 0.265 | 0.000 | 0.735 |
| Logistic Regression | 0.850 | 0.637 | 0.624 | 0.740 |
| **MLP (PyTorch)** | **0.846** | **0.635** | **0.617** | **0.727** |

Com threshold 0.34 (otimizado por custo de negócio): recall de churn **95.4%**, custo estimado **11.5% menor** que threshold padrão 0.5.

Ver [`docs/model_card.md`](docs/model_card.md) para análise completa de limitações, vieses e plano de monitoramento.
