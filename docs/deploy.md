# Deploy em GCP Cloud Run

Este guia descreve como construir a imagem Docker e publicar a API de inferência no Google Cloud Run.

## Pré-requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e rodando
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) instalado
- Conta no Google Cloud com faturamento ativo (o Cloud Run tem free tier generoso: 2 M req/mês)
- Artefatos de modelo presentes localmente: `models/churn_mlp.pt` e `models/preprocessor.joblib`
  - Se não tiver, execute os notebooks `03_training.ipynb` e `04_model_comparison.ipynb` primeiro

---

## Passo 1 — Configurar o gcloud

```bash
gcloud auth login
gcloud config set project SEU_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

Para descobrir ou criar um projeto:
```bash
gcloud projects list
gcloud projects create telecom-churn-mlp --name="Telecom Churn MLP"
```

---

## Passo 2 — Testar a imagem localmente

Antes de fazer o deploy, confirme que a imagem funciona:

```bash
# Construir
make docker-build
# ou: docker build -t telecom-churn-mlp .

# Rodar localmente
make docker-run
# ou: docker run --rm -p 8000:8000 telecom-churn-mlp

# Testar (em outro terminal)
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "SeniorCitizen": 0, "tenure": 12, "MonthlyCharges": 65.0,
    "TotalCharges": 780.0, "gender": "Female", "Partner": "Yes",
    "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
  }'
```

---

## Passo 3 — Fazer o push para o Google Container Registry

```bash
# Autenticar o Docker com o GCP
gcloud auth configure-docker

# Tag e push (substitua SEU_PROJECT_ID pelo ID real)
GCP_PROJECT=SEU_PROJECT_ID make docker-push
# ou manualmente:
docker tag telecom-churn-mlp gcr.io/SEU_PROJECT_ID/telecom-churn-mlp
docker push gcr.io/SEU_PROJECT_ID/telecom-churn-mlp
```

---

## Passo 4 — Deploy no Cloud Run

```bash
gcloud run deploy telecom-churn-mlp \
  --image gcr.io/SEU_PROJECT_ID/telecom-churn-mlp \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8000 \
  --memory 1Gi \
  --cpu 1
```

O comando retorna uma URL pública no formato:
```
https://telecom-churn-mlp-xxxxxxxxxxxx-uc.a.run.app
```

---

## Passo 5 — Verificar o deploy

```bash
# Substitua pela URL real retornada pelo gcloud
URL=https://telecom-churn-mlp-xxxxxxxxxxxx-uc.a.run.app

curl $URL/health

curl -X POST $URL/predict \
  -H "Content-Type: application/json" \
  -d '{"SeniorCitizen": 0, "tenure": 12, "MonthlyCharges": 65.0,
       "TotalCharges": 780.0, "gender": "Female", "Partner": "Yes",
       "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
       "InternetService": "Fiber optic", "OnlineSecurity": "No",
       "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
       "StreamingTV": "No", "StreamingMovies": "No",
       "Contract": "Month-to-month", "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check"}'
```

Acesse também a documentação interativa em `$URL/docs`.

---

## Comandos alternativos com Cloud Build (sem Docker local)

Se preferir não instalar Docker, o Cloud Build faz a construção na nuvem:

```bash
gcloud builds submit --tag gcr.io/SEU_PROJECT_ID/telecom-churn-mlp .
gcloud run deploy telecom-churn-mlp \
  --image gcr.io/SEU_PROJECT_ID/telecom-churn-mlp \
  --region us-central1 --platform managed --allow-unauthenticated --port 8000
```

---

## Free Tier do Cloud Run

| Recurso | Free Tier (por mês) |
|---|---|
| Requisições | 2.000.000 |
| Tempo de CPU | 360.000 vCPU-segundos |
| Memória | 180.000 GB-segundos |

O serviço **escala para zero** quando não há requisições — sem custo em repouso.

---

## Incluir a URL no README

Após o deploy, adicione a URL pública no `README.md`:

```markdown
## API em Produção

Endpoint público (GCP Cloud Run):
- `GET  https://telecom-churn-mlp-xxxx-uc.a.run.app/health`
- `POST https://telecom-churn-mlp-xxxx-uc.a.run.app/predict`
- Documentação: `https://telecom-churn-mlp-xxxx-uc.a.run.app/docs`
```
