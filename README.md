# telecom-churn-mlp

Predição de churn em telecom com rede neural MLP, pipeline sklearn e API de inferência com FastAPI.

## Setup

```bash
pip install -e ".[dev]"
```

## Comandos

```bash
make lint      # verifica estilo do código
make test      # executa a suite de testes
make run       # sobe a API localmente na porta 8000
```

## Estrutura

```
src/           código-fonte principal
tests/         testes automatizados
notebooks/     análise exploratória e experimentos
data/          dados brutos e processados (não versionados)
models/        artefatos de modelo (não versionados)
docs/          documentação e model card
```

## Dataset

Telco Customer Churn — IBM Sample Dataset

```bash
make download-data
```

Requer a [Kaggle CLI](https://github.com/Kaggle/kaggle-api) configurada (`~/.kaggle/kaggle.json`).  
O arquivo é baixado e renomeado para `data/raw/telco_churn.csv` automaticamente.
