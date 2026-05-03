import logging
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema

from src.config import DATA_RAW, DROP_COLS, TARGET_COLUMN

logger = logging.getLogger(__name__)

_SCHEMA = DataFrameSchema(
    {
        "gender": Column(str, pa.Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, pa.Check.isin([0, 1])),
        "Partner": Column(str, pa.Check.isin(["Yes", "No"])),
        "Dependents": Column(str, pa.Check.isin(["Yes", "No"])),
        "tenure": Column(int, pa.Check.ge(0)),
        "PhoneService": Column(str, pa.Check.isin(["Yes", "No"])),
        "MultipleLines": Column(str),
        "InternetService": Column(str, pa.Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(str),
        "OnlineBackup": Column(str),
        "DeviceProtection": Column(str),
        "TechSupport": Column(str),
        "StreamingTV": Column(str),
        "StreamingMovies": Column(str),
        "Contract": Column(str, pa.Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": Column(str, pa.Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(str),
        "MonthlyCharges": Column(float, pa.Check.ge(0)),
        "TotalCharges": Column(float, pa.Check.ge(0)),
        TARGET_COLUMN: Column(str, pa.Check.isin(["Yes", "No"])),
    },
    coerce=True,
)


def load_raw(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or DATA_RAW / "telco_churn.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {csv_path}. "
            "Execute 'make download-data' para obtê-lo."
        )

    df = pd.read_csv(csv_path)
    logger.info("Carregados %d registros × %d colunas de %s", *df.shape, csv_path.name)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    if n_missing:
        median_val = df["TotalCharges"].median()
        logger.warning(
            "TotalCharges: %d valores ausentes — preenchendo com mediana (%.2f)",
            n_missing,
            median_val,
        )
        df["TotalCharges"] = df["TotalCharges"].fillna(median_val)

    df = df.drop(columns=DROP_COLS, errors="ignore")

    _SCHEMA.validate(df)
    logger.info("Validação de schema concluída — %d registros válidos", len(df))

    return df
