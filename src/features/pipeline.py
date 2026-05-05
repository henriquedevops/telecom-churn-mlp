import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLS,
    MODELS_DIR,
    NUMERIC_COLS,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)

_NUMERIC_ALL = NUMERIC_COLS + ["SeniorCitizen"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), _NUMERIC_ALL),
            (
                "cat",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
    )


def encode_target(series: pd.Series) -> np.ndarray:
    return (series == "Yes").astype(int).values


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = encode_target(df[TARGET_COLUMN])

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    logger.info(
        "Split — treino/val: %d registros | teste holdout: %d registros (%.0f%%)",
        len(X_train_val),
        len(X_test),
        TEST_SIZE * 100,
    )
    return X_train_val, X_test, y_train_val, y_test


def fit_preprocessor(X_train: pd.DataFrame, save: bool = True) -> ColumnTransformer:
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    logger.info(
        "Preprocessador ajustado — %d features de saída",
        preprocessor.transform(X_train).shape[1],
    )

    if save:
        path = MODELS_DIR / "preprocessor.joblib"
        joblib.dump(preprocessor, path)
        logger.info("Preprocessador salvo em %s", path)

    return preprocessor
