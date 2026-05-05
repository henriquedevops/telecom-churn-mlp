import logging

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.config import MODELS_DIR, RANDOM_SEED
from src.models.mlp import ChurnMLP, build_model

logger = logging.getLogger(__name__)

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
EXPERIMENT_NAME = "telecom-churn-mlp"


class ChurnDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def _train_epoch(
    model: ChurnMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def _eval_epoch(
    model: ChurnMLP,
    loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            total_loss += criterion(logits, y_batch).item() * len(y_batch)
            all_logits.append(logits.cpu())
            all_labels.append(y_batch.cpu())

    y_prob = torch.sigmoid(torch.cat(all_logits)).numpy()
    y_true = torch.cat(all_labels).numpy().astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "loss": total_loss / len(loader.dataset),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.3,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    run_name: str = "mlp_v1",
) -> dict:
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo: %s", device)

    if hidden_dims is None:
        hidden_dims = [128, 64]

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

    train_loader = DataLoader(
        ChurnDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(ChurnDataset(X_val, y_val), batch_size=batch_size)

    model = build_model(hidden_dims=hidden_dims, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    best_weights: dict = {}
    best_epoch = 0
    no_improve = 0
    history: list[dict] = []

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "hidden_dims": str(hidden_dims),
                "dropout": dropout,
                "lr": lr,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "patience": patience,
                "pos_weight": round(pos_weight.item(), 4),
                "random_seed": RANDOM_SEED,
            }
        )

        for epoch in range(1, max_epochs + 1):
            train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
            val = _eval_epoch(model, val_loader, criterion, device)
            history.append({"epoch": epoch, "train_loss": train_loss, **val})

            mlflow.log_metrics(
                {
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val["loss"], 4),
                    "val_auc_roc": round(val["auc_roc"], 4),
                    "val_pr_auc": round(val["pr_auc"], 4),
                    "val_f1": round(val["f1"], 4),
                },
                step=epoch,
            )

            if val["loss"] < best_val_loss:
                best_val_loss = val["loss"]
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or no_improve == 0:
                logger.info(
                    "Epoch %3d | train: %.4f | val_loss: %.4f | val_auc: %.4f | val_f1: %.4f",
                    epoch,
                    train_loss,
                    val["loss"],
                    val["auc_roc"],
                    val["f1"],
                )

            if no_improve >= patience:
                logger.info("Early stopping na epoch %d (patience=%d)", epoch, patience)
                break

        model.load_state_dict(best_weights)
        mlflow.log_metric("best_epoch", best_epoch)

        model_path = MODELS_DIR / "churn_mlp.pt"
        torch.save(model.state_dict(), model_path)
        logger.info("Melhor modelo (epoch %d) salvo em %s", best_epoch, model_path)

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_metrics": history[best_epoch - 1],
    }
