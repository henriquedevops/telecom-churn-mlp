import torch
import torch.nn as nn

# 4 numéricas (StandardScaler) + 26 categóricas (OHE drop='first')
INPUT_DIM = 30


class ChurnMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def build_model(
    input_dim: int = INPUT_DIM,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.3,
) -> ChurnMLP:
    return ChurnMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
