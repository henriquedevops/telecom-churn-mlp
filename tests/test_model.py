import torch

from src.models import INPUT_DIM, ChurnMLP, build_model


def test_build_model_returns_churn_mlp():
    assert isinstance(build_model(), ChurnMLP)


def test_forward_output_shape_batch():
    model = build_model()
    model.eval()
    x = torch.randn(16, INPUT_DIM)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (16,)


def test_forward_single_sample_probability_range():
    model = build_model()
    model.eval()
    x = torch.randn(1, INPUT_DIM)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    assert 0.0 <= prob <= 1.0


def test_custom_architecture_forward():
    model = build_model(hidden_dims=[32, 16], dropout=0.1)
    model.eval()
    x = torch.randn(8, INPUT_DIM)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8,)
