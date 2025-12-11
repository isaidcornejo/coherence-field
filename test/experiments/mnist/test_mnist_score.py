import torch
from torch import nn

from src.experiments.mnist.score import compute_scores
from src.experiments.mnist.model import MLP


def test_compute_scores_shape_and_type():
    """
    The compute_scores() function must return a *flattened* 1D tensor containing
    the full gradient of the loss with respect to all model parameters.

    For a model with parameter tensors {θ₁, θ₂, …}, the returned gradient vector:

        g = ∂L/∂θ   reshaped into a single 1D tensor

    must satisfy:

        dim(g) = 1
        g.numel() = Σ_i numel(θᵢ)

    This test ensures that the score representation is compatible with:
        • empirical Fisher computation,
        • covariance estimation,
        • spectral analysis of sensitivity operators.
    """
    model = MLP()
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))

    g = compute_scores(model, loss_fn, x, y)

    # Vector must be 1D
    assert g.dim() == 1

    # Length must match total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    assert g.numel() == total_params


def test_compute_scores_gradients_not_zero():
    """
    A valid backward pass must produce nonzero gradients for almost all input
    batches. This test verifies that:

        g = ∂L/∂θ

    contains at least some nonzero entries, confirming:
        • A loss was computed,
        • The computational graph was constructed correctly,
        • Backpropagation propagated through all differentiable layers.
    """
    model = MLP()
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))

    g = compute_scores(model, loss_fn, x, y)

    assert torch.sum(torch.abs(g)) > 0


def test_compute_scores_repeatability_on_same_input():
    """
    With the same model state, same loss function, and identical inputs, the
    score vector must be deterministic. That is:

        compute_scores(model, x, y)  must produce identical results
        on repeated calls *without modifying the model parameters*.

    This guarantees functional determinism required for:
        • reproducible experiments,
        • deterministic testing,
        • covariance estimation that assumes consistent score evaluations.
    """
    model = MLP()
    loss_fn = nn.CrossEntropyLoss()

    torch.manual_seed(0)
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))

    g1 = compute_scores(model, loss_fn, x, y)
    g2 = compute_scores(model, loss_fn, x, y)

    assert torch.allclose(g1, g2)


def test_compute_scores_different_batch_produces_different_gradients():
    """
    For neural networks, the gradient ∂L/∂θ is highly sensitive to the input batch
    and labels. Therefore, different mini-batches should produce distinct score
    vectors almost always:

        compute_scores(model, x1, y1) ≠ compute_scores(model, x2, y2)

    This property is essential for:
        • empirical Fisher computations,
        • sensitivity-spectrum estimation,
        • alignment diagnostics based on the distribution of score vectors.
    """
    model = MLP()
    loss_fn = nn.CrossEntropyLoss()

    x1 = torch.randn(8, 1, 28, 28)
    y1 = torch.randint(0, 10, (8,))
    g1 = compute_scores(model, loss_fn, x1, y1)

    x2 = torch.randn(8, 1, 28, 28)
    y2 = torch.randint(0, 10, (8,))
    g2 = compute_scores(model, loss_fn, x2, y2)

    # Distinct batches should yield non-identical gradient signatures
    assert not torch.allclose(g1, g2)
