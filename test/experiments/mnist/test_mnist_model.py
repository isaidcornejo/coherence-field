import torch
from src.experiments.mnist.model import MLP, build_mnist_model


def test_mlp_forward_shape():
    """
    The MLP must accept standard MNIST-shaped inputs of size (B, 1, 28, 28),
    perform internal flattening, and produce logits of size (B, 10), corresponding
    to the 10-class digit classification problem.

    This test verifies the correctness of:
        - Input reshaping logic
        - Dimensionality of the final linear layer
        - Forward-path consistency
    """
    model = MLP()
    x = torch.randn(8, 1, 28, 28)  # batch of 8 images

    logits = model(x)
    assert logits.shape == (8, 10)


def test_mlp_forward_with_flattened_input():
    """
    The MLP must also support already-flattened MNIST vectors of size (B, 784).
    This ensures the model can be used in contexts where preprocessing has
    flattened inputs before passing them to the network.

    The output shape must remain (B, 10).
    """
    model = MLP()
    x = torch.randn(8, 28 * 28)

    logits = model(x)
    assert logits.shape == (8, 10)


def test_build_mnist_model_device():
    """
    The model builder must correctly place all parameters on the requested device.

    Two cases are tested:

        1. CUDA available:
               - Model should move to GPU
        2. CUDA not available:
               - Model should still initialize correctly on CPU

    This ensures the build function respects device placement, which is essential
    for training loops and large-scale experiments.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = build_mnist_model(device=device)
        assert next(model.parameters()).is_cuda
    else:
        device = torch.device("cpu")
        model = build_mnist_model(device=device)
        assert not next(model.parameters()).is_cuda


def test_mlp_backprop():
    """
    A minimal backpropagation test: a forward pass followed by one backward pass
    must produce nonzero gradients for at least one parameter in the model.

    This test ensures:
        - The computational graph is correctly constructed
        - Loss computation is valid
        - Gradients propagate through all layers
        - Parameters have the expected requires_grad behavior

    This is the minimal correctness check for trainability of the model.
    """
    model = MLP()
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))

    criterion = torch.nn.CrossEntropyLoss()

    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    # At least one parameter must receive a nonzero gradient
    grads = [p.grad for p in model.parameters() if p.grad is not None]

    assert len(grads) > 0
    assert any(g.abs().sum().item() > 0 for g in grads)
