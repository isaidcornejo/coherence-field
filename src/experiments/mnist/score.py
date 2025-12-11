import torch


def compute_scores(model, loss_fn, x, y):
    """
    Compute a *flattened gradient vector* for a single batch.

    This function is the core of the Fisher and empirical alignment
    machinery in MNIST experiments. It extracts the gradient of the loss
    with respect to *all model parameters*, concatenated into a single
    1D vector g ∈ R^D.

    Pipeline:
        1. Zero gradients to avoid accumulation.
        2. Compute forward pass → logits.
        3. Compute scalar loss.
        4. Backpropagate to obtain ∂loss/∂θ for each parameter θ.
        5. Flatten each parameter's gradient and concatenate them.

    Args:
        model (torch.nn.Module):
            Neural network whose gradients define the score vector.
        loss_fn (callable):
            Differentiable loss function such as CrossEntropyLoss.
        x (Tensor):
            Input batch, moved to the correct device beforehand.
        y (Tensor):
            Ground-truth labels for the batch.
    
    Returns:
        Tensor: Flattened gradient vector of shape (D,), where
                D = total number of trainable parameters.

    Notes:
        - The function *detaches* the output to ensure no autograd
          graph is carried forward (we don't want higher-order derivatives).
        - `clone()` ensures returning a clean tensor unaffected by future ops.
        - If a parameter does not have a gradient (rare), it is skipped.
    """

    # Reset gradients before computing new ones
    model.zero_grad(set_to_none=True)

    # Forward + loss
    logits = model(x)
    loss = loss_fn(logits, y)

    # Compute backward pass
    loss.backward()

    # Flatten all parameter gradients into a single vector
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))

    g = torch.cat(grads, dim=0)

    # Detach and clone ensure independence from autograd graph
    return g.detach().clone()
