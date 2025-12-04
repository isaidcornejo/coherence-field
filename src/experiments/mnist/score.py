import torch


def compute_scores(model, loss_fn, x, y):
    """
    Devuelve un solo vector de gradientes aplanado para el batch.
    """
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()

    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    g = torch.cat(grads, dim=0)
    return g.detach().clone()
