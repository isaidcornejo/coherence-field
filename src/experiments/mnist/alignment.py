import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import build_mnist_model
from .score import compute_scores

from src.utils.alignment_core import compute_alignment_operator


# -------------------------------------------------------------
# Dataloaders
# -------------------------------------------------------------
def _get_dataloaders(batch_size, seed):
    torch.manual_seed(seed)
    transform = transforms.ToTensor()

    train_ds = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=True
    )
    return train_loader, test_loader


# -------------------------------------------------------------
# Main MNIST Alignment Experiment
# -------------------------------------------------------------
def run_mnist_alignment(
    batch_size=128,
    num_batches_train=200,
    num_batches_eval=100,
    lr=1e-2,
    seed=123,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    train_loader, test_loader = _get_dataloaders(batch_size, seed)

    # Build model
    model = build_mnist_model(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # ---------------------------------------------------------
    # TRAINING PHASE
    # ---------------------------------------------------------
    train_losses = []
    model.train()
    train_iter = iter(train_loader)

    for step in range(num_batches_train):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        train_losses.append(float(loss.item()))

    # ---------------------------------------------------------
    # TEST PHASE (Loss curve only)
    # ---------------------------------------------------------
    test_losses = []
    model.eval()
    test_iter = iter(test_loader)
    with torch.no_grad():
        for step in range(num_batches_eval):
            try:
                x, y = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                x, y = next(test_iter)

            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            test_losses.append(float(loss.item()))

    # ---------------------------------------------------------
    # ESTIMATE FISHER MATRIX G (Model distribution p)
    # ---------------------------------------------------------
    model.train()  
    train_loader_G, _ = _get_dataloaders(batch_size, seed + 1)
    train_iter_G = iter(train_loader_G)

    # First gradient
    x0, y0 = next(train_iter_G)
    x0 = x0.to(device)
    y0 = y0.to(device)
    g0 = compute_scores(model, loss_fn, x0, y0)

    g0_np = g0.detach().cpu().numpy().astype(np.float64)
    D = g0_np.shape[0]

    G = np.zeros((D, D), dtype=np.float64)
    count_G = 0

    G += np.outer(g0_np, g0_np)
    count_G += 1

    for _ in range(num_batches_eval - 1):
        try:
            x, y = next(train_iter_G)
        except StopIteration:
            train_iter_G = iter(train_loader_G)
            x, y = next(train_iter_G)

        x = x.to(device)
        y = y.to(device)
        g = compute_scores(model, loss_fn, x, y)
        g_np = g.detach().cpu().numpy().astype(np.float64)

        G += np.outer(g_np, g_np)
        count_G += 1

    G /= float(count_G)

    # ---------------------------------------------------------
    # ESTIMATE EMPIRICAL COVARIANCE C (Empirical distribution q)
    # ---------------------------------------------------------
    _, test_loader_C = _get_dataloaders(batch_size, seed + 2)
    test_iter_C = iter(test_loader_C)

    C = np.zeros((D, D), dtype=np.float64)
    count_C = 0

    for _ in range(num_batches_eval):
        try:
            x, y = next(test_iter_C)
        except StopIteration:
            test_iter_C = iter(test_loader_C)
            x, y = next(test_iter_C)

        x = x.to(device)
        y = y.to(device)
        g = compute_scores(model, loss_fn, x, y)
        g_np = g.detach().cpu().numpy().astype(np.float64)

        C += np.outer(g_np, g_np)
        count_C += 1

    C /= float(count_C)

    # ---------------------------------------------------------
    # STABILIZE & ALIGN (High-dimensional H)
    # ---------------------------------------------------------
    # Symmetrize
    G = 0.5 * (G + G.T)
    C = 0.5 * (C + C.T)

    # Compute Fisher-normalized alignment operator
    H = compute_alignment_operator(G, C, eps=1e-3)

    # Stable eigenvalues
    eigvals = np.linalg.eigvalsh(H)

    # Scalar invariant A = Σ(λ - 1)
    A_q = float(np.sum(eigvals - 1.0))

    # Rectified amplitude φ
    phi_q = float(np.sqrt(A_q) if A_q > 0 else 0.0)

    # ---------------------------------------------------------
    # Return experiment results
    # ---------------------------------------------------------
    return {
        "G": G,
        "C": C,
        "H": H,
        "lambdas": eigvals,
        "A": A_q,
        "phi": phi_q,
        "train_loss_curve": np.array(train_losses, dtype=np.float32),
        "test_loss_curve": np.array(test_losses, dtype=np.float32),
        "batch_size": batch_size,
        "num_batches_train": num_batches_train,
        "num_batches_eval": num_batches_eval,
        "lr": lr,
        "seed": seed,
    }
