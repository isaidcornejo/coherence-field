import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import build_mnist_model
from .score import compute_scores
from src.utils.alignment_core import compute_alignment_operator


# =============================================================
#  DATALOADER CONSTRUCTION
# =============================================================

def _get_dataloaders(batch_size, seed):
    """
    Construct train/test dataloaders for MNIST using consistent seeding.

    Args:
        batch_size (int): Batch size for loaders.
        seed (int): Seed to ensure consistent shuffling.

    Returns:
        (train_loader, test_loader)
    """
    torch.manual_seed(seed)
    transform = transforms.ToTensor()

    # Download MNIST if missing
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


# =============================================================
#  MAIN MNIST ALIGNMENT EXPERIMENT
# =============================================================

def run_mnist_alignment(
    batch_size=128,
    num_batches_train=200,
    num_batches_eval=100,
    lr=1e-2,
    seed=123,
):
    """
    Run the full MNIST Fisher–Empirical alignment pipeline.

    This experiment moves beyond analytical models and applies your
    alignment diagnostics to a *high-dimensional neural network model*.

    Pipeline:
        1. Load MNIST dataset and construct dataloaders.
        2. Build a neural classifier (simple CNN).
        3. Train the classifier for `num_batches_train` gradient steps.
        4. Track training and test loss curves.
        5. Estimate Fisher matrix G from model gradients under p(x|θ).
        6. Estimate empirical covariance C from gradients under q(x).
        7. Compute alignment operator H = G^{-1/2} C G^{-1/2}.
        8. Extract eigenvalues λ_i, scalar invariant A = Σ(λ - 1),
           and rectified amplitude φ.

    Notes:
        • This is a *stochastic*, *GPU-dependent* experiment.
        • It is **not** appropriate for unit tests.
        • The outputs are intended for scientific figures and notebooks.

    Returns:
        dict with:
            - G, C, H: alignment objects
            - lambdas: eigenvalues of H
            - A, phi: scalar diagnostics
            - train_loss_curve, test_loss_curve: monitoring
            - experiment settings
    """

    # ---------------------------------------------------------
    # DEVICE + SEEDS
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---------------------------------------------------------
    # DATALOADERS
    # ---------------------------------------------------------
    train_loader, test_loader = _get_dataloaders(batch_size, seed)

    # ---------------------------------------------------------
    # BUILD MODEL & OPTIMIZER
    # ---------------------------------------------------------
    model = build_mnist_model(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # =========================================================
    #  TRAINING PHASE
    # =========================================================
    train_losses = []
    model.train()
    train_iter = iter(train_loader)

    for step in range(num_batches_train):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        train_losses.append(float(loss.item()))

    # =========================================================
    #  TEST PHASE (Loss only)
    # =========================================================
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

            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            test_losses.append(float(loss.item()))

    # =========================================================
    #  FISHER ESTIMATION (Model distribution p)
    # =========================================================
    model.train()
    train_loader_G, _ = _get_dataloaders(batch_size, seed + 1)
    train_iter_G = iter(train_loader_G)

    # First gradient defines dimensionality
    x0, y0 = next(train_iter_G)
    x0, y0 = x0.to(device), y0.to(device)
    g0 = compute_scores(model, loss_fn, x0, y0)

    g0_np = g0.detach().cpu().numpy().astype(np.float64)
    D = g0_np.shape[0]              # dimension of gradient vector

    G = np.zeros((D, D), dtype=np.float64)
    count_G = 0

    # First outer product
    G += np.outer(g0_np, g0_np)
    count_G += 1

    # Next batches
    for _ in range(num_batches_eval - 1):
        try:
            x, y = next(train_iter_G)
        except StopIteration:
            train_iter_G = iter(train_loader_G)
            x, y = next(train_iter_G)

        x, y = x.to(device), y.to(device)
        g = compute_scores(model, loss_fn, x, y)
        g_np = g.detach().cpu().numpy().astype(np.float64)

        G += np.outer(g_np, g_np)
        count_G += 1

    G /= float(count_G)

    # =========================================================
    #  EMPIRICAL COVARIANCE (Empirical distribution q)
    # =========================================================
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

        x, y = x.to(device), y.to(device)
        g = compute_scores(model, loss_fn, x, y)
        g_np = g.detach().cpu().numpy().astype(np.float64)

        C += np.outer(g_np, g_np)
        count_C += 1

    C /= float(count_C)

    # =========================================================
    #  ALIGNMENT OPERATOR (High-dimensional)
    # =========================================================
    G = 0.5 * (G + G.T)
    C = 0.5 * (C + C.T)

    H = compute_alignment_operator(G, C, eps=1e-3)
    eigvals = np.linalg.eigvalsh(H)

    A_q = float(np.sum(eigvals - 1.0))
    phi_q = float(np.sqrt(A_q) if A_q > 0 else 0.0)

    # =========================================================
    #  OUTPUT PACKAGE
    # =========================================================
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
