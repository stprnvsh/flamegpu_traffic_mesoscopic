#!/usr/bin/env python3
import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset


# =========================
# Padé optimizers
# =========================

class PadeAdam(Optimizer):
    """
    Adam with Padé-style gradient bounding.
    θ <- θ - lr * m_hat / (sqrt(v_hat) + eps + lam * |m_hat|)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        lam=0.01,
        weight_decay=0.0,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if lam < 0:
            raise ValueError("lam must be >= 0")
        defaults = dict(lr=lr, betas=betas, eps=eps, lam=lam, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, lam, eps = group["lr"], group["lam"], group["eps"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m, v = state["m"], state["v"]
                state["step"] += 1

                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                m_hat = m / bc1
                v_hat = v / bc2

                denom = v_hat.sqrt().add_(eps).add_(lam * m_hat.abs())
                p.addcdiv_(m_hat, denom, value=-lr)

        return loss


class PadeSGD(Optimizer):
    """
    SGD with Padé normalization using a global grad norm per group:
    Δθ = -lr * g / (1 + lam * ||g||)
    """

    def __init__(
        self,
        params,
        lr=0.01,
        momentum=0.0,
        lam=0.1,
        dampening=0.0,
        nesterov=False,
        weight_decay=0.0,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if lam < 0:
            raise ValueError("lam must be >= 0")
        defaults = dict(
            lr=lr, momentum=momentum, lam=lam,
            dampening=dampening, nesterov=nesterov,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, lam = group["lr"], group["lam"]
            momentum, dampening = group["momentum"], group["dampening"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]

            gn2 = 0.0
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)
                gn2 += float(g.norm()) ** 2
            grad_norm = math.sqrt(gn2)
            scale = 1.0 / (1.0 + lam * grad_norm)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                if momentum != 0.0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = g.detach().clone()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(g, alpha=1 - dampening)
                    g = g.add(buf, alpha=momentum) if nesterov else buf

                p.add_(g, alpha=-lr * scale)

        return loss


# =========================
# Utility: metrics
# =========================

def mae(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred - y).abs().mean().item())


def accuracy_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = (logits.sigmoid() >= 0.5).long().squeeze(1)
    return float((pred == y).float().mean().item())


@torch.no_grad()
def auroc_binary(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    AUROC without sklearn. O(n log n). y is 0/1 long tensor (N,)
    """
    scores = logits.sigmoid().squeeze(1)
    y = y.long()

    # sort by score descending
    order = torch.argsort(scores, descending=True)
    y_sorted = y[order]

    P = int(y.sum().item())
    N = int((1 - y).sum().item())
    if P == 0 or N == 0:
        return float("nan")

    tps = torch.cumsum(y_sorted, dim=0)
    fps = torch.cumsum(1 - y_sorted, dim=0)

    tpr = tps / P
    fpr = fps / N

    # trapezoidal integral
    # prepend (0,0)
    z = torch.zeros(1, device=y.device)
    fpr = torch.cat([z, fpr])
    tpr = torch.cat([z, tpr])

    return float(torch.trapz(tpr, fpr).item())


# =========================
# Realistic data generators
# =========================

def make_realistic_regression(
    n: int,
    d: int,
    outlier_frac: float,
    outlier_scale: float,
    seed: int,
    device: torch.device,
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
    """
    Realistic-ish tabular regression:
    - Ill-conditioned features (log-spaced scales)
    - Nonlinear target
    - Heteroscedastic noise
    - Moderate outliers (not insane 25x everywhere)
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    scales = torch.logspace(-1, 1.5, d)  # moderate conditioning spread
    X = X * scales

    # Nonlinear ground truth
    y = (
        1.2 * torch.sin(X[:, : d // 4].sum(dim=1))
        + 0.3 * (X[:, d // 4 : d // 2] ** 2).sum(dim=1)
        - 0.8 * (X[:, d // 2 :].sum(dim=1))
    )

    # Heteroscedastic noise: noise grows with |x|
    noise_std = 0.05 + 0.03 * X.abs().mean(dim=1)
    y = y + noise_std * torch.randn(y.shape, generator=g)
    y = y.unsqueeze(1)

    # Outliers: a small fraction of targets get scaled (common in logs / sensor spikes)
    k = max(1, int(outlier_frac * n))
    idx = torch.randperm(n, generator=g)[:k]
    y[idx] *= outlier_scale

    # Split
    ntr = int(0.8 * n)
    Xtr, ytr = X[:ntr].to(device), y[:ntr].to(device)
    Xte, yte = X[ntr:].to(device), y[ntr:].to(device)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)
    return loader, Xte, yte


def make_realistic_imbalanced_classification(
    n: int,
    d: int,
    pos_frac: float,
    seed: int,
    device: torch.device,
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor, float]:
    """
    Realistic-ish binary classification:
    - Feature scale differences
    - Nonlinear logit
    - Class imbalance (pos_frac)
    Returns loader, Xte, yte, pos_weight for BCEWithLogitsLoss
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    scales = torch.logspace(-1, 2, d)
    X = X * scales

    # Build a nonlinear score; then threshold to enforce imbalance
    raw = (
        0.8 * torch.sin(X[:, : d // 3].sum(dim=1))
        + 0.6 * X[:, d // 3 : 2 * d // 3].mean(dim=1)
        - 0.4 * (X[:, 2 * d // 3 :] ** 2).mean(dim=1)
    )
    raw = raw + 0.2 * torch.randn(raw.shape, generator=g)

    # choose threshold so that about pos_frac are positive
    thr = torch.quantile(raw, 1.0 - pos_frac)
    y = (raw >= thr).long()

    # Split
    ntr = int(0.8 * n)
    Xtr, ytr = X[:ntr].to(device), y[:ntr].to(device)
    Xte, yte = X[ntr:].to(device), y[ntr:].to(device)

    # pos_weight for BCE: Nneg/Npos on training set
    P = float(ytr.sum().item())
    N = float((1 - ytr).sum().item())
    pos_weight = (N / max(P, 1.0))

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=512, shuffle=True)
    return loader, Xte, yte, pos_weight


# =========================
# Models
# =========================

def mlp_reg(d_in: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_in, 256),
        nn.GELU(),
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Linear(128, 1),
    )


def mlp_bin(d_in: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_in, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )


# =========================
# Training loop
# =========================

@dataclass
class RunResult:
    best_test_primary: float
    final_test_primary: float
    diverged: bool
    seconds: float


def train_eval(
    task_type: str,
    model: nn.Module,
    loader: DataLoader,
    Xte: torch.Tensor,
    yte: torch.Tensor,
    loss_fn: nn.Module,
    test_primary: Callable[[torch.Tensor, torch.Tensor], float],
    optimizer: Optimizer,
    epochs: int,
    grad_clip: Optional[float],
) -> RunResult:
    best = None
    diverged = False
    t0 = time.time()

    for _ in range(epochs):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(Xb)
            loss = loss_fn(out, yb)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if diverged:
            break

        model.eval()
        with torch.no_grad():
            out_te = model(Xte)
            val = test_primary(out_te, yte)
        if best is None:
            best = val
        else:
            # lower is better for regression loss/mae; higher is better for classification accuracy/auroc
            if task_type == "reg":
                best = min(best, val)
            else:
                best = max(best, val)

    seconds = time.time() - t0

    model.eval()
    with torch.no_grad():
        out_te = model(Xte)
        final = test_primary(out_te, yte)

    if best is None:
        best = float("nan")

    return RunResult(best, final, diverged, seconds)


def med(xs: List[float]) -> float:
    xs = sorted(xs)
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--stress_lr", type=float, default=1.0,
                    help="Multiply baseline learning rates (use 2-5 to stress stability).")
    ap.add_argument("--outlier_frac", type=float, default=0.01)
    ap.add_argument("--outlier_scale", type=float, default=8.0)
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not available."

    # Optimizer configs: tuned to be reasonable, plus a stress knob via --stress_lr
    base_adam_lr = 1e-3 * args.stress_lr
    base_sgd_lr = 5e-2 * args.stress_lr

    optimizers = {
        "Adam": lambda params: torch.optim.Adam(params, lr=base_adam_lr),
        "Adam+clip1.0": lambda params: torch.optim.Adam(params, lr=base_adam_lr),
        "PadeAdam(lam=0.01)": lambda params: PadeAdam(params, lr=base_adam_lr, lam=0.01),
        "PadeAdam(lam=0.05)": lambda params: PadeAdam(params, lr=base_adam_lr, lam=0.05),

        "SGD(mom=0.9)": lambda params: torch.optim.SGD(params, lr=base_sgd_lr, momentum=0.9),
        "SGD+clip1.0": lambda params: torch.optim.SGD(params, lr=base_sgd_lr, momentum=0.9),
        "PadeSGD(lam=0.1,mom=0.9)": lambda params: PadeSGD(params, lr=base_sgd_lr, momentum=0.9, lam=0.1),
        "PadeSGD(lam=0.5,mom=0.9)": lambda params: PadeSGD(params, lr=base_sgd_lr, momentum=0.9, lam=0.5),
    }
    clip_map = {
        "Adam": None,
        "Adam+clip1.0": 1.0,
        "PadeAdam(lam=0.01)": None,
        "PadeAdam(lam=0.05)": None,
        "SGD(mom=0.9)": None,
        "SGD+clip1.0": 1.0,
        "PadeSGD(lam=0.1,mom=0.9)": None,
        "PadeSGD(lam=0.5,mom=0.9)": None,
    }

    # -------------------------
    # Task 1: Robust regression (Huber)
    # -------------------------
    print("\n" + "=" * 80)
    print("Realistic Regression (Huber) — report: TEST MAE (lower is better)")
    print("=" * 80)

    reg_results: Dict[str, List[RunResult]] = {k: [] for k in optimizers}

    for opt_name, opt_fn in optimizers.items():
        for s in range(args.seeds):
            torch.manual_seed(1000 + s)
            model = mlp_reg(64).to(device)
            loader, Xte, yte = make_realistic_regression(
                n=20000, d=64,
                outlier_frac=args.outlier_frac,
                outlier_scale=args.outlier_scale,
                seed=2000 + s,
                device=device,
            )
            # Huber training loss (robust), MAE evaluation (interpretable)
            huber = nn.SmoothL1Loss(beta=1.0)
            opt = opt_fn(model.parameters())
            clip = clip_map[opt_name]

            res = train_eval(
                task_type="reg",
                model=model,
                loader=loader,
                Xte=Xte,
                yte=yte,
                loss_fn=huber,
                test_primary=lambda out, y: mae(out, y),
                optimizer=opt,
                epochs=args.epochs,
                grad_clip=clip,
            )
            reg_results[opt_name].append(res)

        bests = [r.best_test_primary for r in reg_results[opt_name]]
        finals = [r.final_test_primary for r in reg_results[opt_name]]
        div = sum(int(r.diverged) for r in reg_results[opt_name])
        sec = sum(r.seconds for r in reg_results[opt_name]) / len(reg_results[opt_name])

        print(f"{opt_name:26s}  median(best_MAE)={med(bests):.4f}  median(final_MAE)={med(finals):.4f}  diverged={div}/{args.seeds}  avg_sec={sec:.1f}")

    # -------------------------
    # Task 2: Imbalanced classification (BCE + AUROC)
    # -------------------------
    print("\n" + "=" * 80)
    print("Realistic Classification (Imbalanced) — report: TEST AUROC (higher is better)")
    print("=" * 80)

    cls_results: Dict[str, List[RunResult]] = {k: [] for k in optimizers}

    for opt_name, opt_fn in optimizers.items():
        for s in range(args.seeds):
            torch.manual_seed(3000 + s)
            model = mlp_bin(40).to(device)
            loader, Xte, yte, pos_w = make_realistic_imbalanced_classification(
                n=30000, d=40, pos_frac=0.15, seed=4000 + s, device=device
            )
            pos_weight = torch.tensor([pos_w], device=device)
            bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            opt = opt_fn(model.parameters())
            clip = clip_map[opt_name]

            res = train_eval(
                task_type="cls",
                model=model,
                loader=loader,
                Xte=Xte,
                yte=yte.unsqueeze(1).float(),
                loss_fn=bce,
                test_primary=lambda out, y: auroc_binary(out, y.long().squeeze(1)),
                optimizer=opt,
                epochs=args.epochs,
                grad_clip=clip,
            )
            cls_results[opt_name].append(res)

        bests = [r.best_test_primary for r in cls_results[opt_name]]
        finals = [r.final_test_primary for r in cls_results[opt_name]]
        div = sum(int(r.diverged) for r in cls_results[opt_name])
        sec = sum(r.seconds for r in cls_results[opt_name]) / len(cls_results[opt_name])

        print(f"{opt_name:26s}  median(best_AUROC)={med(bests):.4f}  median(final_AUROC)={med(finals):.4f}  diverged={div}/{args.seeds}  avg_sec={sec:.1f}")

    print("\nNotes:")
    print("- Regression uses SmoothL1 (Huber) for training and MAE for test reporting.")
    print("- Classification is imbalanced; AUROC is a meaningful metric here.")
    print("- Use --stress_lr 2 or 3 to see whether Padé stays stable where baseline optimizers degrade.")


if __name__ == "__main__":
    main()
