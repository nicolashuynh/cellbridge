from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import ot
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_coupling_dense(coupling: np.ndarray) -> np.ndarray:
    G = np.asarray(coupling, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError("Dense coupling must be 2D (n x m).")
    if np.any(G < 0):
        logger.warning("Coupling has negative entries; clipping at 0.")
        G = np.clip(G, 0.0, None)
    s = float(G.sum())
    if s <= 0:
        raise ValueError("Coupling weights sum to zero.")
    return G / s


def _sample_edges_from_dense_coupling(
    coupling: np.ndarray,
    *,
    num_edges: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample edge indices from the full dense coupling without top-k sparsification.

    This provides a memory-safe approximation path for dense couplings where
    materializing all non-zero edges (n*m) is impractical.
    """
    if num_edges <= 0:
        raise ValueError(f"num_edges must be > 0, got {num_edges}.")

    G = _normalize_coupling_dense(coupling)
    n, m = G.shape
    row_mass = G.sum(axis=1)
    row_mass = row_mass / max(float(row_mass.sum()), 1e-12)

    rng = np.random.default_rng(seed)
    rows = rng.choice(n, size=num_edges, replace=True, p=row_mass).astype(np.int64)

    order = np.argsort(rows, kind="stable")
    rows_sorted = rows[order]
    cols_sorted = np.empty(num_edges, dtype=np.int64)

    start = 0
    while start < num_edges:
        i = int(rows_sorted[start])
        end = start
        while end < num_edges and int(rows_sorted[end]) == i:
            end += 1
        count = end - start

        probs = G[i]
        mass = float(row_mass[i])
        if mass <= 0:
            cond = np.full(m, 1.0 / m, dtype=np.float64)
        else:
            cond = probs / mass

        cols_sorted[start:end] = rng.choice(
            m, size=count, replace=True, p=cond
        ).astype(np.int64)
        start = end

    cols = np.empty(num_edges, dtype=np.int64)
    cols[order] = cols_sorted

    w = np.full(num_edges, 1.0 / num_edges, dtype=np.float64)
    return rows, cols, w


def process_coupling(
    coupling: np.ndarray,
    *,
    topk: int | None = None,
    tau: float = 0.0,
    keep_marginals: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract edge samples and weights from a coupling matrix."""
    G = np.asarray(coupling)
    if G.ndim != 2:
        raise ValueError("Dense coupling must be 2D (n x m).")
    n, m = G.shape
    if not keep_marginals:
        if topk is not None and topk > 0:
            # Per-row threshold then top-k
            rows_l, cols_l, w_l = [], [], []
            for i in range(n):
                gi = G[i]
                if tau > 0:
                    gi = gi * (gi >= tau)
                if topk < m:
                    idx = np.argpartition(gi, -topk)[-topk:]
                else:
                    idx = np.arange(m)
                vals = gi[idx]
                mask = vals > 0
                if mask.any():
                    rows_l.extend([i] * int(mask.sum()))
                    cols_l.extend(idx[mask])
                    w_l.extend(vals[mask])
            rows = np.asarray(rows_l, dtype=np.int64)
            cols = np.asarray(cols_l, dtype=np.int64)
            w = np.asarray(w_l, dtype=np.float64)
        else:
            if tau > 0:
                G = G * (G >= tau)
            rows, cols = np.nonzero(G)
            logger.info("Number of nonzeros in coupling: %d", len(rows))
            w = G[rows, cols]
        w = np.asarray(w, dtype=np.float64)
        s = float(w.sum())
        if s <= 0:
            raise ValueError(
                "Coupling weights sum to zero after sparsification/thresholding."
            )
        w = w / s
    else:
        logger.info("Projecting coupling to keep marginals uniform.")
        p = np.ones(n) / n
        q = np.ones(m) / m
        if tau > 0:
            G = G * (G >= tau)
        G_projected = project_topk_transport(
            G, p, q, topk if topk is not None else min(n, m)
        )
        rows, cols = np.nonzero(G_projected)  # type: ignore
        w = G_projected[rows, cols]  # type: ignore

        w = np.asarray(w, dtype=np.float64)
        assert w.sum() > 0
        assert np.all(w >= 0)

        s = float(w.sum())
        assert np.isclose(s, 1.0), f"s={s}"
        if s <= 0:
            raise ValueError(
                "Coupling weights sum to zero after sparsification/thresholding."
            )

    return (rows.astype(np.int64), cols.astype(np.int64), w)


def project_topk_transport(G, p, q, k, big=1e6):
    """Project a top-k mask back to feasible marginals."""
    idx = np.argpartition(-G, k - 1, axis=1)[:, :k]  # keep top-k per row
    mask = np.zeros_like(G, bool)
    mask[np.arange(G.shape[0])[:, None], idx] = True
    C = (~mask).astype(float) * big - G  # forbid outside-mask, prefer big G
    return ot.emd(p, q, C)  # sparse & feasible


#  Train/val split (edges)


def _split_coupling_by_edges(
    coupling: np.ndarray,
    *,
    topk: int | None,
    tau: float,
    val_frac: float,
    seed: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Bernoulli thinning on edges with probability=val_frac, then renormalize per split.
    Ensures both splits are non-empty (falls back to assign the argmax edge).
    """
    rows, cols, w = process_coupling(coupling, topk=topk, tau=tau)  # w sums to 1
    rng = np.random.default_rng(seed)
    val_mask = rng.random(len(w)) < val_frac

    val_idx = np.nonzero(val_mask)[0]
    trn_idx = np.nonzero(~val_mask)[0]

    if val_idx.size == 0:
        val_idx = np.array([int(w.argmax())])
        trn_idx = np.setdiff1d(np.arange(len(w)), val_idx, assume_unique=True)
    if trn_idx.size == 0:
        trn_idx = np.array([int(w.argmax())])
        val_idx = np.setdiff1d(np.arange(len(w)), trn_idx, assume_unique=True)

    w_trn = w[trn_idx]
    w_val = w[val_idx]
    w_trn = w_trn / max(float(w_trn.sum()), 1e-12)
    w_val = w_val / max(float(w_val.sum()), 1e-12)
    return (rows[trn_idx], cols[trn_idx], w_trn), (rows[val_idx], cols[val_idx], w_val)


#  Dataset


class CoupledPairsDataset(Dataset):
    """Dataset of paired cells sampled from a transport coupling."""

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        coupling: tuple,
        *,
        emit: Literal["pair", "flow"] = "flow",
        t_sampler: Literal["uniform", "beta"] = "uniform",
        beta_a: float = 2.0,
        beta_b: float = 2.0,
        noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.emit = emit
        self.t_sampler = t_sampler
        self.beta_a = float(beta_a)
        self.beta_b = float(beta_b)
        self.noise_std = float(noise_std)

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2D arrays.")
        if X.shape[1] != Y.shape[1]:
            raise ValueError(f"Dim mismatch: X {X.shape} vs Y {Y.shape}.")
        self._X = X  # keep numpy for zero-copy torch.from_numpy
        self._Y = Y
        self.D = X.shape[1]

        rows, cols, w = coupling
        self.rows = torch.from_numpy(rows)  # (E,)
        self.cols = torch.from_numpy(cols)  # (E,)
        self.p = torch.from_numpy(w).float()  # (E,), sums to 1

    def __len__(self) -> int:
        return int(self.rows.numel())

    def _sample_t(self) -> torch.Tensor:
        if self.t_sampler == "uniform":
            return torch.rand(())  # scalar tensor
        dist = torch.distributions.Beta(self.beta_a, self.beta_b)
        return dist.sample()  # scalar tensor

    def __getitem__(self, idx: int):
        i = int(self.rows[idx])
        j = int(self.cols[idx])
        p = float(self.p[idx])
        xi = torch.from_numpy(self._X[i])
        yj = torch.from_numpy(self._Y[j])

        t = self._sample_t()
        xt = (1.0 - t) * xi + t * yj
        if self.noise_std > 0:
            xt = xt + torch.randn_like(xt) * self.noise_std
        v_star = yj - xi
        return {
            "x_t": xt,
            "xi": xi,
            "yj": yj,
            "t": t,
            "v": v_star,
            "i": i,
            "j": j,
            "p": p,
        }


#  Builders


def create_train_val_datasets(
    X: np.ndarray,
    Y: np.ndarray,
    coupling: np.ndarray,
    *,
    coupling_sampling: Literal["sparse", "dense_mc"] = "sparse",
    topk: int | None = 64,
    tau: float = 0.0,
    val_frac: float = 0.1,
    dense_train_edges: int | None = None,
    dense_val_edges: int | None = None,
    seed: int = 0,
    emit: Literal["pair", "flow"] = "flow",
    t_sampler: Literal["uniform", "beta"] = "uniform",
    beta_a: float = 2.0,
    beta_b: float = 2.0,
    noise_std: float = 0.0,
):
    """
    Disjoint edge split via Bernoulli thinning.
    """
    if coupling_sampling == "sparse":
        (r_trn, c_trn, w_trn), (r_val, c_val, w_val) = _split_coupling_by_edges(
            coupling, topk=topk, tau=tau, val_frac=val_frac, seed=seed
        )
    elif coupling_sampling == "dense_mc":
        n = int(np.asarray(coupling).shape[0])
        target_total = max(10_000, n * max(1, int(topk if topk is not None else 64)))
        if dense_train_edges is None and dense_val_edges is None:
            total_edges = target_total
            n_val = max(1, int(round(total_edges * float(val_frac))))
            n_train = max(1, total_edges - n_val)
        else:
            n_train = (
                int(dense_train_edges)
                if dense_train_edges is not None
                else max(1, int(target_total * (1.0 - float(val_frac))))
            )
            n_val = (
                int(dense_val_edges)
                if dense_val_edges is not None
                else max(
                    1,
                    int(
                        round(
                            n_train
                            * float(val_frac)
                            / max(1e-12, 1.0 - float(val_frac))
                        )
                    ),
                )
            )

        logger.info(
            "Using dense_mc coupling sampling (no top-k). train_edges=%d val_edges=%d",
            n_train,
            n_val,
        )
        r_trn, c_trn, w_trn = _sample_edges_from_dense_coupling(
            coupling, num_edges=n_train, seed=seed
        )
        r_val, c_val, w_val = _sample_edges_from_dense_coupling(
            coupling, num_edges=n_val, seed=seed + 1
        )
    else:
        raise ValueError(
            f"Unknown coupling_sampling='{coupling_sampling}'. "
            "Use 'sparse' or 'dense_mc'."
        )

    training_coupling = (r_trn, c_trn, w_trn)
    val_coupling = (r_val, c_val, w_val)
    train_ds = CoupledPairsDataset(
        X,
        Y,
        training_coupling,
        emit=emit,
        t_sampler=t_sampler,
        beta_a=beta_a,
        beta_b=beta_b,
        noise_std=noise_std,
    )
    val_ds = CoupledPairsDataset(
        X,
        Y,
        val_coupling,
        emit=emit,
        t_sampler=t_sampler,
        beta_a=beta_a,
        beta_b=beta_b,
        noise_std=noise_std,
    )
    logger.info(
        "The lengths of the training and validation datasets are: %d, %d",
        len(train_ds),
        len(val_ds),
    )
    return train_ds, val_ds


def _default_worker_init_fn(worker_id: int):
    """Make time -sampling reproducible across DataLoader workers."""
    # torch sets base seed per worker; use it to seed numpy/random if needed.
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)


def make_train_dataloader(
    ds: CoupledPairsDataset,
    *,
    batch_size: int,
    steps_per_epoch: int,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Training loader sampled with probability ∝ p (lowest-variance; Q=P).
    Returns exactly steps_per_epoch * batch_size samples per epoch.
    """
    assert isinstance(ds, CoupledPairsDataset)
    # Compute the number of steps per epoch

    num_samples = steps_per_epoch * batch_size
    gen = torch.Generator().manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights=ds.p.tolist(), num_samples=num_samples, replacement=True, generator=gen
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_default_worker_init_fn if num_workers > 0 else None,
    )


def make_eval_dataloader(
    ds: CoupledPairsDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Evaluation loader that iterates deterministically over all edges once.
    """
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_default_worker_init_fn if num_workers > 0 else None,
    )
