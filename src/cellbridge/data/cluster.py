from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)


def _pick_rep(adata: ad.AnnData, use_rep: str | None = None) -> str:
    """Choose the embedding used to build the Leiden neighbor graph."""
    if use_rep is not None:
        if use_rep in adata.obsm:
            return use_rep
        raise KeyError(f"use_rep='{use_rep}' not found in adata.obsm")

    if "X_pca_harmony" in adata.obsm:
        return "X_pca_harmony"
    if "X_pca" in adata.obsm:
        return "X_pca"

    raise KeyError("Neither 'X_pca_harmony' nor 'X_pca' found in adata.obsm.")


def _one_hot_from_labels(labels: pd.Categorical) -> np.ndarray:
    """Build an N x K one-hot membership matrix from categorical labels."""
    codes = labels.codes.astype(int)
    one_hot = np.zeros((codes.shape[0], len(labels.categories)), dtype=float)
    one_hot[np.arange(codes.shape[0], dtype=int), codes] = 1.0
    return one_hot


def build_neighbors(
    adata: ad.AnnData,
    *,
    n_neighbors: int = 30,
    use_rep: str | None = None,
    metric: str = "euclidean",
    key_added: str | None = None,
    random_state: int = 0,
) -> ad.AnnData:
    """Compute a kNN graph on the selected embedding."""
    sc.pp.neighbors(
        adata,
        n_neighbors=int(n_neighbors),
        use_rep=_pick_rep(adata, use_rep),
        metric=metric,  # type: ignore[arg-type]
        key_added=key_added,
        random_state=random_state,
    )
    return adata


def tune_leiden_for_target_size(
    adata: ad.AnnData,
    *,
    target_cells_per_cluster: int = 40,
    grid: tuple[float, ...] = (
        0.2,
        0.4,
        0.8,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
        20.0,
        50.0,
        100.0,
        200.0,
    ),
    key_added: str = "leiden",
    neighbors_key: str | None = None,
    random_state: int = 0,
) -> tuple[pd.Categorical, float]:
    """Select the Leiden resolution with median cluster size closest to target."""
    best_labels: pd.Categorical | None = None
    best_res = float(grid[0])
    best_gap = float("inf")

    for res in grid:
        sc.tl.leiden(
            adata,
            resolution=float(res),
            key_added=key_added,
            neighbors_key=neighbors_key,
            random_state=random_state,
            flavor="leidenalg",
        )
        labels = adata.obs[key_added].astype("category")
        sizes = pd.Series(labels).value_counts().sort_index().to_numpy()
        if sizes.size == 0:
            continue

        gap = abs(float(np.median(sizes)) - float(target_cells_per_cluster))
        if gap < best_gap:
            best_gap = gap
            best_res = float(res)
            best_labels = pd.Categorical(labels)

    if best_labels is None:
        raise RuntimeError("Leiden tuning failed to produce any clusters.")

    adata.obs[key_added] = best_labels
    return best_labels, best_res


def cluster_pipeline_identity(
    adata: ad.AnnData,
) -> tuple[pd.Categorical, np.ndarray, float]:
    """Use each cell as its own cluster."""
    labels = pd.Categorical(np.arange(adata.n_obs, dtype=int))
    return labels, _one_hot_from_labels(labels), 1.0


def cluster_pipeline_leiden(
    adata: ad.AnnData,
    *,
    n_neighbors: int = 30,
    use_rep: str | None = None,
    metric: str = "euclidean",
    target_cells_per_cluster: int = 40,
    label_key: str = "leiden",
    neighbors_key: str | None = None,
    random_state: int = 0,
) -> tuple[pd.Categorical, np.ndarray, float]:
    """Run the metacell ablation clustering pipeline."""
    build_neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
        metric=metric,
        key_added=neighbors_key,
        random_state=random_state,
    )
    labels, res = tune_leiden_for_target_size(
        adata,
        target_cells_per_cluster=target_cells_per_cluster,
        key_added=label_key,
        neighbors_key=neighbors_key,
        random_state=random_state,
    )
    logger.info("Found K=%d clusters with resolution=%s", len(labels.categories), res)
    return labels, _one_hot_from_labels(labels), res
