import numpy as np


def truncate(Z: np.ndarray, n_dims: int) -> np.ndarray:
    """Keep the first n representation dimensions."""
    if not isinstance(Z, np.ndarray):
        Z = np.asarray(Z)

    if n_dims is None:
        return Z
    if n_dims < 0:
        raise ValueError(f"n_dims must be non-negative, got {n_dims}")

    if Z.ndim == 1:
        return Z[: min(Z.shape[0], n_dims)]
    if Z.ndim >= 2:
        return Z[:, : min(Z.shape[1], n_dims)]

    return Z


class TransformPipeline:
    """Apply representation transforms in sequence."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        """Transform a representation matrix."""
        for transform in self.transforms:
            Z = transform(Z)
        return Z
