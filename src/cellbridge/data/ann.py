import anndata as ad


def load_adata(path: str) -> ad.AnnData:
    """Load an AnnData object from disk."""
    return ad.read_h5ad(path)
