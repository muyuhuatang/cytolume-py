"""Build a spine straight from an AnnData object or an .h5ad file.

AnnData is duck-typed (``.obsm``, ``.obs``, ``.obs_vector``), so this module does
not import ``anndata`` unless it has to read a file from disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .spine import build_spine

# obsm keys picked up automatically, in display order.
_AUTO_KEYS = ("X_umap", "X_tsne", "X_pca", "X_draw_graph_fa", "X_phate", "X_diffmap")


def _layout_name(key: str, dims: int) -> str:
    base = key[2:] if key.startswith("X_") else key
    return f"{base}{dims}d"


def _dense(a: Any) -> np.ndarray:
    if hasattr(a, "toarray"):
        a = a.toarray()
    if hasattr(a, "to_numpy"):
        a = a.to_numpy()
    return np.asarray(a)


def _gene_vector(adata: Any, gene: str, layer: str | None) -> np.ndarray:
    """One gene's values across all cells, from ``adata.X`` or ``adata.layers[layer]``."""
    try:
        idx = adata.var_names.get_loc(gene)
    except KeyError:
        raise KeyError(f"{gene!r} is neither an adata.obs column nor a gene in adata.var_names") from None
    mat = adata.layers[layer] if layer else adata.X
    return _dense(mat[:, idx]).reshape(-1).astype(np.float64)


def layouts_from_obsm(adata: Any, keys: Sequence[str] | Mapping[str, str] | None = None, dims: int | None = None) -> dict[str, np.ndarray]:
    """Collect embedding layouts from ``adata.obsm``.

    ``keys`` may be a list of obsm keys or a ``{layout_name: obsm_key}`` mapping.
    Embeddings with more than 3 columns (e.g. PCA) use their first 3 (or ``dims``).
    """
    obsm = adata.obsm
    if keys is None:
        keys = [k for k in _AUTO_KEYS if k in obsm]
        if not keys:
            raise ValueError(f"no embedding found in adata.obsm (looked for {', '.join(_AUTO_KEYS)}); pass layouts=")
    pairs = list(keys.items()) if isinstance(keys, Mapping) else [(None, k) for k in keys]

    out: dict[str, np.ndarray] = {}
    for name, key in pairs:
        if key not in obsm:
            raise KeyError(f"adata.obsm has no {key!r} (available: {list(obsm.keys())})")
        arr = _dense(obsm[key]).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"adata.obsm[{key!r}] must have at least 2 columns")
        d = dims or min(arr.shape[1], 3)
        d = min(d, arr.shape[1])
        out[name or _layout_name(key, d)] = arr[:, :d]
    return out


def from_anndata(
    adata: Any,
    out_dir: str | Path,
    *,
    label_key: str | None = "cell_type",
    value_key: str | None = None,
    layouts: Sequence[str] | Mapping[str, str] | None = None,
    dims: int | None = None,
    layer: str | None = None,
    dataset: str | None = None,
    **kwargs: Any,
) -> Path:
    """Write a spine from an AnnData object (or a path to an ``.h5ad`` file).

    Args:
        adata: ``anndata.AnnData`` or a path to an ``.h5ad`` file.
        out_dir: output spine directory.
        label_key: ``adata.obs`` column used as the cell category. ``None`` for no labels.
            If the column is missing, common HRA/CELLxGENE names are tried
            (``cell_type``, ``CL_label``, ``final_level_labels``, ``leiden``, ``louvain``).
        value_key: an ``adata.obs`` column or a gene in ``adata.var_names`` used as the
            continuous value. ``None`` for no value channel.
        layouts: obsm keys to export (default: every common embedding found).
        dims: force 2 or 3 columns per embedding (default: up to 3).
        layer: read gene values from ``adata.layers[layer]`` instead of ``adata.X``.
        dataset: provenance name; defaults to the file name when a path is given.
        **kwargs: passed to :func:`cytolume.build_spine` (e.g. ``value_unit``, ``leaf_target``).
    """
    if isinstance(adata, (str, Path)):
        try:
            import anndata
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("reading .h5ad files needs anndata: pip install 'cytolume[anndata]'") from e
        dataset = dataset or Path(adata).name
        adata = anndata.read_h5ad(adata, backed=None)

    lay = layouts_from_obsm(adata, layouts, dims)

    labels = None
    if label_key is not None:
        candidates = [label_key, "cell_type", "CL_label", "final_level_labels", "leiden", "louvain"]
        found = next((k for k in candidates if k in adata.obs.columns), None)
        if found is None:
            raise KeyError(f"adata.obs has no {label_key!r} column (available: {list(adata.obs.columns)})")
        labels = adata.obs[found]
        if found != label_key:
            print(f"  label column {label_key!r} not found; using {found!r}")

    values = None
    if value_key is not None:
        if value_key in adata.obs.columns:
            values = _dense(adata.obs[value_key]).astype(np.float64)
        else:
            values = _gene_vector(adata, value_key, layer)
        kwargs.setdefault("value_label", value_key)

    return build_spine(out_dir, lay, labels, values, dataset=dataset, **kwargs)
