"""Build and validate a Cytolume *spine* — the static, browser-ready data store.

A spine directory is everything the viewer reads over plain HTTP:

    manifest.json                    entry point: layouts, tile trees, channels
    tiles/<layout>/<lod>/<id>.bin    quantized position tiles (see binfmt.py)
    dict/CL_label.json               category names
    columns/expr.f32                 continuous value per cell (LE float32, by cell_id)

Every layout is keyed to the SAME dense ``cell_id`` (row index), which is what
lets the viewer morph one embedding into another and join values by id.

The viewer's channel names are fixed: the categorical channel is ``CL_label`` and
the continuous channel is ``expr``. The human-facing label, unit and domain of the
value channel are declared in the manifest, so the colour key shows what the
values actually are.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import binfmt
from .tree import build_tree

CATEGORY_CHANNEL = "CL_label"
VALUE_CHANNEL = "expr"
DEFAULT_LEAF_TARGET = 60_000
_MAX_CATEGORIES = 32_767  # categories are stored as int16
_CORE_24 = 24  # size of the viewer's categorical palette


def _as_points(name: str, coords: Any, n: int | None) -> np.ndarray:
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(f"layout {name!r} must be (N, 2) or (N, 3), got {pts.shape}")
    if n is not None and pts.shape[0] != n:
        raise ValueError(f"layout {name!r} has {pts.shape[0]} rows, expected {n}")
    if not np.isfinite(pts).all():
        raise ValueError(f"layout {name!r} contains NaN or infinite coordinates")
    return pts


def encode_labels(labels: Any, n: int) -> tuple[np.ndarray, list[str]]:
    """Turn labels (strings, ints, or a pandas Categorical/Series) into int16 codes + names."""
    if labels is None:
        return np.zeros(n, dtype=np.int16), ["all cells"]

    cat = getattr(labels, "cat", None)  # pandas Series with categorical dtype
    if cat is not None:
        labels = labels.values
    if hasattr(labels, "categories") and hasattr(labels, "codes"):  # pandas Categorical
        codes = np.asarray(labels.codes, dtype=np.int64)
        names = [str(c) for c in labels.categories]
        if (codes < 0).any():
            names.append("NA")
            codes = np.where(codes < 0, len(names) - 1, codes)
    else:
        arr = np.asarray(labels)
        if arr.dtype.kind == "f":
            arr = np.where(np.isnan(arr), -1, arr).astype(np.int64)
        if arr.dtype.kind == "O":
            arr = np.array(["NA" if v is None else str(v) for v in arr])
        uniq, codes = np.unique(arr, return_inverse=True)
        names = [str(u) for u in uniq]

    codes = np.asarray(codes).reshape(-1)
    if codes.shape[0] != n:
        raise ValueError(f"labels has {codes.shape[0]} entries, expected {n}")
    if len(names) > _MAX_CATEGORIES:
        raise ValueError(f"{len(names)} categories exceeds the int16 limit of {_MAX_CATEGORIES}")
    return codes.astype(np.int16), names


def _normalize_values(values: Any, n: int, domain: Any) -> tuple[np.ndarray, list[float]]:
    """Rescale values to [0, 1] for the shader; return them plus the domain in data units.

    ``domain`` is "auto" (1st–99th percentile, robust to outliers), "extent"
    (min/max), or an explicit (lo, hi) pair.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.shape[0] != n:
        raise ValueError(f"values has {v.shape[0]} entries, expected {n}")
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        raise ValueError("values contains no finite numbers")
    if isinstance(domain, str):
        if domain == "auto":
            lo, hi = np.percentile(finite, [1.0, 99.0])
            if hi <= lo:  # e.g. a sparse gene that is zero in >99% of cells
                lo, hi = finite.min(), finite.max()
        elif domain == "extent":
            lo, hi = finite.min(), finite.max()
        else:
            raise ValueError("value_domain must be 'auto', 'extent' or a (lo, hi) pair")
    else:
        lo, hi = (float(x) for x in domain)
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    norm = np.where(np.isfinite(norm), norm, 0.0)
    return norm.astype(np.float32), [float(lo), float(hi)]


def _id_checksum(n: int) -> str:
    ids = np.arange(n, dtype="<u4")
    return hashlib.sha256(ids.tobytes()).hexdigest()


def build_spine(
    out_dir: str | Path,
    layouts: Mapping[str, Any],
    labels: Any = None,
    values: Any = None,
    *,
    value_label: str | None = None,
    value_unit: str | None = None,
    value_transform: str | None = None,
    value_domain: Any = "auto",
    dataset: str | None = None,
    source: str | None = None,
    cell_type_ontology: str | None = None,
    synthetic: bool = False,
    leaf_target: int = DEFAULT_LEAF_TARGET,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Write a spine for one set of cells under ``out_dir`` and return the manifest path.

    Args:
        out_dir: output directory (created; must be empty unless ``overwrite``).
        layouts: ``{"umap": (N, 2|3) array, "pca": ...}``. Every layout must list
            the same cells in the same row order. 2D and 3D layouts can be mixed.
        labels: per-cell category (strings, ints, or a pandas Categorical). Drives
            the categorical colouring, legend and lasso summaries.
        values: per-cell continuous value (e.g. one gene's expression). Optional.
        value_label / value_unit / value_transform: how the colour key names the
            value channel, e.g. ``"CD3E"``, ``"log1p counts"``.
        value_domain: "auto" (1st–99th percentile), "extent", or ``(lo, hi)``.
        dataset / source / cell_type_ontology: provenance printed on exported figures.
        synthetic: mark the data as synthetic; exported figures then say so.
        leaf_target: maximum points per leaf tile.
    """
    out = Path(out_dir)
    if not layouts:
        raise ValueError("at least one layout is required")

    points: dict[str, np.ndarray] = {}
    n: int | None = None
    for name, coords in layouts.items():
        if not name or "/" in name or "\\" in name:
            raise ValueError(f"invalid layout name {name!r}")
        points[name] = _as_points(name, coords, n)
        n = points[name].shape[0]
    assert n is not None
    if n == 0:
        raise ValueError("layouts are empty")
    if n >= 2**32 - 1:
        raise ValueError("too many cells for uint32 cell ids")

    codes, names = encode_labels(labels, n)

    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty (pass overwrite=True to replace it)")
        if not (out / "manifest.json").exists():
            raise FileExistsError(f"{out} is not empty and is not a spine; refusing to delete it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    log = print if verbose else (lambda *a, **k: None)
    cell_id = np.arange(n, dtype=np.uint32)

    # --- categorical dict ---
    (out / "dict").mkdir(exist_ok=True)
    cl_dict = {"names": names, "colorSlot": [i % _CORE_24 for i in range(len(names))]}
    (out / "dict" / f"{CATEGORY_CHANNEL}.json").write_text(json.dumps(cl_dict), encoding="utf-8")

    # --- continuous column ---
    continuous: dict[str, Any]
    (out / "columns").mkdir(exist_ok=True)
    if values is not None:
        norm, dom = _normalize_values(values, n, value_domain)
        continuous = {
            "uri": f"columns/{VALUE_CHANNEL}.f32",
            "label": value_label or "value",
            "domain": dom,
        }
        if value_unit:
            continuous["unit"] = value_unit
        continuous["transform"] = value_transform or f"linear rescale of [{dom[0]:.4g}, {dom[1]:.4g}], clipped"
    else:
        # The viewer always expects the channel; zeros render as the ramp's floor.
        norm = np.zeros(n, dtype=np.float32)
        continuous = {"uri": f"columns/{VALUE_CHANNEL}.f32", "label": "no value channel"}
    (out / "columns" / f"{VALUE_CHANNEL}.f32").write_bytes(np.ascontiguousarray(norm, dtype="<f4").tobytes())

    # --- tiles per layout ---
    manifest_layouts: dict[str, dict] = {}
    for name, pts in points.items():
        dims = pts.shape[1]
        tree = build_tree(pts, cell_id, leaf_target)
        tiles = []
        for node in tree["nodes"].values():
            pidx = node["point_indices"]
            uri = f"tiles/{name}/{node['lod']}/{node['id']}.bin"
            binfmt.write_tile(out / uri, dims, pts[pidx], cell_id[pidx], codes[pidx])
            tiles.append(
                {
                    "id": node["id"],
                    "lod": node["lod"],
                    "bbox": node["bbox"],
                    "count": node["count"],
                    "children": [int(c) for c in node["children"]],
                    "uri": uri,
                }
            )
        tiles.sort(key=lambda t: (t["lod"], t["id"]))
        manifest_layouts[name] = {
            "dims": dims,
            "bounds": tree["bounds"],
            "tree": tree["tree"],
            "leafTarget": tree["leafTarget"],
            "lodLevels": tree["lodLevels"],
            "tiles": tiles,
        }
        leaves = sum(1 for t in tiles if not t["children"])
        log(f"  layout {name!r}: {dims}D {tree['tree']}, {len(tiles)} tiles ({leaves} leaves), {tree['lodLevels']} LOD levels")

    provenance: dict[str, Any] = {"synthetic": bool(synthetic), "source": source or "cytolume-python"}
    if dataset:
        provenance["dataset"] = dataset
    if cell_type_ontology:
        provenance["cellTypeOntology"] = cell_type_ontology

    manifest = {
        "version": 1,
        "cellCount": int(n),
        "layouts": manifest_layouts,
        "categorical": {CATEGORY_CHANNEL: {"dict": f"dict/{CATEGORY_CHANNEL}.json", "count": len(names)}},
        "continuous": {VALUE_CHANNEL: continuous},
        "idSpace": {
            "field": "cell_id",
            "stable": True,
            "sharedAcrossLayouts": True,
            "checksum": _id_checksum(n),
        },
        "provenance": provenance,
    }
    problems = validate_manifest(manifest)
    if problems:
        raise AssertionError("built an invalid manifest:\n  - " + "\n  - ".join(problems))

    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    log(f"  wrote {n:,} cells, {len(names)} categories, {size / 1e6:.1f} MB -> {out}")
    return path


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Structural checks on a manifest dict. Returns a list of problems (empty == valid)."""
    problems: list[str] = []
    if manifest.get("version") != 1:
        problems.append("version must be 1")
    cell_count = manifest.get("cellCount")
    if not isinstance(cell_count, int) or cell_count <= 0:
        problems.append("cellCount must be a positive int")

    layouts = manifest.get("layouts") or {}
    if not layouts:
        problems.append("no layouts present")
    for key, entry in layouts.items():
        dims = entry.get("dims")
        if dims not in (2, 3):
            problems.append(f"{key}: dims must be 2 or 3")
            continue
        if len(entry.get("bounds", [])) != dims * 2:
            problems.append(f"{key}: bounds must have {dims * 2} numbers")
        if entry.get("tree") != ("quadtree" if dims == 2 else "octree"):
            problems.append(f"{key}: wrong tree type for dims={dims}")
        tiles = entry.get("tiles") or []
        if not tiles:
            problems.append(f"{key}: no tiles")
            continue
        ids = {t["id"] for t in tiles}
        if len(ids) != len(tiles):
            problems.append(f"{key}: duplicate tile ids")
        leaf_total = 0
        for t in tiles:
            if len(t.get("bbox", [])) != dims * 2:
                problems.append(f"{key}: tile {t.get('id')} bbox wrong length")
            for c in t.get("children", []):
                if c not in ids:
                    problems.append(f"{key}: tile {t['id']} references missing child {c}")
            if not t.get("children"):
                leaf_total += int(t.get("count", 0))
        if isinstance(cell_count, int) and leaf_total != cell_count:
            problems.append(f"{key}: leaf point sum {leaf_total} != cellCount {cell_count}")

    idspace = manifest.get("idSpace") or {}
    if not idspace.get("checksum"):
        problems.append("idSpace.checksum missing")
    if idspace.get("field") != "cell_id":
        problems.append("idSpace.field must be 'cell_id'")
    if VALUE_CHANNEL not in (manifest.get("continuous") or {}):
        problems.append(f"continuous.{VALUE_CHANNEL} missing")
    if CATEGORY_CHANNEL not in (manifest.get("categorical") or {}):
        problems.append(f"categorical.{CATEGORY_CHANNEL} missing")
    return problems


def validate_spine(spine_dir: str | Path, deep: bool = True) -> list[str]:
    """Validate a spine on disk. With ``deep``, decode every leaf tile and check that each
    layout contains every cell exactly once. Returns a list of problems (empty == valid)."""
    root = Path(spine_dir)
    mpath = root / "manifest.json"
    if not mpath.exists():
        return [f"missing {mpath}"]
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    problems = validate_manifest(manifest)
    if problems:
        return problems

    n = manifest["cellCount"]
    cont = manifest["continuous"][VALUE_CHANNEL]
    col = root / (cont if isinstance(cont, str) else cont["uri"])
    if not col.exists():
        problems.append(f"missing value column {col}")
    elif col.stat().st_size != n * 4:
        problems.append(f"value column has {col.stat().st_size // 4} entries, expected {n}")
    dpath = root / manifest["categorical"][CATEGORY_CHANNEL]["dict"]
    n_cat = 0
    if not dpath.exists():
        problems.append(f"missing category dict {dpath}")
    else:
        n_cat = len(json.loads(dpath.read_text(encoding="utf-8"))["names"])

    for key, entry in manifest["layouts"].items():
        seen = np.zeros(n, dtype=bool) if deep else None
        for t in entry["tiles"]:
            tpath = root / t["uri"]
            if not tpath.exists():
                problems.append(f"{key}: missing tile {t['uri']}")
                continue
            if not deep or t["children"]:
                continue
            tile = binfmt.read_tile(tpath)
            if tile["dims"] != entry["dims"] or tile["count"] != t["count"]:
                problems.append(f"{key}: tile {t['uri']} header disagrees with manifest")
                continue
            cid = tile["cell_ids"]
            if cid.size and (cid.max() >= n or seen[cid].any()):
                problems.append(f"{key}: tile {t['uri']} has out-of-range or duplicate cell ids")
                continue
            seen[cid] = True
            if n_cat and tile["categories"].size and tile["categories"].max() >= n_cat:
                problems.append(f"{key}: tile {t['uri']} has category codes beyond the dict")
        if deep and not seen.all():
            problems.append(f"{key}: {int((~seen).sum())} cells missing from leaf tiles")
    return problems


def make_demo(
    n_cells: int = 1_000_000,
    n_types: int = 48,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Seeded synthetic data: clustered 2D + 3D layouts, power-law cell-type sizes and a
    value that correlates with type. Returns ``(layouts, labels, values)``."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, n_types + 1) ** 1.1
    rng.shuffle(p)
    labels = rng.choice(n_types, size=n_cells, p=p / p.sum())

    layouts: dict[str, np.ndarray] = {}
    for name, dims, spread, sigma in (("umap3d", 3, 80.0, 4.0), ("tsne2d", 2, 100.0, 3.5), ("pca3d", 3, 34.0, 9.0)):
        centers = rng.normal(0.0, spread, size=(n_types, dims))
        scale = rng.uniform(0.6, 1.6, size=(n_types, dims)) * sigma
        layouts[name] = (centers[labels] + rng.normal(size=(n_cells, dims)) * scale[labels]).astype(np.float32)

    base = rng.uniform(0.0, 1.0, size=n_types)
    values = np.clip(base[labels] + rng.normal(0.0, 0.15, size=n_cells), 0.0, 1.0).astype(np.float32)
    names = np.array([f"type {i:02d}" for i in range(n_types)])
    return layouts, names[labels], values


def build_demo(out_dir: str | Path, n_cells: int = 1_000_000, seed: int = 0, **kwargs: Any) -> Path:
    """Build a synthetic demo spine (marked synthetic, so exported figures say so)."""
    layouts, labels, values = make_demo(n_cells=n_cells, seed=seed)
    kwargs.setdefault("value_label", "demo score")
    kwargs.setdefault("dataset", f"cytolume demo ({n_cells:,} synthetic cells, seed {seed})")
    return build_spine(out_dir, layouts, labels, values, synthetic=True, value_domain="extent", **kwargs)


def layout_names(spine_dir: str | Path) -> Sequence[str]:
    """List the layouts in an existing spine."""
    manifest = json.loads((Path(spine_dir) / "manifest.json").read_text(encoding="utf-8"))
    return list(manifest["layouts"])
