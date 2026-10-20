import json
import urllib.request

import numpy as np
import pytest

import cytolume
from cytolume import binfmt
from cytolume.server import viewer_available
from cytolume.tree import build_tree


def test_tile_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(500, 3)).astype(np.float32)
    ids = np.arange(500, dtype=np.uint32)
    cats = rng.integers(0, 40, 500).astype(np.int16)
    binfmt.write_tile(tmp_path / "t.bin", 3, pos, ids, cats)
    t = binfmt.read_tile(tmp_path / "t.bin")
    assert t["count"] == 500
    span = pos.max(0) - pos.min(0)
    assert np.all(np.abs(t["positions"] - pos) <= span / 65535 + 1e-6)
    assert np.array_equal(t["cell_ids"], ids) and np.array_equal(t["categories"], cats)


@pytest.mark.parametrize("dims", [2, 3])
def test_tree_conserves_points(dims):
    pts = np.random.default_rng(2).normal(size=(20_000, dims))
    t = build_tree(pts, np.arange(20_000), leaf_target=1_000)
    leaves = [n for n in t["nodes"].values() if n["is_leaf"]]
    assert t["lodLevels"] > 1 and all(n["count"] <= 1_000 for n in leaves)
    allidx = np.concatenate([n["point_indices"] for n in leaves])
    assert np.array_equal(np.sort(allidx), np.arange(20_000))


def test_tree_duplicate_points_stop_splitting():
    t = build_tree(np.zeros((5_000, 2)), np.arange(5_000), leaf_target=100)
    assert len(t["nodes"]) == 1


def test_build_and_validate(tmp_path):
    layouts, labels, values = cytolume.make_demo(n_cells=30_000, n_types=12)
    out = tmp_path / "spine"
    cytolume.build_spine(out, layouts, labels, values, value_label="score", value_unit="a.u.",
                         dataset="unit test", leaf_target=2_000, verbose=False)
    assert cytolume.validate_spine(out) == []
    m = json.loads((out / "manifest.json").read_text())
    assert m["cellCount"] == 30_000 and set(m["layouts"]) == {"umap3d", "tsne2d", "pca3d"}
    assert m["continuous"]["expr"]["label"] == "score"
    assert m["provenance"]["synthetic"] is False
    expr = np.fromfile(out / "columns" / "expr.f32", dtype="<f4")
    assert expr.min() >= 0 and expr.max() <= 1

    with pytest.raises(FileExistsError):
        cytolume.build_spine(out, layouts, labels, verbose=False)
    cytolume.build_spine(out, {"only": layouts["tsne2d"]}, verbose=False, overwrite=True)
    assert cytolume.validate_spine(out) == []


def test_validate_detects_missing_tile(tmp_path):
    layouts, labels, _ = cytolume.make_demo(n_cells=5_000, n_types=4)
    out = tmp_path / "s"
    cytolume.build_spine(out, {"u": layouts["umap3d"]}, labels, leaf_target=500, verbose=False)
    next((out / "tiles" / "u").rglob("*.bin")).unlink()
    assert cytolume.validate_spine(out)


def test_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        cytolume.build_spine(tmp_path / "a", {"x": np.zeros((10, 4))}, verbose=False)
    with pytest.raises(ValueError):
        cytolume.build_spine(tmp_path / "b", {"x": np.zeros((10, 2)), "y": np.zeros((9, 2))}, verbose=False)
    with pytest.raises(ValueError):
        cytolume.build_spine(tmp_path / "c", {"x": np.full((10, 2), np.nan)}, verbose=False)


def test_label_encodings():
    from cytolume.spine import encode_labels

    codes, names = encode_labels(np.array(["b", "a", "b"]), 3)
    assert names == ["a", "b"] and codes.tolist() == [1, 0, 1]
    pd = pytest.importorskip("pandas")
    codes, names = encode_labels(pd.Series(pd.Categorical(["x", None, "y"])), 3)
    assert names == ["x", "y", "NA"] and codes.tolist() == [0, 2, 1]


@pytest.mark.skipif(not viewer_available(), reason="viewer bundle not built")
def test_server_routes(tmp_path):
    layouts, labels, values = cytolume.make_demo(n_cells=2_000, n_types=4)
    cytolume.build_spine(tmp_path, layouts, labels, values, verbose=False)
    with cytolume.show(tmp_path) as srv:
        with urllib.request.urlopen(srv.url) as r:
            assert b"Cytolume" in r.read()
        with urllib.request.urlopen(srv.url + "spine/manifest.json") as r:
            assert json.load(r)["cellCount"] == 2_000
        index = urllib.request.urlopen(srv.url).read().decode()
        script = index.split('src="', 1)[1].split('"', 1)[0]
        with urllib.request.urlopen(srv.url + script.lstrip("/")) as r:
            assert r.headers["Content-Type"].startswith("text/javascript")


def test_bundled_viewer_records_no_local_path():
    import json
    from pathlib import Path

    viewer = Path(cytolume.__file__).parent / "viewer"
    assert (viewer / "index.html").exists()
    info = json.loads((viewer / "BUILD.json").read_text(encoding="utf-8"))
    assert info["source"].startswith("https://"), info["source"]
