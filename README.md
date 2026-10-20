# cytolume (Python)

**Explore million-cell single-cell embeddings in 2D and 3D, rendered on the GPU in your browser.**

`cytolume` turns an AnnData object (or plain NumPy arrays) into a tiled, static
*data spine*, and serves the [Cytolume](https://muyuhuatang.github.io/Cytolume/)
WebGL2 viewer on top of it, locally or inside Jupyter. It needs only NumPy.

```bash
pip install cytolume            # + "cytolume[anndata]" to read .h5ad files
```

## TestPyPI Page
[Cytolume-TestPyPI-Package](https://test.pypi.org/project/cytolume/)

## Quick start

```python
import cytolume

# From AnnData: every common embedding in .obsm (X_umap, X_tsne, X_pca, ...) becomes a layout.
cytolume.from_anndata(adata, "lung_spine", label_key="cell_type", value_key="SFTPC",
                      value_unit="log1p counts", dataset="HRA lung v1.4")

cytolume.show("lung_spine")     # Jupyter: renders inline. Elsewhere: open .url
```

From the command line:

```bash
cytolume build lung.h5ad -o lung_spine --label cell_type --value SFTPC
cytolume serve lung_spine                     # opens http://localhost:8765/
cytolume validate lung_spine                  # structural + point-conservation check

cytolume demo -o demo_spine --cells 1000000   # synthetic 1M-cell spine to try it
```

From arrays:

```python
cytolume.build_spine(
    "my_spine",
    layouts={"umap3d": umap_xyz, "tsne2d": tsne_xy},   # (N, 2|3), same cells, same order
    labels=cell_types,                                 # N strings / ints / Categorical
    values=score,                                      # N floats (optional)
    value_label="score",
)
```

## How it works

Two halves, joined by a small static file format.

**1. The data spine (this package, CPU, once).** Each layout is indexed by a
**quadtree (2D) or octree (3D)**, split until no leaf holds more than
`leaf_target` (default 60,000) cells:

- **Point conservation:** the leaves together hold every cell exactly once. This
  is checked during the build and again by `validate_spine`.
- **Stable level of detail:** internal nodes store a subsample chosen by a hash of
  `cell_id`, so the same cells represent a region at every zoom level.
- **Compact tiles:** each tile quantizes positions to int16 relative to its own
  bounding box, which is sub-pixel precise at any zoom. A tile is
  `CYTL` header + positions + `cell_id` (u32) + category (i16), so 1M 3D cells
  take about 12 MB per layout.
- **One id space:** all layouts share the same dense `cell_id` (row index). Values
  live in one float32 column indexed by `cell_id`, which is what lets the viewer
  morph between layouts cell by cell.

**2. The viewer (browser, GPU, every frame).** WebGL2 draws every cell as an
instanced, depth-correct **sphere impostor**:

- **Continuous mode** adds *optical depth* into an HDR target. Lightness reports
  how many cells overlap and hue reports their value, so density and value stay
  readable separately. There is no fixed alpha, so a dense core doesn't become a
  blob.
- **Categorical mode** is opaque and depth-tested, so two cell types never blend
  into a colour that isn't in the data.
- A screen-space minimum radius keeps rare cells visible.
- Layout switches animate on the GPU.
- Picking, the lasso and figure export all key off `cell_id`.

## Spine layout

```
my_spine/
  manifest.json                     layouts, tile trees, channels, provenance
  tiles/<layout>/<lod>/<id>.bin     quantized tiles (leaves + coarse LOD)
  dict/CL_label.json                category names
  columns/expr.f32                  value per cell, rescaled to [0, 1]
```

The viewer's channel names are fixed (`CL_label`, `expr`). The real label, unit
and data-unit domain of your value are written to the manifest, so the colour key
shows them. `value_domain="auto"` maps the 1st–99th percentile to the colour ramp.

The spine is plain static files, so you can host it with the viewer on any static
host (GitHub Pages, S3) as well as with `cytolume serve`.

## Anatomy panel

The HRA 3D anatomy snapshot (135 MB) is too large to ship on PyPI. Without it the
map works normally and the Anatomy panel shows an error state. To enable the panel,
point the server at a local snapshot, such as `web/public/vendor/hra` from the
Cytolume repository:

```bash
cytolume serve lung_spine --anatomy path/to/Cytolume/web/public/vendor/hra
```

The panel's cell-to-organ links are illustrative (**Demo mapping**). They are not
derived from your data.

## Limits

- The viewer loads every leaf tile of the active layout into GPU memory. A few
  million cells is comfortable on a recent laptop GPU. Streaming tiles by view
  frustum is the next step in the tile format.
- Up to 32,767 categories (int16). Colours come from a 24-colour palette, so they
  repeat beyond 24 types.
- One continuous value channel per spine. To switch genes, build one spine per
  gene or rebuild with a different `value_key`.

## Development

```bash
python scripts/bundle_viewer.py --web path/to/Cytolume/web  # Node 24: build + copy the viewer
pip install -e ".[test]"
pytest
python -m build                                              # sdist + wheel in dist/
```

`src/cytolume/viewer/BUILD.json` records which web commit the bundled viewer was
built from.

## Releasing

Releases are published by [`publish.yml`](.github/workflows/publish.yml) with PyPI
Trusted Publishing, so no API token is involved.

1. Bump `__version__` in `src/cytolume/__init__.py` and commit.
2. Rehearse: **Actions → Publish → Run workflow** uploads to TestPyPI.
   TestPyPI rejects a version it already has, so rehearse each version once.
3. Release: create a GitHub release with tag `v<version>`. The workflow checks that
   the tag matches `__version__`, runs the tests, and uploads to PyPI.

## License

MIT. The bundled viewer includes the Inter font (SIL Open Font License 1.1; see
`viewer/vendor/inter/LICENSE.txt`).
