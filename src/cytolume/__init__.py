"""Cytolume — GPU embedding visualization for million-cell single-cell data.

    import cytolume
    cytolume.from_anndata(adata, "my_spine", label_key="cell_type", value_key="CD3E")
    cytolume.show("my_spine")          # in Jupyter; or `cytolume serve my_spine`
"""

__version__ = "0.1.0"

from .anndata_io import from_anndata, layouts_from_obsm
from .binfmt import read_tile, write_tile
from .server import serve, show
from .spine import build_demo, build_spine, make_demo, validate_spine
from .tree import build_tree

__all__ = [
    "__version__",
    "build_spine",
    "build_demo",
    "make_demo",
    "from_anndata",
    "layouts_from_obsm",
    "validate_spine",
    "serve",
    "show",
    "build_tree",
    "read_tile",
    "write_tile",
]
