"""Command line: ``cytolume build|demo|serve|validate``."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _cmd_build(a: argparse.Namespace) -> int:
    from .anndata_io import from_anndata

    from_anndata(
        a.input,
        a.out,
        label_key=None if a.label == "" else a.label,
        value_key=a.value,
        layouts=a.layout or None,
        dims=a.dims,
        layer=a.layer,
        dataset=a.dataset,
        value_unit=a.unit,
        leaf_target=a.leaf_target,
        overwrite=a.overwrite,
    )
    print(f"\nNext: cytolume serve {a.out}")
    return 0


def _cmd_demo(a: argparse.Namespace) -> int:
    from .spine import build_demo

    build_demo(a.out, n_cells=a.cells, seed=a.seed, leaf_target=a.leaf_target, overwrite=a.overwrite)
    print(f"\nNext: cytolume serve {a.out}")
    return 0


def _cmd_serve(a: argparse.Namespace) -> int:
    from .server import serve

    serve(a.spine, port=a.port, host=a.host, anatomy=a.anatomy, open_browser=not a.no_browser, quiet=not a.verbose)
    return 0


def _cmd_validate(a: argparse.Namespace) -> int:
    from .spine import validate_spine

    problems = validate_spine(a.spine, deep=not a.quick)
    if problems:
        print("INVALID:\n  - " + "\n  - ".join(problems))
        return 1
    print("OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cytolume", description="GPU embedding viewer for million-cell single-cell data.")
    p.add_argument("--version", action="version", version=f"cytolume {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a spine from an .h5ad file")
    b.add_argument("input", help="path to an .h5ad file")
    b.add_argument("-o", "--out", required=True, help="output spine directory")
    b.add_argument("--label", default="cell_type", help="obs column for cell categories ('' for none)")
    b.add_argument("--value", help="obs column or gene name for the continuous value")
    b.add_argument("--unit", help="unit of the value, e.g. 'log1p counts'")
    b.add_argument("--layer", help="adata.layers key to read gene values from")
    b.add_argument("--layout", action="append", help="obsm key to export (repeatable; default: all common embeddings)")
    b.add_argument("--dims", type=int, choices=(2, 3), help="force 2D or 3D layouts")
    b.add_argument("--dataset", help="dataset name for figure provenance")
    b.add_argument("--leaf-target", type=int, default=60_000)
    b.add_argument("--overwrite", action="store_true")
    b.set_defaults(func=_cmd_build)

    d = sub.add_parser("demo", help="build a synthetic demo spine")
    d.add_argument("-o", "--out", required=True)
    d.add_argument("--cells", type=int, default=1_000_000)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--leaf-target", type=int, default=60_000)
    d.add_argument("--overwrite", action="store_true")
    d.set_defaults(func=_cmd_demo)

    s = sub.add_parser("serve", help="open the viewer on a spine")
    s.add_argument("spine", help="spine directory (contains manifest.json)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--anatomy", help="local HRA snapshot directory (vendor/hra) for the anatomy panel")
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--verbose", action="store_true", help="log every request")
    s.set_defaults(func=_cmd_serve)

    v = sub.add_parser("validate", help="check a spine for structural and point-conservation errors")
    v.add_argument("spine")
    v.add_argument("--quick", action="store_true", help="skip decoding tiles")
    v.set_defaults(func=_cmd_validate)

    a = p.parse_args(argv)
    try:
        return a.func(a)
    except (KeyError, ValueError, FileNotFoundError, FileExistsError, ImportError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(f"cytolume {a.cmd}: error: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
