"""Build the Cytolume web viewer and copy it into src/cytolume/viewer/.

Run before building a wheel:

    python scripts/bundle_viewer.py --web <path to Cytolume checkout>/web

The web source is copied to a temporary directory and built there with Node
(`npm ci` + Vite, base path "/"), so the source checkout is never modified.
Left out of the bundle:
  * public/spine/     — users serve their own spine.
  * public/vendor/hra — the 135 MB HRA anatomy snapshot, too large for PyPI.
                        `cytolume serve --anatomy <dir>` mounts it at runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
DEST = PKG_ROOT / "src" / "cytolume" / "viewer"

_SKIP_TOP = {"node_modules", "dist", "dist-app", "screenshots", ".vite"}


def _ignore(src: str, names: list[str]) -> set[str]:
    p = Path(src)
    skip = {n for n in names if n in _SKIP_TOP or n == "__pycache__"}
    if p.name == "public":
        skip.add("spine")
    if p.name == "vendor" and p.parent.name == "public":
        skip.add("hra")
    if p.name == "spine" and p.parent.name != "public":
        skip.add("build")
    return skip


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    exe = shutil.which(cmd[0])
    if exe is None:
        sys.exit(f"{cmd[0]} not found on PATH (Node.js is required to build the viewer)")
    subprocess.run([exe, *cmd[1:]], cwd=cwd, env=env, check=True)


def _git_commit(web: Path) -> str | None:
    git = shutil.which("git")
    if not git:
        return None
    r = subprocess.run([git, "rev-parse", "HEAD"], cwd=web, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--web", type=Path, required=True, help="the web/ directory of a Cytolume checkout")
    a = ap.parse_args()

    web = a.web.resolve()
    if not (web / "vite.app.config.ts").exists():
        sys.exit(f"{web} does not look like the Cytolume web/ directory")

    with tempfile.TemporaryDirectory(prefix="cytolume-viewer-") as tmp:
        work = Path(tmp) / "web"
        shutil.copytree(web, work, ignore=_ignore)
        _run(["npm", "ci", "--no-audit", "--no-fund"], work)
        _run(["npx", "tsc", "--noEmit"], work)
        env = {**os.environ, "BASE_PATH": "/"}
        _run(["npx", "vite", "build", "--config", "vite.app.config.ts"], work, env)

        built = work / "dist-app"
        if not (built / "index.html").exists():
            sys.exit("vite build produced no index.html")
        shutil.rmtree(built / "spine", ignore_errors=True)
        shutil.rmtree(built / "vendor" / "hra", ignore_errors=True)

        if DEST.exists():
            shutil.rmtree(DEST)
        shutil.copytree(built, DEST)

    pkg = json.loads((web / "package.json").read_text(encoding="utf-8"))
    info = {
        "source": "https://github.com/muyuhuatang/Cytolume",  # never a local path
        "sourceCommit": _git_commit(web),
        "webVersion": pkg.get("version"),
        "builtAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "basePath": "/",
        "excluded": ["spine/", "vendor/hra/"],
    }
    (DEST / "BUILD.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")

    files = [f for f in DEST.rglob("*") if f.is_file()]
    size = sum(f.stat().st_size for f in files)
    print(f"\nviewer bundled: {len(files)} files, {size / 1e6:.1f} MB -> {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
