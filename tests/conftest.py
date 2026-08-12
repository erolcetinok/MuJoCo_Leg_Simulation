"""Make `import quadruped` work whether or not the editable install is healthy.

`pip install -e .` writes a `.pth` into site-packages, and on macOS that file can
end up carrying the BSD `hidden` flag — CPython 3.13's `site.addpackage` silently
skips hidden `.pth` files, so the install reports success and then does nothing.
Symptom: `ModuleNotFoundError: No module named 'quadruped'` from a package that
`pip list` swears is installed. Fix it with

    chflags nohidden .venv/lib/python3.13/site-packages/*.pth

Rather than have the whole suite hostage to that, put `src/` on the path here.
This also means a fresh clone can run `pytest` with no install at all.

Tests that launch `scripts/*.py` in a subprocess build their own environment
with `src/` on `PYTHONPATH` (see `test_cli_entrypoints.py`) — they are testing
the command, not the health of the install.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
