"""`quad-codegen` entry point — delegates to scripts/codegen.py."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "codegen.py"


def main() -> int:
    if not SCRIPT.exists():
        sys.stderr.write(f"codegen script not found at {SCRIPT}\n")
        return 1
    # Run scripts/codegen.py as __main__ so its argparse sees our argv as-is.
    sys.argv[0] = str(SCRIPT)
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
