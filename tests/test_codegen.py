"""Verify that the committed generated artifacts match configs/robot.yaml."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_codegen_check_passes():
    """`scripts/codegen.py --check` must exit 0 on a clean checkout."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "codegen.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"codegen --check failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
