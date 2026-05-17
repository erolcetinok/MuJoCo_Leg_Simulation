"""Every `quad-*` CLI entry point must respond to --help cleanly.

This is the cheapest possible "did we wire all the entry points" check —
catches typos in pyproject.toml, broken imports inside the CLI modules,
and accidental top-level side effects.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

CLI_MODULES = [
    "quadruped.cli.send_foot",
    "quadruped.cli.send_angles",
    "quadruped.cli.jog",
    "quadruped.cli.view",
    "quadruped.cli.ik_demo",
    "quadruped.cli.gait_demo",
    "quadruped.cli.codegen",
]


@pytest.mark.parametrize("module", CLI_MODULES)
def test_cli_help(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, (
        f"{module} --help exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower(), (
        f"{module} --help did not print usage info"
    )


def test_gui_cli_help_optional_dep():
    """quad-gui imports DearPyGui lazily; --help should always work."""
    result = subprocess.run(
        [sys.executable, "-m", "quadruped.cli.gui", "--help"],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, (
        f"quad-gui --help exited {result.returncode}\n{result.stderr}"
    )
