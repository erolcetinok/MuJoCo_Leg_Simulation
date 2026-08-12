"""Every command in scripts/ must respond to --help, and be documented.

The command list is discovered by globbing rather than hand-listed, so a new
script is covered the moment it lands — the previous hand-written list had
silently missed four commands, and `docs/CLI.md` had missed the same four.

`scripts/` holds nothing but commands now: shared helpers live in the package
(`quadruped.cli_args`), so there is no exclusion list to keep in sync.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
CLI_DOC = ROOT / "docs" / "CLI.md"

# Launch commands with src/ on the path so these test the command, not the
# health of the editable install (see conftest.py for why that can be dead).
SCRIPT_ENV = dict(os.environ)
SCRIPT_ENV["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(ROOT / "src"), SCRIPT_ENV.get("PYTHONPATH")) if p
)

COMMANDS = sorted(p.name for p in SCRIPTS.glob("*.py"))


def test_command_list_is_not_empty():
    """Guard against the glob silently matching nothing after a move."""
    assert len(COMMANDS) >= 10, f"only found {COMMANDS} in {SCRIPTS}"


def test_no_sibling_imports():
    """A script importing a sibling only resolves when run directly.

    `python -m`, an import from a notebook, or any launcher that puts the repo
    root on sys.path instead of scripts/ would break it. Shared code belongs in
    the package.
    """
    # Single leading underscore only — `from __future__ import ...` is fine.
    sibling = re.compile(r"^\s*(?:from|import)\s+_(?!_)", re.MULTILINE)
    offenders = [
        name for name in COMMANDS
        if sibling.search((SCRIPTS / name).read_text())
    ]
    assert not offenders, (
        f"{offenders} import a sibling module; move the shared code into "
        f"src/quadruped/ (see quadruped.cli_args)"
    )


def _run_help(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--help"],
        capture_output=True, text=True, timeout=30, env=SCRIPT_ENV,
    )


@pytest.mark.parametrize("script", COMMANDS)
def test_cli_help(script):
    result = _run_help(script)
    assert result.returncode == 0, (
        f"scripts/{script} --help exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower(), (
        f"scripts/{script} --help did not print usage info"
    )


def test_gui_help_without_optional_dep():
    """gui.py imports DearPyGui lazily; --help must work without the [gui] extra."""
    result = _run_help("gui.py")
    assert result.returncode == 0, (
        f"scripts/gui.py --help exited {result.returncode}\n{result.stderr}"
    )


@pytest.mark.skipif(not CLI_DOC.exists(), reason="docs/ is gitignored; not present here")
@pytest.mark.parametrize("script", COMMANDS)
def test_command_is_documented(script):
    """Every command must be named in docs/CLI.md — that doc drifted for months."""
    assert script in CLI_DOC.read_text(), (
        f"scripts/{script} is undocumented; add a section to docs/CLI.md"
    )


@pytest.mark.skipif(not CLI_DOC.exists(), reason="docs/ is gitignored; not present here")
@pytest.mark.parametrize("script", COMMANDS)
def test_every_flag_is_documented(script):
    """And every *flag* must appear too.

    Naming the command isn't enough — `CLI.md` is a hand-written mirror of
    twelve argparse blocks, which is exactly the shape of doc that rots. This
    reads the real `--help` so adding a flag without documenting it fails.
    """
    doc = CLI_DOC.read_text()
    out = _run_help(script)
    flags = sorted(set(re.findall(r"--[a-z][a-z0-9-]+", out.stdout + out.stderr)) - {"--help"})
    undocumented = [f for f in flags if f not in doc]
    assert not undocumented, (
        f"scripts/{script} has flags missing from docs/CLI.md: {undocumented}"
    )


def test_editable_install_is_alive_or_says_why():
    """Surface the dead-editable-install trap instead of letting it confuse you.

    The rest of the suite is deliberately immune (conftest puts src/ on the
    path), so without this check a broken install would only ever show up as
    every command failing by hand, with no clue why.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    probe = subprocess.run(
        [sys.executable, "-c", "import quadruped"],
        capture_output=True, text=True, timeout=30, env=env,
    )
    if probe.returncode == 0:
        return
    installed = list((Path(sys.prefix) / "lib").glob("python*/site-packages/*quadruped*.dist-info"))
    if not installed:
        pytest.skip("package not installed here; tests run via conftest's src/ path")
    pytest.skip(
        "editable install is present but DEAD — `import quadruped` fails outside "
        "pytest, so scripts/*.py will not run. Its .pth in site-packages has the "
        "macOS hidden flag and CPython 3.13 skips hidden .pth files. Fix:\n"
        "    chflags nohidden $(python -c 'import site;print(site.getsitepackages()[0])')/*.pth\n"
        "Or prefix commands with PYTHONPATH=src. See docs/STATUS.md."
    )
