#!/usr/bin/env python3
"""Thin shim — real CLI lives in quadruped.cli.gui. See `quad-gui` after install."""
from quadruped.cli.gui import main
import sys
sys.exit(main())
