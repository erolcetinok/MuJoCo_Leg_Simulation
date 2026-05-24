#!/usr/bin/env python3
"""Thin shim — real CLI lives in quadruped.cli.swing_hw. See `quad-swing-hw` after install."""
from quadruped.cli.swing_hw import main
import sys
sys.exit(main())
