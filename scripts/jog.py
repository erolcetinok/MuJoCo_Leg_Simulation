#!/usr/bin/env python3
"""Thin shim — real CLI lives in quadruped.cli.jog. See `quad-jog` after install."""
from quadruped.cli.jog import main
import sys
sys.exit(main())
