#!/usr/bin/env python3
"""Thin shim — real CLI lives in quadruped.cli.view. See `quad-view` after install."""
from quadruped.cli.view import main
import sys
sys.exit(main())
