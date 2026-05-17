#!/usr/bin/env python3
"""Thin shim — real CLI lives in quadruped.cli.ik_demo. See `quad-ik-demo` after install."""
from quadruped.cli.ik_demo import main
import sys
sys.exit(main())
