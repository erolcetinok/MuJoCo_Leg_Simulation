#!/usr/bin/env python3
"""Thin shim — real CLI lives in quadruped.cli.gait_demo. See `quad-gait-demo` after install."""
from quadruped.cli.gait_demo import main
import sys
sys.exit(main())
