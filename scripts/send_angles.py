#!/usr/bin/env python3
"""Thin shim — real CLI lives in quadruped.cli.send_angles. See `quad-send-angles` after install."""
from quadruped.cli.send_angles import main
import sys
sys.exit(main())
