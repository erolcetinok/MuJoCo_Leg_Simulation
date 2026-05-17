#!/usr/bin/env python3
"""Thin shim — real CLI lives in quadruped.cli.send_foot. See `quad-send-foot` after install."""
from quadruped.cli.send_foot import main
import sys
sys.exit(main())
