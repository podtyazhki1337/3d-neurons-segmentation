"""Kept for older scripts: the benchmark metrics now live in neuroseg/metrics.py (unchanged definitions)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from neuroseg.metrics import *  # noqa: E402,F401,F403
from neuroseg.metrics import METRIC_COLUMNS, evaluate_volume, summarize  # noqa: E402,F401
