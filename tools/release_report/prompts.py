"""Moved to tools/measure/prompts.py (one measuring method, docs/measurement.md); this name stays for old imports."""
import sys

from tools.measure import prompts as _moved

sys.modules[__name__] = _moved
