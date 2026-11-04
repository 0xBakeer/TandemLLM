"""Moved to tools/measure/box.py (one measuring method, docs/measurement.md); this name stays for old imports."""
import sys

from tools.measure import box as _moved

sys.modules[__name__] = _moved
