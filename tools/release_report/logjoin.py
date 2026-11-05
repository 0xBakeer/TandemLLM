"""Moved to tools/measure/logjoin.py (one measuring method, docs/measurement.md); this name stays for old imports."""
import sys

from tools.measure import logjoin as _moved

sys.modules[__name__] = _moved
