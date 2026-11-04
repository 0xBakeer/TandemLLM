"""Moved to tools/measure/page.py (the report view of tools/measure); this name stays for old imports."""
import sys

from tools.measure import page as _moved

sys.modules[__name__] = _moved
