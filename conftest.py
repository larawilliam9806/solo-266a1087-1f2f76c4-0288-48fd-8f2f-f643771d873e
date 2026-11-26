import os
import sys

# Ensure the repository root is importable regardless of pytest's import mode.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
