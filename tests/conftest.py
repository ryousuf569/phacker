import sys
from pathlib import Path

# prefer an installed phacker (it has the compiled kernel); fall back to the source tree
try:
    import phacker  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
