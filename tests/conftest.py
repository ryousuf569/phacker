import sys
from pathlib import Path

# make `src` importable whether pytest is run from the repo root or tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
