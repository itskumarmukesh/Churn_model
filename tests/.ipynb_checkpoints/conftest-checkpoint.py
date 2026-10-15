import sys
from pathlib import Path

# tests import churn package from src

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))