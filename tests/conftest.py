"""Gemeinsame Fixtures und Suchpfade: Demo-Ordner und tools/ liegen im Importpfad."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
