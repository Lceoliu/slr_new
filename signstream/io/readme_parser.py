from __future__ import annotations

from pathlib import Path
from typing import Dict


def read_dataset_readme(path: str | Path) -> str:
    """Return contents of dataset README for logging purposes."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
