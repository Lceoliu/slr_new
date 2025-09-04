"""Utilities to read dataset README files."""
from __future__ import annotations

from typing import Dict


def read_text(path: str) -> str:
    """Return the contents of a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
