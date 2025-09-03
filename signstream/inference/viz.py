from __future__ import annotations

import matplotlib.pyplot as plt
from typing import Dict, List


def plot_code_usage(tokens: Dict[str, List[List[int]]]) -> None:
    """Simple bar plot of code usage for each stream."""
    for stream, seq in tokens.items():
        flat = [code for codes in seq for code in codes]
        plt.figure()
        plt.hist(flat, bins=50)
        plt.title(stream)
        plt.show()
