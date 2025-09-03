from __future__ import annotations

from typing import Any, List


def run_length_encode(seq: List[Any]) -> List[Any]:
    """Encode a sequence using simple run-length encoding.

    The first element of a run is kept, and subsequent repetitions are
    represented as ["NC", count].
    """
    if not seq:
        return []
    result: List[Any] = [seq[0]]
    prev = seq[0]
    count = 1
    for item in seq[1:]:
        if item == prev:
            count += 1
        else:
            if count > 1:
                result.append(["NC", count - 1])
            result.append(item)
            prev = item
            count = 1
    if count > 1:
        result.append(["NC", count - 1])
    return result
