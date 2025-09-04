"""Simple run-length encoding utilities."""
from __future__ import annotations

from typing import Iterable, List, Tuple, TypeVar

T = TypeVar("T")


def run_length_encode(seq: Iterable[T]) -> List[Tuple[T, int]]:
    """Run-length encode a sequence."""
    iterator = iter(seq)
    try:
        prev = next(iterator)
    except StopIteration:
        return []
    count = 1
    result: List[Tuple[T, int]] = []
    for item in iterator:
        if item == prev:
            count += 1
        else:
            result.append((prev, count))
            prev = item
            count = 1
    result.append((prev, count))
    return result
