from __future__ import annotations

from math import ceil
from typing import Any


def nonnegative_int(value: Any) -> int:
    if value is None:
        return 0
    return max(0, ceil(float(value)))


def floor_log2(value: int) -> int:
    if value <= 0:
        raise ValueError('floor_log2 requires a positive integer')
    return value.bit_length() - 1


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError('alignment must be positive')
    return (value + alignment - 1) // alignment * alignment
