# SPDX-License-Identifier: MIT OR Apache-2.0
# Structural pattern matching: literals, captures, sequences, mappings,
# classes, guards, or-patterns and the wildcard.
from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int


def describe(v):
    match v:
        case 0 | 1:
            return "small int"
        case int(n) if n < 0:
            return f"negative {n}"
        case int():
            return "int"
        case str() as s if s.startswith("#"):
            return "comment"
        case [x, y]:
            return f"pair {x} {y}"
        case [first, *rest]:
            return f"seq starting {first} with {len(rest)} more"
        case {"type": "circle", "r": r}:
            return f"circle {r}"
        case Point(x=0, y=0):
            return "origin"
        case Point(x=px, y=py):
            return f"point {px},{py}"
        case None:
            return "none"
        case _:
            return "other " + type(v).__name__


for v in [0, 1, -4, 9, "#x", "y", [1, 2], [1, 2, 3], (5,), {"type": "circle", "r": 2, "z": 0}, Point(0, 0), Point(1, 2), None, 2.5]:
    print(repr(v), describe(v))
