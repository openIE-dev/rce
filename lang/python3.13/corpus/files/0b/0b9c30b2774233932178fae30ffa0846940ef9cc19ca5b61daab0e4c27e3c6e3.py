# SPDX-License-Identifier: MIT OR Apache-2.0
# Generated methods of dataclasses: repr, equality, ordering, defaults,
# frozen instances and replace.
from dataclasses import FrozenInstanceError, asdict, dataclass, field, replace


@dataclass(order=True)
class Item:
    priority: int
    name: str = field(compare=False)
    tags: list = field(default_factory=list, repr=False)


@dataclass(frozen=True)
class P:
    x: float
    y: float = 0.0


items = [Item(3, "c"), Item(1, "a"), Item(2, "b"), Item(1, "z")]
print(sorted(items))
print(Item(1, "a") == Item(1, "other"), Item(1, "a") < Item(2, "a"), asdict(Item(5, "e", ["t"])))
p = P(1.5)
print(p, replace(p, y=-0.0), p == P(1.5, 0.0), hash(p) == hash(P(1.5)))
try:
    p.x = 2
except FrozenInstanceError as e:
    print("FrozenInstanceError", e)
