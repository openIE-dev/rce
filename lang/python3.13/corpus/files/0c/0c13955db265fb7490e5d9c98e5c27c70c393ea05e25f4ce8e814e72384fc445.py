# SPDX-License-Identifier: MIT OR Apache-2.0
# The iteration protocol: iter, next, StopIteration, a class with __iter__,
# and the itertools building blocks.
import itertools


class Countdown:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.n <= 0:
            raise StopIteration
        self.n -= 1
        return self.n + 1


print(list(Countdown(4)))
it = iter([1, 2])
print(next(it), next(it), next(it, "done"))
print(list(itertools.islice(itertools.count(10, 5), 4)))
print(list(itertools.chain([1], (2, 3), "ab")))
print(list(itertools.product("ab", repeat=2)), list(itertools.permutations([1, 2, 3], 2)))
print(list(itertools.combinations("abcd", 2)), list(itertools.accumulate([1, 2, 3, 4])))
print([(k, list(g)) for k, g in itertools.groupby("aaabccaa")])
print(list(itertools.zip_longest([1, 2, 3], "ab", fillvalue="-")), list(itertools.pairwise("abcd")))
print(list(map(lambda x, y: x + y, [1, 2], [10, 20])), list(filter(None, [0, 1, "", "a"])))
print(list(reversed(range(3))), sorted(Countdown(3)), max(Countdown(5)), sum(Countdown(4)))
