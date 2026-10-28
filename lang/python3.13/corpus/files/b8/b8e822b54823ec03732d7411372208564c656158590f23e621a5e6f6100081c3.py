# SPDX-License-Identifier: MIT OR Apache-2.0
# The towers of Hanoi: the moves for four discs and the count for twenty.


def hanoi(n, src, dst, via, moves):
    if n == 0:
        return
    hanoi(n - 1, src, via, dst, moves)
    moves.append(src + dst)
    hanoi(n - 1, via, dst, src, moves)


moves = []
hanoi(4, "A", "C", "B", moves)
print(len(moves), moves)
big = []
hanoi(20, "A", "C", "B", big)
print(len(big), 2 ** 20 - 1, big[-1])
