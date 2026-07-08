from __future__ import annotations

__all__ = ["eval", "__version__"]

__version__ = "0.8.1"


def eval(seq1: object, seq2: object) -> int:
    left = tuple(seq1)
    right = tuple(seq2)

    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (0 if left_item == right_item else 1)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]
