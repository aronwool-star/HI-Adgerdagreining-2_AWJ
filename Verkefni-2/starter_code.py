from __future__ import annotations
from fractions import Fraction as F
from typing import List


def mat_inv(M: List[List[F]]) -> List[List[F]]:
    """Naekvaem fylkjaandhverfa (Gauss-Jordan) yfir fractions.Fraction."""
    n = len(M)
    aug = [row[:] + [F(1) if i == j else F(0) for j in range(n)] for i, row in enumerate(M)]
    for col in range(n):
        piv = next(r for r in range(col, n) if aug[r][col] != 0)
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [x / pv for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [x - f * y for x, y in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def feasible_patterns(weights: List[int], W: int):
    """Oll logleg skurdarmynstur (tamandi talin) -- gagnlegt fyrir bakpokaleit
    med skurdum virkum (sja kafla 2.4 i verkefnislysingu), og fyrir verify.py."""
    n = len(weights)
    pats = []

    def rec(i, cur, rem):
        if i == n:
            if any(cur):
                pats.append(tuple(cur))
            return
        for k in range(rem // weights[i] + 1):
            cur.append(k)
            rec(i + 1, cur, rem - k * weights[i])
            cur.pop()

    rec(0, [], W)
    return pats


# --- Gagnasett  ------------------------------

MAIN_INSTANCE = dict(weights=[7, 5, 3], demands=[12, 9, 7], W=10)
BONUS_INSTANCE = dict(weights=[9, 7, 5, 3], demands=[8, 10, 6, 14], W=16)


if __name__ == "__main__":
    B = [[F(2), F(0)], [F(0), F(4)]]
    print("mat_inv smoke test:", mat_inv(B))

    main_patterns = feasible_patterns(
        **{k: v for k, v in BONUS_INSTANCE.items() if k != "demands"}
    )

    print("feasible_patterns(MAIN_INSTANCE):", len(main_patterns))

    print("\nAll feasible patterns:")
    for i, pattern in enumerate(main_patterns, start=1):
        print(f"{i}: {pattern}")