from fractions import Fraction as F

import price_and_cut as pac


# ---------------------------------------------------------
# Gagnasett
# ---------------------------------------------------------

MAIN_INSTANCE = {
    "weights": [7, 5, 3],
    "demands": [12, 9, 7],
    "W": 10,
}

BONUS_INSTANCE = {
    "weights": [9, 7, 5, 3],
    "demands": [8, 10, 6, 14],
    "W": 16,
}


# ---------------------------------------------------------
# Við breytum EKKI solve_master.
# Við geymum upprunalega fallið og vefjum bara logging utan um það.
# ---------------------------------------------------------

original_solve_master = pac.solve_master

degeneracy_log = []


def solve_master_with_log(*args, **kwargs):
    result = original_solve_master(*args, **kwargs)

    xB = result["xB"]

    zero_rows = [
        i
        for i, value in enumerate(xB)
        if value == F(0)
    ]

    degeneracy_log.append({
        "xB": xB[:],
        "zero_rows": zero_rows,
        "degenerate": bool(zero_rows),
    })

    return result


pac.solve_master = solve_master_with_log


# ---------------------------------------------------------
# Keyrsla
# ---------------------------------------------------------

def run_test(name, instance):
    degeneracy_log.clear()

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    result = pac.price_and_cut(**instance)

    print()
    print("Lokaniðurstaða:")
    print(f"  total_rolls = {result['total_rolls']}")
    print(f"  n_cuts = {result['n_cuts']}")
    print(f"  n_pricing_iterations = {result['n_pricing_iterations']}")
    print(f"  n_dual_pivots = {result['n_dual_pivots']}")

    print()
    print("Grunnlausnir:")

    for i, entry in enumerate(degeneracy_log, start=1):

        status = (
            "ÚRKYNJUÐ"
            if entry["degenerate"]
            else "ekki úrkynjuð"
        )

        print(
            f"{i:2d}. "
            f"xB = {[str(x) for x in entry['xB']]} "
            f"-> {status}"
        )

        if entry["degenerate"]:
            print(
                f"    grunnbreytur með gildi 0: "
                f"{entry['zero_rows']}"
            )

    degenerate_steps = [
        (i, entry)
        for i, entry in enumerate(
            degeneracy_log,
            start=1,
        )
        if entry["degenerate"]
    ]

    print()
    print("Samantekt:")
    print(
        f"  Fjöldi solve_master keyrslna: "
        f"{len(degeneracy_log)}"
    )

    print(
        f"  Fjöldi úrkynjaðra grunnlausna: "
        f"{len(degenerate_steps)}"
    )

    if degenerate_steps:
        print(
            "  Úrkynjun kom upp í solve_master keyrslum:",
            [i for i, _ in degenerate_steps],
        )
    else:
        print("  Engin úrkynjun kom upp.")


if __name__ == "__main__":

    run_test(
        "AÐALGAGNASETT",
        MAIN_INSTANCE,
    )

    run_test(
        "AUKAGAGNASETT",
        BONUS_INSTANCE,
    )