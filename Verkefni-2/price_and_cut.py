
from fractions import Fraction as F
from functools import lru_cache

from starter_code import mat_inv
from starter_code import feasible_patterns as _feasible_patterns


@lru_cache(maxsize=None)
def _patterns_cached(weights, W):
    return tuple(_feasible_patterns(list(weights), W))


def feasible_patterns(weights, W):
    """
    Öll lögleg mynstur (sama og í starter_code).
    Reiknað aðeins einu sinni fyrir hvert (weights, W)
    í stað þess að telja upp í hverri ítrun.
    """
    return _patterns_cached(tuple(weights), W)


def frac(x):
    """
    Skilar brothluta x.
    """
    x = F(x)
    return x - (x.numerator // x.denominator)


def is_integer(x):
    """
    Athugar hvort Fraction-gildi sé heiltala.
    """
    x = F(x)
    return x.denominator == 1


def all_integer(values):
    """
    Athugar hvort öll gildi í lista séu heiltölur.
    """
    return all(is_integer(x) for x in values)


def dot(a, b):
    """
    Nákvæm innfeldismargföldun með Fraction.
    """
    return sum(
        (F(x) * F(y) for x, y in zip(a, b)),
        F(0),
    )


def mat_vec_mul(A, x):
    """
    Margfaldar fylki A með vigri x.
    """
    return [dot(row, x) for row in A]


def transpose(A):
    """
    Skilar bylta fylkinu A.
    """
    return [list(row) for row in zip(*A)]


def extend_column(col, cuts, start=0):
    """
    Bætir við stuðlum dálksins í skurðunum cuts[start:].

    Stuðull í skurði j er frac(u_j · dálkur), reiknaður af
    dálkinum eins og hann leit út þegar skurður j var myndaður
    (u_j hefur einn þátt fyrir hverja pöntun og hvern eldri skurð).
    """
    col = list(col)

    for cut in cuts[start:]:
        u = cut["u"]
        coeff = frac(dot(u, col[:len(u)]))
        col.append(coeff)

    return col


def pattern_column(pattern, cuts):
    """
    Byggir allan dálkinn fyrir mynstur.
    Fyrst koma demand-stuðlarnir og svo
    einn stuðull fyrir hvern Gomory-skurð.
    """
    return extend_column(
        [F(x) for x in pattern],
        cuts,
    )


def surplus_column(k, n_demands, cuts):
    """
    Byggir allan dálkinn fyrir surplus-breytu skurðar k (1-vísað).

    0 í demand-röðum og eldri skurðum, -1 í eigin röð, og
    frac(u_j · dálkur) í öllum yngri skurðum j > k.

    Það síðasta skiptir máli: ef s_k er utan grunns þegar
    skurður j er myndaður getur stuðullinn verið brotinn, og þá
    verður skurður j að taka tillit til s_k eins og hvers annars
    dálks (sama hugmynd og í kafla 2.2 fyrir mynstur).
    """
    col = [
        F(0)
        for _ in range(n_demands + k - 1)
    ] + [F(-1)]

    return extend_column(
        col,
        cuts,
        start=k,
    )


def solve_master(B, rhs, cB, n_demands):
    """
    Leysir núverandi grunn master-vandans.
    """

    B = [[F(x) for x in row] for row in B]
    rhs = [F(x) for x in rhs]
    cB = [F(x) for x in cB]

    B_inv = mat_inv(B)

    xB = mat_vec_mul(B_inv, rhs)

    pi = mat_vec_mul(
        transpose(B_inv),
        cB,
    )

    y = pi[:n_demands]
    mu = pi[n_demands:]

    objective = dot(cB, xB)

    return {
        "B_inv": B_inv,
        "xB": xB,
        "y": y,
        "mu": mu,
        "objective": objective,
    }


def price_column(weights, W, y, mu=None, cuts=None):
    """
    Finnur besta nýja skurðarmynstrið miðað við
    núverandi nykurverð.
    """

    cuts = cuts or []
    mu = mu or []

    if len(mu) != len(cuts):
        raise ValueError(
            "Fjöldi mu-gilda þarf að passa við fjölda skurða."
        )

    n = len(weights)

    # Án skurða er þetta venjulegt óbundið bakpokaverkefni.
    if not cuts:

        best_value = [
            F(0)
            for _ in range(W + 1)
        ]

        best_pattern = [
            tuple(0 for _ in range(n))
            for _ in range(W + 1)
        ]

        for capacity in range(1, W + 1):

            # Það má skilja eftir ónotað pláss.
            best_value[capacity] = best_value[capacity - 1]
            best_pattern[capacity] = best_pattern[capacity - 1]

            for i, w in enumerate(weights):

                if w <= capacity:

                    value = (
                        best_value[capacity - w]
                        + F(y[i])
                    )

                    if value > best_value[capacity]:

                        pattern = list(
                            best_pattern[capacity - w]
                        )

                        pattern[i] += 1

                        best_value[capacity] = value
                        best_pattern[capacity] = tuple(pattern)

        pattern = best_pattern[W]
        value = best_value[W]
        column = pattern_column(pattern, cuts)

    # Þegar skurðir eru virkir er frac-liðurinn ekki samleggjandi.
    # Þá förum við yfir öll lögleg mynstur.
    else:

        pattern = None
        column = None
        value = None

        for candidate in feasible_patterns(
            list(weights),
            W,
        ):

            candidate_col = pattern_column(
                candidate,
                cuts,
            )

            demand_part = dot(
                y,
                candidate,
            )

            cut_part = dot(
                mu,
                candidate_col[n:],
            )

            candidate_value = (
                demand_part
                + cut_part
            )

            if (
                value is None
                or candidate_value > value
            ):
                pattern = candidate
                column = candidate_col
                value = candidate_value

    return {
        "pattern": pattern,
        "column": column,
        "value": value,
        "improves": value > 1,
    }


def derive_cut(B_inv, xB):
    """
    Finnur þá brotnu grunnbreytu sem hefur stærstan
    brothluta og býr til Gomory-skurð úr þeirri röð.
    (Algeng regla sem gefur yfirleitt dýpri skurði en
    að taka einfaldlega fyrstu brotnu röðina.)
    """

    best_row = None
    best_frac = None

    for row, value in enumerate(xB):

        if not is_integer(value):

            f = frac(value)

            # Við jafntefli höldum við fyrstu röðinni.
            if best_frac is None or f > best_frac:
                best_row = row
                best_frac = f

    if best_row is None:
        return None

    # Röðin úr B^-1 er fryst og fylgir skurðinum áfram.
    u = tuple(
        F(x)
        for x in B_inv[best_row]
    )

    return {
        "row": best_row,
        "u": u,
        "rhs": best_frac,
    }


def pivot_in_column(
    B,
    cB,
    basis_vars,
    B_inv,
    xB,
    entering_column,
    entering_var,
    entering_cost=1,
):
    """
    Setur nýjan dálk inn í grunninn með
    venjulegu lágmarks-hlutfallsprófi.
    """

    a_bar = mat_vec_mul(
        B_inv,
        entering_column,
    )

    ratios = []

    for i, coeff in enumerate(a_bar):

        if coeff > 0:

            ratios.append(
                (
                    i,
                    xB[i] / coeff,
                )
            )

    if not ratios:
        raise ValueError(
            "Engin lögleg vendilína fannst."
        )

    leaving_row, _ = min(
        ratios,
        key=lambda item: (
            item[1],
            item[0],
        ),
    )

    leaving_var = basis_vars[leaving_row]

    B_new = [
        row[:]
        for row in B
    ]

    for r in range(len(B_new)):
        B_new[r][leaving_row] = F(
            entering_column[r]
        )

    cB_new = list(cB)
    cB_new[leaving_row] = F(
        entering_cost
    )

    basis_new = list(basis_vars)
    basis_new[leaving_row] = entering_var

    return {
        "B": B_new,
        "cB": cB_new,
        "basis_vars": basis_new,
        "a_bar": a_bar,
        "ratios": ratios,
        "leaving_row": leaving_row,
        "leaving_var": leaving_var,
    }


def add_cut_to_basis(
    B,
    rhs,
    cB,
    basis_vars,
    cut,
    cut_number,
):
    """
    Bætir Gomory-skurði við grunninn.
    Ný surplus-breyta verður grunnbreyta fyrst.
    """

    n = len(B)

    # B fær eina nýja röð og einn nýjan dálk.
    # Allar núverandi grunnbreytur hafa stuðul 0 í nýja
    # skurðinum (B^-1 B = I gefur heiltölu, frac = 0).
    B_new = [
        [F(x) for x in row] + [F(0)]
        for row in B
    ]

    # Skurðurinn er >=, þannig að surplus-breytan
    # kemur inn með stuðul -1.
    new_row = [
        F(0)
        for _ in range(n)
    ] + [F(-1)]

    B_new.append(new_row)

    rhs_new = [
        F(x)
        for x in rhs
    ] + [F(cut["rhs"])]

    # Surplus-breytan kostar ekkert.
    cB_new = [
        F(x)
        for x in cB
    ] + [F(0)]

    basis_new = list(basis_vars) + [
        ("surplus", cut_number)
    ]

    return {
        "B": B_new,
        "rhs": rhs_new,
        "cB": cB_new,
        "basis_vars": basis_new,
    }


def dual_pivot(
    B,
    cB,
    basis_vars,
    master,
    weights,
    W,
    cuts,
):
    """
    Gerir eina dual-simplex vendingu þegar
    einhver grunnbreyta er neikvæð.
    """

    B_inv = master["B_inv"]
    xB = master["xB"]
    y = master["y"]
    mu = master["mu"]

    negative_rows = [
        i
        for i, value in enumerate(xB)
        if value < 0
    ]

    if not negative_rows:
        return None

    leaving_row = negative_rows[0]
    leaving_var = basis_vars[leaving_row]

    candidates = []
    basis_set = set(basis_vars)

    # Prófum öll lögleg mynstur sem mögulegan vendidálk.
    for pattern in feasible_patterns(
        list(weights),
        W,
    ):

        var = (
            "pattern",
            pattern,
        )

        if var in basis_set:
            continue

        column = pattern_column(
            pattern,
            cuts,
        )

        a_bar = mat_vec_mul(
            B_inv,
            column,
        )

        coeff = a_bar[leaving_row]

        if coeff < 0:

            reduced_cost = (
                F(1)
                - dot(y, pattern)
                - dot(
                    mu,
                    column[len(weights):],
                )
            )

            ratio = (
                reduced_cost
                / (-coeff)
            )

            candidates.append({
                "var": var,
                "column": column,
                "a_bar": a_bar,
                "reduced_cost": reduced_cost,
                "ratio": ratio,
                "cost": F(1),
            })

    # Gömul surplus-breyta getur líka komið aftur inn.
    for k in range(len(cuts)):

        var = (
            "surplus",
            k + 1,
        )

        if var in basis_set:
            continue

        # Fullur dálkur, með stuðlum í yngri skurðum.
        column = surplus_column(
            k + 1,
            len(weights),
            cuts,
        )

        a_bar = mat_vec_mul(
            B_inv,
            column,
        )

        coeff = a_bar[leaving_row]

        if coeff < 0:

            # Kostnaður 0, svo minnkaði kostnaðurinn er -pi · dálkur.
            reduced_cost = -dot(
                mu,
                column[len(weights):],
            )

            ratio = (
                reduced_cost
                / (-coeff)
            )

            candidates.append({
                "var": var,
                "column": column,
                "a_bar": a_bar,
                "reduced_cost": reduced_cost,
                "ratio": ratio,
                "cost": F(0),
            })

    if not candidates:
        raise ValueError(
            "Enginn löglegur vendidálkur fannst í dual simplex."
        )

    entering = min(
        candidates,
        key=lambda c: c["ratio"],
    )

    B_new = [
        row[:]
        for row in B
    ]

    for r in range(len(B_new)):
        B_new[r][leaving_row] = (
            entering["column"][r]
        )

    cB_new = list(cB)
    cB_new[leaving_row] = (
        entering["cost"]
    )

    basis_new = list(basis_vars)
    basis_new[leaving_row] = (
        entering["var"]
    )

    return {
        "B": B_new,
        "cB": cB_new,
        "basis_vars": basis_new,
        "leaving_row": leaving_row,
        "leaving_var": leaving_var,
        "entering_var": entering["var"],
        "a_bar": entering["a_bar"],
        "reduced_cost": entering["reduced_cost"],
        "ratio": entering["ratio"],
    }


def find_entering_surplus(basis_vars, mu, cuts, n_demands):
    """
    Finnur surplus-breytu utan grunns með neikvæðan
    minnkaðan kostnað (ef einhver er).
    """

    basis_set = set(basis_vars)

    for k in range(1, len(cuts) + 1):

        if ("surplus", k) in basis_set:
            continue

        column = surplus_column(
            k,
            n_demands,
            cuts,
        )

        reduced_cost = -dot(
            mu,
            column[n_demands:],
        )

        if reduced_cost < 0:
            return k, column

    return None


def price_and_cut(
    weights,
    demands,
    W,
    max_iter=200,
):
    """
    Leysir skurðarvandann með price-and-cut:
    dálkamyndun + Gomory-skurðum.
    """

    weights = list(weights)
    demands = list(demands)

    m = len(weights)

    if len(demands) != m:
        raise ValueError(
            "weights og demands þurfa að vera jafn langir."
        )

    if (
        any(w <= 0 for w in weights)
        or W <= 0
    ):
        raise ValueError(
            "Breiddir og W þurfa að vera jákvæð."
        )

    if any(w > W for w in weights):
        raise ValueError(
            "Allar breiddir þurfa að komast fyrir á rúllunni."
        )

    # Einfaldur upphafsgrunnur.
    diagonal = [
        W // w
        for w in weights
    ]

    B = [
        [
            F(diagonal[j])
            if i == j
            else F(0)
            for j in range(m)
        ]
        for i in range(m)
    ]

    rhs = [
        F(d)
        for d in demands
    ]

    cB = [
        F(1)
        for _ in range(m)
    ]

    basis_vars = []

    for i in range(m):

        pattern = tuple(
            diagonal[i]
            if j == i
            else 0
            for j in range(m)
        )

        basis_vars.append(
            (
                "pattern",
                pattern,
            )
        )

    cuts = []

    n_pricing_iterations = 0
    n_dual_pivots = 0

    for _ in range(max_iter):

        master = solve_master(
            B=B,
            rhs=rhs,
            cB=cB,
            n_demands=m,
        )

        # Eftir nýjan skurð getur grunnurinn verið ólöglegur.
        if any(
            value < 0
            for value in master["xB"]
        ):

            pivot = dual_pivot(
                B=B,
                cB=cB,
                basis_vars=basis_vars,
                master=master,
                weights=weights,
                W=W,
                cuts=cuts,
            )

            if pivot is None:
                raise RuntimeError(
                    "Ekki tókst að laga ólöglegan grunn."
                )

            B = pivot["B"]
            cB = pivot["cB"]
            basis_vars = pivot["basis_vars"]

            n_dual_pivots += 1
            continue

        pricing = price_column(
            weights=weights,
            W=W,
            y=master["y"],
            mu=master["mu"],
            cuts=cuts,
        )

        # Ef nýr dálkur borgar sig fer hann inn í grunninn.
        if pricing["improves"]:

            pivot = pivot_in_column(
                B=B,
                cB=cB,
                basis_vars=basis_vars,
                B_inv=master["B_inv"],
                xB=master["xB"],
                entering_column=pricing["column"],
                entering_var=(
                    "pattern",
                    pricing["pattern"],
                ),
                entering_cost=1,
            )

            B = pivot["B"]
            cB = pivot["cB"]
            basis_vars = pivot["basis_vars"]

            n_pricing_iterations += 1
            continue

        # Surplus-breyta utan grunns með neikvæðan minnkaðan
        # kostnað á líka að fara inn áður en LP telst leyst.
        entering_surplus = find_entering_surplus(
            basis_vars=basis_vars,
            mu=master["mu"],
            cuts=cuts,
            n_demands=m,
        )

        if entering_surplus is not None:

            k, column = entering_surplus

            pivot = pivot_in_column(
                B=B,
                cB=cB,
                basis_vars=basis_vars,
                B_inv=master["B_inv"],
                xB=master["xB"],
                entering_column=column,
                entering_var=(
                    "surplus",
                    k,
                ),
                entering_cost=0,
            )

            B = pivot["B"]
            cB = pivot["cB"]
            basis_vars = pivot["basis_vars"]

            continue

        # Ef ekkert mynstur borgar sig og lausnin er heil erum við búin.
        if all_integer(master["xB"]):

            solution = {}

            for var, value in zip(
                basis_vars,
                master["xB"],
            ):

                if (
                    var[0] == "pattern"
                    and value > 0
                ):

                    pattern = var[1]

                    solution[pattern] = (
                        solution.get(
                            pattern,
                            0,
                        )
                        + int(value)
                    )

            total_rolls = sum(
                solution.values()
            )

            return {
                "solution": solution,
                "total_rolls": total_rolls,
                "n_pricing_iterations": n_pricing_iterations,
                "n_cuts": len(cuts),
                "cuts": cuts,
                "n_dual_pivots": n_dual_pivots,
                "xB": master["xB"],
                "y": master["y"],
                "mu": master["mu"],
            }

        # Enginn dálkur borgar sig en lausnin er brotin.
        # Þá bætum við við Gomory-skurði.
        cut = derive_cut(
            B_inv=master["B_inv"],
            xB=master["xB"],
        )

        if cut is None:
            raise RuntimeError(
                "Lausnin er brotin en enginn skurður fannst."
            )

        cuts.append(cut)

        cut_state = add_cut_to_basis(
            B=B,
            rhs=rhs,
            cB=cB,
            basis_vars=basis_vars,
            cut=cut,
            cut_number=len(cuts),
        )

        B = cut_state["B"]
        rhs = cut_state["rhs"]
        cB = cut_state["cB"]
        basis_vars = cut_state["basis_vars"]

    raise RuntimeError(
        f"price_and_cut náði max_iter={max_iter} án þess að klára."
    )
