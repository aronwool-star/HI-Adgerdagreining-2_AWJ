"""
verify.py -- Óháð sannprófun á niðurstöðu price_and_cut.

Hugmyndin er að staðfesta svarið eftir ÖÐRUM leiðum en reikniferlið notar.
Ekkert hér er flutt inn úr price_and_cut.py nema sjálft fallið price_and_cut,
og ekkert úr starter_code.py: mynstrin eru talin upp upp á nýtt, brothlutinn
er reiknaður með math.floor, og heiltöluhámarkið er fundið með alveg
annarri aðferð (kvikri bestun yfir pantanavigra).

Prófanir:
  1. Lokalausnin er lögleg: hvert mynstur kemst á rúlluna, fjöldi er
     jákvæð heiltala og allar pantanir eru uppfylltar.
  2. total_rolls = summa x_j í lausninni.
  3. Lokanykurverðin (y, mu) eru tvíleg-lögleg: ekkert af ÖLLUM löglegum
     mynstrum hefur jákvæðan minnkaðan kostnað (kafli 2.4) og mu_k >= 0.
  4. Heiltölulausnin uppfyllir alla Gomory-skurðina sem voru myndaðir.
  5. Sterk tvíun: d·y + sum_k mu_k * rhs_k = total_rolls.  Ásamt (3) er
     þetta vottorð um að lausnin sé best fyrir LP-ið með skurðunum.
  6. Neðra mark: LP-slökun yfir öll mynstur (leyst sjálfstætt) gefur
     ceil(z_LP) <= total_rolls.  Ef jafnt er lausnin sönnuð best.
  7. Nákvæmt heiltöluhámark með kvikri bestun (DP) yfir pantanavigra --
     engir utanaðkomandi pakkar, allt önnur aðferð en dálkamyndun.
  8. (Valfrjálst) krossprófun með scipy.optimize.milp ef scipy er til.
"""

from __future__ import annotations

import math
import sys
import time
from fractions import Fraction as F
from functools import lru_cache
from itertools import product

from price_and_cut import price_and_cut


MAIN_INSTANCE = dict(weights=[7, 5, 3], demands=[12, 9, 7], W=10)
BONUS_INSTANCE = dict(weights=[9, 7, 5, 3], demands=[8, 10, 6, 14], W=16)

# Hámarksfjöldi staða í DP-inu (prófun 7); stærri vandamálum er sleppt.
DP_STATE_LIMIT = 2_000_000


# ---------------------------------------------------------------------------
# Hjálparföll (skrifuð upp á nýtt, óháð price_and_cut.py / starter_code.py)
# ---------------------------------------------------------------------------

def enumerate_patterns(weights, W):
    """Öll lögleg mynstur (a != 0) með itertools.product -- önnur aðferð
    en endurkvæma talningin í starter_code.feasible_patterns."""
    ranges = [range(W // w + 1) for w in weights]
    return [
        a for a in product(*ranges)
        if any(a) and sum(w * k for w, k in zip(weights, a)) <= W
    ]


def frac_part(x):
    x = F(x)
    return x - math.floor(x)


def cut_coefficients(cuts, pattern):
    """Stuðlar mynstursins í öllum skurðunum, í röð.

    Frysti vigurinn u_k er röð í B^-1 á þeim tíma sem skurður k var myndaður,
    svo hann hefur m + (k-1) þætti: einn fyrir hverja pöntunarskorðu OG einn
    fyrir hvern eldri skurð. Staða mynstursins í þeim grunni er því
    u_k · (a, c_1, ..., c_{k-1}), þar sem c_j er stuðull a í skurði j."""
    col = [F(x) for x in pattern]
    coeffs = []
    for c in cuts:
        u = [F(v) for v in c["u"]]
        if len(u) != len(col):
            raise ValueError(f"lengd u ({len(u)}) passar ekki við dálkinn ({len(col)})")
        ck = frac_part(sum(ui * ai for ui, ai in zip(u, col)))
        coeffs.append(ck)
        col.append(ck)
    return coeffs


def surplus_cut_coefficients(cuts, j, n_demands):
    """Stuðlar surplus-breytu s_j (1-vísað) í öllum skurðum.
    -1 í eigin skurði, 0 í eldri skurðum, og frac(u_k · dálkur) í yngri
    skurðum k > j (sama skilgreining og fyrir mynstur, bara annar dálkur)."""
    col = [F(0)] * (n_demands + j - 1) + [F(-1)]
    coeffs = [F(0)] * (j - 1) + [F(-1)]
    for c in cuts[j:]:
        u = [F(v) for v in c["u"]]
        ck = frac_part(sum(ui * ai for ui, ai in zip(u, col)))
        coeffs.append(ck)
        col.append(ck)
    return coeffs


class Report:
    def __init__(self, name):
        self.name = name
        self.ok = True

    def check(self, cond, msg):
        print(f"  [{'OK ' if cond else 'VILLA'}] {msg}")
        if not cond:
            self.ok = False
        return cond

    def info(self, msg):
        print(f"  [ .. ] {msg}")


# ---------------------------------------------------------------------------
# Sjálfstæð LP-slökun (prófun 6): nákvæmur tveggja fasa Simplex með Fraction
# yfir ÖLL mynstur í einu -- engin dálkamyndun, engin verðlagning.
# ---------------------------------------------------------------------------

def lp_relaxation_exact(weights, demands, W):
    """min sum x_j  m.t.t.  sum_j a_ij x_j >= d_i,  x >= 0.
    Leyst sem tvíverkefnið:  max d·y  m.t.t.  a·y <= 1 fyrir öll mynstur,
    y >= 0.  Þá er upphafsgrunnurinn slakbreyturnar (löglegur, b = 1 >= 0)
    og aðeins þarf einn fasa af Bland-Simplex."""
    pats = enumerate_patterns(weights, W)
    m, n = len(weights), len(pats)
    # Tafla: n línur (ein fyrir hvert mynstur), m + n dálkar + hægri hlið.
    T = []
    for j, a in enumerate(pats):
        row = [F(a[i]) for i in range(m)] + [F(1 if k == j else 0) for k in range(n)] + [F(1)]
        T.append(row)
    obj = [F(-d) for d in demands] + [F(0)] * n + [F(0)]   # max d·y -> lágmarka -d·y
    basis = [m + j for j in range(n)]

    while True:
        # Bland-regla: minnsti vísir með neikvæðan kostnað fer inn (engin hringrás).
        enter = next((c for c in range(m + n) if obj[c] < 0), None)
        if enter is None:
            break
        best = None
        for r in range(n):
            if T[r][enter] > 0:
                ratio = T[r][-1] / T[r][enter]
                if best is None or ratio < best[0] or (ratio == best[0] and basis[r] < basis[best[1]]):
                    best = (ratio, r)
        if best is None:
            raise RuntimeError("LP-ið er ótakmarkað (á ekki að geta gerst).")
        r = best[1]
        pv = T[r][enter]
        T[r] = [x / pv for x in T[r]]
        for rr in range(n):
            if rr != r and T[rr][enter] != 0:
                f = T[rr][enter]
                T[rr] = [x - f * y for x, y in zip(T[rr], T[r])]
        f = obj[enter]
        obj = [x - f * y for x, y in zip(obj, T[r])]
        basis[r] = enter

    return obj[-1]   # = max d·y = z_LP (sterk tvíun)


# ---------------------------------------------------------------------------
# Nákvæmt heiltöluhámark (prófun 7): kvik bestun yfir eftirstandandi pantanir.
#   f(d) = 0                              ef d = 0
#   f(d) = 1 + min_a f(max(d - a, 0))    annars
# Þetta er skilgreiningin á skurðarvandanum beint, óháð LP og skurðum.
# ---------------------------------------------------------------------------

def integer_optimum_dp(weights, demands, W):
    n_states = math.prod(d + 1 for d in demands)
    if n_states > DP_STATE_LIMIT:
        return None
    pats = enumerate_patterns(weights, W)
    # Aðeins "hámarks" mynstur skipta máli (hægt að bæta við stykki = aldrei verra).
    maximal = [
        a for a in pats
        if not any(
            sum(w * k for w, k in zip(weights, a)) + weights[i] <= W
            for i in range(len(weights))
        )
    ]
    sys.setrecursionlimit(max(10_000, sum(demands) * 10 + 1000))

    @lru_cache(maxsize=None)
    def f(rem):
        if not any(rem):
            return 0
        best = math.inf
        for a in maximal:
            if any(a[i] > 0 and rem[i] > 0 for i in range(len(rem))):
                nxt = tuple(max(r - k, 0) for r, k in zip(rem, a))
                v = f(nxt)
                if v < best:
                    best = v
        return 1 + best

    return f(tuple(demands))


# ---------------------------------------------------------------------------
# Valfrjálst (prófun 8): scipy/HiGHS sem óháð krossprófun.
# ---------------------------------------------------------------------------

def integer_optimum_scipy(weights, demands, W):
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError:
        return None
    pats = enumerate_patterns(weights, W)
    A = np.array(pats, dtype=float).T
    res = milp(
        c=np.ones(len(pats)),
        constraints=LinearConstraint(A, lb=np.array(demands, dtype=float), ub=np.inf),
        integrality=np.ones(len(pats)),
        bounds=Bounds(0, np.inf),
    )
    return round(res.fun) if res.success else None


# ---------------------------------------------------------------------------
# Aðalsannprófun
# ---------------------------------------------------------------------------

def verify(weights, demands, W, result=None, name="gagnasett"):
    weights, demands = list(weights), list(demands)
    m = len(weights)
    rep = Report(name)
    print(f"\n=== {name}: w={tuple(weights)}, d={tuple(demands)}, W={W} ===")

    if result is None:
        t0 = time.perf_counter()
        result = price_and_cut(weights, demands, W)
        rep.info(f"price_and_cut keyrði á {time.perf_counter() - t0:.3f} s")

    sol = result["solution"]
    total = result["total_rolls"]
    y = [F(v) for v in result["y"]]
    mu = [F(v) for v in result["mu"]]
    cuts = result["cuts"]
    rep.info(f"total_rolls={total}, n_cuts={result['n_cuts']}, "
             f"n_pricing_iterations={result['n_pricing_iterations']}")

    all_pats = enumerate_patterns(weights, W)
    all_set = set(all_pats)
    rep.info(f"{len(all_pats)} lögleg mynstur talin upp sjálfstætt")

    # 1. Lögmæti lausnar
    rep.check(all(tuple(p) in all_set for p in sol),
              "öll mynstur í lausninni eru lögleg (sum w_i a_i <= W)")
    rep.check(all(isinstance(x, int) and x > 0 for x in sol.values()),
              "allir fjöldar x_j eru jákvæðar heiltölur")
    produced = [sum(p[i] * x for p, x in sol.items()) for i in range(m)]
    rep.check(all(produced[i] >= demands[i] for i in range(m)),
              f"allar pantanir uppfylltar: framleitt {produced} >= {demands}")

    # 2. total_rolls
    rep.check(sum(sol.values()) == total,
              f"summa x_j = {sum(sol.values())} passar við total_rolls = {total}")
    rep.check(result["n_cuts"] == len(cuts), "n_cuts passar við lengd cuts-listans")

    # 3. Tvíleg lögmæti lokanykurverða yfir ÖLL mynstur
    rep.check(len(mu) == len(cuts), "eitt mu-gildi fyrir hvern skurð")
    rep.check(all(v >= 0 for v in mu), f"mu_k >= 0 (mu = {[str(v) for v in mu]})")
    worst_pat, worst_rc = None, None
    for a in all_pats:
        rc = 1 - sum(yi * ai for yi, ai in zip(y, a)) \
               - sum(mk * ck for mk, ck in zip(mu, cut_coefficients(cuts, a)))
        if worst_rc is None or rc < worst_rc:
            worst_pat, worst_rc = a, rc
    rep.check(worst_rc >= 0,
              f"ekkert mynstur hefur jákvæðan minnkaðan kostnað "
              f"(minnsti minnkaði kostnaður = {worst_rc} í {worst_pat})")
    # Surplus-breytur skurða eru líka breytur í LP-inu; kostnaður þeirra er 0.
    rc_surplus = [
        -sum(mk * ck for mk, ck in zip(mu, surplus_cut_coefficients(cuts, j, m)))
        for j in range(1, len(cuts) + 1)
    ]
    rep.check(all(r >= 0 for r in rc_surplus),
              "engin surplus-breyta hefur jákvæðan minnkaðan kostnað")

    # 4. Heiltölulausnin uppfyllir alla skurði.
    # Skurður k:  sum_j c_kj x_j + sum_{i<k} g_ki s_i - s_k = rhs_k.
    # Við reiknum surplus-gildin s_k í röð út frá lausninni og athugum að
    # þau séu >= 0 (skurðurinn uppfylltur) og heiltölur (Gomory-eiginleiki:
    # surplus fractional-skurðar er alltaf heiltala í heiltölulausn).
    coeffs_sol = {p: cut_coefficients(cuts, p) for p in sol}
    s_coeffs = [surplus_cut_coefficients(cuts, j, m) for j in range(1, len(cuts) + 1)]
    s_vals = []
    for k, c in enumerate(cuts):
        lhs = sum(coeffs_sol[p][k] * x for p, x in sol.items()) \
            + sum(s_coeffs[i][k] * s_vals[i] for i in range(k))
        s_k = lhs - F(c["rhs"])
        s_vals.append(s_k)
        rep.check(s_k >= 0 and s_k.denominator == 1,
                  f"skurður {k + 1}: uppfylltur, surplus s = {s_k} (>= 0 og heiltala)")

    # 5. Sterk tvíun
    dual_obj = sum(F(d) * yi for d, yi in zip(demands, y)) \
             + sum(mk * F(c["rhs"]) for mk, c in zip(mu, cuts))
    rep.check(dual_obj == total,
              f"sterk tvíun: d·y + sum mu_k rhs_k = {dual_obj} = total_rolls")

    # 6. LP-neðra mark, leyst sjálfstætt yfir öll mynstur
    z_lp = lp_relaxation_exact(weights, demands, W)
    lb = math.ceil(z_lp)
    rep.check(lb <= total, f"LP-slökun z_LP = {z_lp} = {float(z_lp):.4g}, ceil = {lb} <= {total}")
    if lb == total:
        rep.info("ceil(z_LP) = total_rolls  =>  lausnin er SÖNNUÐ best")

    # 7. Nákvæmt heiltöluhámark með DP
    opt = integer_optimum_dp(weights, demands, W)
    if opt is None:
        rep.info("DP sleppt (of margar stöður)")
    else:
        rep.check(opt == total, f"DP-heiltöluhámark = {opt} = total_rolls")

    # 8. scipy.milp (valfrjálst)
    opt_s = integer_optimum_scipy(weights, demands, W)
    if opt_s is None:
        rep.info("scipy ekki til staðar -- milp-krossprófun sleppt")
    else:
        rep.check(opt_s == total, f"scipy.milp heiltöluhámark = {opt_s} = total_rolls")

    print(f"  --> {'ALLT STAÐFEST' if rep.ok else 'SANNPRÓFUN MISTÓKST'}")
    return rep.ok


if __name__ == "__main__":
    ok = True
    ok &= verify(**MAIN_INSTANCE, name="Aðalgagnasett")
    ok &= verify(**BONUS_INSTANCE, name="Aukagagnasett")
    sys.exit(0 if ok else 1)
