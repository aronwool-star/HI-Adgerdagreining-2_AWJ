"""
staerdarprof.py -- Einstaklingsverkefni 5: Stærðar- og hraðapróf.

Býr til tilbúin gagnasett af vaxandi stærð, keyrir price_and_cut á þeim öllum
og skráir keyrslutíma, fjölda ítrana dálkamyndunar og fjölda skurða.
Hver lausn er staðfest með verify.py.

Tvær tilraunir:
  A. Vaxandi stærð: fleiri breiddir, hærri pantanir og stærra W (6 stig,
     10 handahófsgagnasett á hverju stigi).
  B. Aðeins pantanir stækka: sömu breiddir og W, pantanir margfaldaðar
     með k = 1, 2, 4, ..., 32.

Úttak: staerdarprof_A.csv, staerdarprof_B.csv og staerdarprof.png
"""

import contextlib
import csv
import io
import random
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from price_and_cut import price_and_cut
import verify

SEED = 2026
N_PER_LEVEL = 10

# (stig, fjöldi breidda m, W, bil fyrir pantanir)
LEVELS = [
    (1, 3, 10, (5, 15)),
    (2, 4, 16, (5, 20)),
    (3, 5, 20, (10, 30)),
    (4, 6, 20, (10, 40)),
    (5, 7, 25, (20, 60)),
    (6, 8, 25, (30, 80)),
]


def make_instance(rng, m, W, d_range):
    """m ólíkar breiddir á bilinu [2, W-1] og handahófskenndar pantanir."""
    weights = sorted(rng.sample(range(2, W), m), reverse=True)
    demands = [rng.randint(*d_range) for _ in range(m)]
    return weights, demands


def run_one(weights, demands, W):
    """Keyrir price_and_cut, mælir tíma og staðfestir með verify.py (hljóðlaust)."""
    t0 = time.perf_counter()
    res = price_and_cut(weights, demands, W)
    dt = time.perf_counter() - t0

    with contextlib.redirect_stdout(io.StringIO()):
        ok = verify.verify(weights, demands, W, result=res)

    return {
        "m": len(weights),
        "W": W,
        "sum_d": sum(demands),
        "n_patterns": len(verify.enumerate_patterns(weights, W)),
        "time_s": dt,
        "pricing_iter": res["n_pricing_iterations"],
        "n_cuts": res["n_cuts"],
        "dual_pivots": res["n_dual_pivots"],
        "total_rolls": res["total_rolls"],
        "verified": ok,
        "weights": weights,
        "demands": demands,
    }


def experiment_A():
    rng = random.Random(SEED)
    rows = []
    for level, m, W, d_range in LEVELS:
        for rep in range(N_PER_LEVEL):
            w, d = make_instance(rng, m, W, d_range)
            r = run_one(w, d, W)
            r["level"] = level
            rows.append(r)
    return rows


def experiment_B():
    weights, W = [11, 8, 6, 5, 4, 3], 20
    base = [7, 9, 5, 12, 8, 10]
    rows = []
    for k in [1, 2, 4, 8, 16, 32]:
        r = run_one(weights, [k * x for x in base], W)
        r["k"] = k
        rows.append(r)
    return rows


def summarize_A(rows):
    out = []
    for level, m, W, d_range in LEVELS:
        g = [r for r in rows if r["level"] == level]
        times = [r["time_s"] for r in g]
        out.append({
            "stig": level,
            "m": m,
            "W": W,
            "pantanir": f"{d_range[0]}-{d_range[1]}",
            "mynstur": round(statistics.mean(r["n_patterns"] for r in g)),
            "tími_miðgildi_s": statistics.median(times),
            "tími_hámark_s": max(times),
            "ítranir_meðal": statistics.mean(r["pricing_iter"] for r in g),
            "skurðir_meðal": statistics.mean(r["n_cuts"] for r in g),
            "skurðir_hámark": max(r["n_cuts"] for r in g),
            "staðfest": f"{sum(r['verified'] for r in g)}/{len(g)}",
        })
    return out


def write_csv(path, rows, keys):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


def plot(rows_A, summary_A, rows_B, path="staerdarprof.png"):
    # Litir: flokkalitir 1 og 2 (blár, appelsínugulur) + grátt fyrir einstök gagnasett.
    # Ólík merki (o / s) svo að aðgreining byggi ekki á lit einum.
    blue, orange, grey = "#2a78d6", "#eb6834", "#a3a29c"
    ink, ink2 = "#0b0b0b", "#52514e"
    plt.rcParams.update({"axes.edgecolor": ink2, "axes.labelcolor": ink,
                         "xtick.color": ink2, "ytick.color": ink2, "font.size": 10})

    fig, ax = plt.subplots(2, 2, figsize=(12, 8.4))
    ax = ax.ravel()
    lv = [s["stig"] for s in summary_A]
    ticks = [f"{s['stig']}\nm={s['m']}\nW={s['W']}" for s in summary_A]

    # (a) Keyrslutími sem fall af stigi (log-ás): einstök gagnasett + miðgildi + hámark
    ax[0].scatter([r["level"] for r in rows_A], [r["time_s"] for r in rows_A],
                  color=grey, s=16, alpha=0.7, label="einstök gagnasett", zorder=2)
    ax[0].plot(lv, [s["tími_miðgildi_s"] for s in summary_A], "-o", color=blue,
               lw=2, ms=7, label="miðgildi", zorder=3)
    ax[0].plot(lv, [s["tími_hámark_s"] for s in summary_A], "--s", color=orange,
               lw=2, ms=6, label="hámark", zorder=3)
    ax[0].set_yscale("log")
    ax[0].set_xticks(lv, ticks)
    ax[0].set_xlabel("Stig")
    ax[0].set_ylabel("Keyrslutími [s] (log-ás)")
    ax[0].set_title("(a) Keyrslutími eftir stærð vandans", loc="left")
    ax[0].legend(frameon=False, fontsize=9)

    # (b) Meðalfjöldi ítrana og skurða sem fall af stigi
    ax[1].plot(lv, [s["ítranir_meðal"] for s in summary_A], "-o", color=blue,
               lw=2, ms=7, label="ítranir dálkamyndunar")
    ax[1].plot(lv, [s["skurðir_meðal"] for s in summary_A], "-s", color=orange,
               lw=2, ms=6, label="Gomory-skurðir")
    ax[1].set_xticks(lv, ticks)
    ax[1].set_xlabel("Stig")
    ax[1].set_ylabel("Meðalfjöldi (10 gagnasett á stigi)")
    ax[1].set_ylim(bottom=0)
    ax[1].set_title("(b) Ítranir og skurðir eftir stærð", loc="left")
    ax[1].legend(frameon=False, fontsize=9)

    # (c) Hvað ræður tímanum? Tími á móti fjölda skurða, stærð punkts = fjöldi mynstra
    sizes = [12 + r["n_patterns"] / 1.5 for r in rows_A]
    ax[2].scatter([r["n_cuts"] for r in rows_A], [r["time_s"] for r in rows_A],
                  s=sizes, color=blue, alpha=0.55, edgecolor="white", linewidth=0.8)
    for n in (25, 75, 150):
        ax[2].scatter([], [], s=12 + n / 1.5, color=blue, alpha=0.55, label=f"{n} mynstur")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("Fjöldi Gomory-skurða")
    ax[2].set_ylabel("Keyrslutími [s] (log-ás)")
    ax[2].set_title("(c) Tíminn fylgir skurðum og fjölda mynstra", loc="left")
    ax[2].legend(frameon=False, fontsize=9, title="stærð punkts", title_fontsize=9,
                 labelspacing=1.1)

    # (d) Tilraun B: aðeins pantanir stækka
    k = [r["k"] for r in rows_B]
    ax[3].plot(k, [r["pricing_iter"] for r in rows_B], "-o", color=blue,
               lw=2, ms=7, label="ítranir dálkamyndunar")
    ax[3].plot(k, [r["n_cuts"] for r in rows_B], "-s", color=orange,
               lw=2, ms=6, label="Gomory-skurðir")
    ax[3].set_xscale("log", base=2)
    ax[3].set_xticks(k, [f"{x}\n(Σd={r['sum_d']})" for x, r in zip(k, rows_B)])
    ax[3].set_xlabel("Margföldunarstuðull k á pantanir")
    ax[3].set_ylabel("Fjöldi")
    ax[3].set_ylim(0, 10)
    ax[3].set_title("(d) Aðeins pantanir stækka (m=6, W=20)", loc="left")
    ax[3].legend(frameon=False, fontsize=9)

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


if __name__ == "__main__":
    rows_A = experiment_A()
    summary_A = summarize_A(rows_A)
    rows_B = experiment_B()

    write_csv("staerdarprof_A.csv", rows_A,
              ["level", "m", "W", "sum_d", "n_patterns", "time_s", "pricing_iter",
               "n_cuts", "dual_pivots", "total_rolls", "verified", "weights", "demands"])
    write_csv("staerdarprof_B.csv", rows_B,
              ["k", "m", "W", "sum_d", "time_s", "pricing_iter", "n_cuts",
               "dual_pivots", "total_rolls", "verified"])
    plot(rows_A, summary_A, rows_B)

    print("Tilraun A -- vaxandi stærð (10 gagnasett á stigi):")
    hdr = f"{'stig':>4} {'m':>2} {'W':>3} {'pantanir':>9} {'mynstur':>7} " \
          f"{'tími mið.':>10} {'tími max':>9} {'ítr.':>5} {'skurðir':>8} {'max sk.':>7} {'staðf.':>6}"
    print(hdr)
    for s in summary_A:
        print(f"{s['stig']:>4} {s['m']:>2} {s['W']:>3} {s['pantanir']:>9} {s['mynstur']:>7} "
              f"{s['tími_miðgildi_s']:>9.3f}s {s['tími_hámark_s']:>8.3f}s "
              f"{s['ítranir_meðal']:>5.1f} {s['skurðir_meðal']:>8.1f} {s['skurðir_hámark']:>7} "
              f"{s['staðfest']:>6}")

    print("\nTilraun B -- aðeins pantanir stækka (w=(11,8,6,5,4,3), W=20):")
    print(f"{'k':>3} {'sum d':>6} {'rúllur':>6} {'tími':>8} {'ítr.':>5} {'skurðir':>8} {'staðf.':>6}")
    for r in rows_B:
        print(f"{r['k']:>3} {r['sum_d']:>6} {r['total_rolls']:>6} {r['time_s']:>7.3f}s "
              f"{r['pricing_iter']:>5} {r['n_cuts']:>8} {str(r['verified']):>6}")
