"""Erzeuger und Prüfer der eingefrorenen Puffer-Tabellen `vwr_puffer.json` (Experiment 3, der Abschnitt „Puffer jenseits der Daten“).

Aufruf (aus dem Demo-Ordner, mit dem Test-venv; OMP_NUM_THREADS=1):
    python tools/vwr_make_puffer.py --write [--jobs 4]   # rechnet alles neu und schreibt die Datei
    python tools/vwr_make_puffer.py --check              # rechnet zwei Zellen neu (relative Toleranz 1e-4, Zählungen ±2 Läufe)

Teil 1: Fälle Vehikel B (exakte Aufzählung, 1 224 Komponenten), Pareto-Verteilung α = 2, Exponential, Normal(100, 20); n = 500, 2 000, 5 000 Beobachtungen; Ziele p = 10⁻², 10⁻³, 10⁻⁴ je Beobachtung;
sechs Verfahren (empirisches Quantil, Normal-Formel, Lognormal-Formel, GPD an die größten 5 %, Hill-Weissman an die größten 5 %, GEV an Blockmaxima zu 100); R = 1 000 Läufe je Zelle. Je Lauf und Ziel:
die wahre Überschreitungswahrscheinlichkeit q des geschätzten Puffers, exakt aus F. Kennzahlen: P(q ≤ p), P(q > 2p), Median q/p, Median b/b_wahr, Aufschlag f90 (90-%-Quantil von b_wahr/b),
Gesamtpuffer (Median b/b_wahr × f90). Teil 2: m Blockmaxima exakt GEV(0, 1, ξ) (ξ = 0 und 0,2), m = 6, 12, 24, 50, Ziele p = 1/12 und 1/120 je Block, vier Verfahren.
Zufall: SplitMix64 (numpy-Vektorform), Zufalls-Seed-Basis 960 000. Zähleinheit: Läufe."""

import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import vwr_blockfit as BF  # noqa: E402
import vwr_estimators as E  # noqa: E402
import vwr_gev as G  # noqa: E402
import vwr_scenario as S  # noqa: E402
import vwr_vecrng as VR  # noqa: E402
import vwr_vehicle as VB  # noqa: E402

OUT = ROOT / "vwr_puffer.json"
BASE = 960000
STAND = "2026-11-20"
CASES = ("vehikel_b", "pareto", "exp", "normal_100_20")
NS = (500, 2000, 5000)
PS = (1e-2, 1e-3, 1e-4)
R = 1000
SHORT_M = (6, 12, 24, 50)
SHORT_XI = (0.0, 0.2)
SHORT_P = (1.0 / 12.0, 1.0 / 120.0)
_ERFC = np.vectorize(math.erfc, otypes=[float])


def sf_array(case: str, b) -> np.ndarray:
    """Überlebensfunktion für viele Puffer auf einmal (exakt bzw. Interpolation beim Vehikel B)."""
    b = np.asarray(b, float)
    if case == "vehikel_b":
        return VB.sf_vec(b)
    if case == "pareto":
        return np.where(b >= 1.0, np.maximum(b, 1.0) ** -2.0, 1.0)
    if case == "exp":
        return np.where(b > 0, np.exp(-np.maximum(b, 0.0)), 1.0)
    return 0.5 * _ERFC(((b - 100.0) / 20.0) / math.sqrt(2.0))


def run_case(args):
    case, n, runs = args
    stream = VR.Stream(S.cell_seed(BASE, CASES.index(case), n))
    btrue = [S.isf(case, p) for p in PS]
    chunk = max(1, 2_000_000 // n)
    levels = {mt: np.full((runs, len(PS)), np.nan) for mt in E.PUFFER_METHODS}
    done = 0
    while done < runs:
        c = min(chunk, runs - done)
        X = S.draw(case, stream, (c, n))
        for i in range(c):
            lv = E.buffer_levels(X[i], PS)
            for mt in E.PUFFER_METHODS:
                levels[mt][done + i, :] = lv[mt]
        done += c
    rows = {}
    for mt in E.PUFFER_METHODS:
        for pi, p in enumerate(PS):
            b = levels[mt][:, pi]
            ok = np.isfinite(b)
            bb = np.maximum(b[ok], 0.0)
            q = sf_array(case, bb)
            ratio = bb / btrue[pi]
            f90 = float(np.quantile(btrue[pi] / np.maximum(bb, 1e-12), 0.90))
            rows[f"{mt}|{p:g}"] = {"fail": int((~ok).sum()), "hit": round(float(np.mean(q <= p)), 6), "gt2": round(float(np.mean(q > 2 * p)), 6), "medq": round(float(np.median(q) / p), 6),
                                   "medr": round(float(np.median(ratio)), 6), "f90": round(f90, 6), "tot": round(float(np.median(ratio)) * f90, 6)}
    return case, n, {"btrue": btrue, "rows": rows}


def run_short(args):
    xi, m = args
    stream = VR.Stream(S.cell_seed(BASE, 99, int(round(xi * 100)), m))
    U = stream.random(R * m).reshape(R, m)
    X = G.gev_ppf(U, 0.0, 1.0, xi)
    names = ("max", "gev_ml", "gumbel", "gev_lmom")
    res = {nm: np.full((R, len(SHORT_P)), np.nan) for nm in names}
    for r in range(R):
        x = X[r]
        res["max"][r, :] = x.max()
        try:
            f = BF.fit_ml(x)
            res["gev_ml"][r, :] = [float(G.pegel(p, f.mu, f.sigma, f.xi)) for p in SHORT_P]
        except (G.FitError, ValueError, OverflowError, ZeroDivisionError):
            pass
        try:
            mu, s = BF.gumbel_ml(x)
            res["gumbel"][r, :] = [float(G.pegel(p, mu, s, 0.0)) for p in SHORT_P]
        except (G.FitError, ValueError, OverflowError, ZeroDivisionError):
            pass
        try:
            mu, s, xh = G.fit_lmoments(x)
            res["gev_lmom"][r, :] = [float(G.pegel(p, mu, s, xh)) for p in SHORT_P]
        except (G.FitError, ValueError, OverflowError, ZeroDivisionError):
            pass
    out = {}
    for nm in names:
        for pi, p in enumerate(SHORT_P):
            b = res[nm][:, pi]
            ok = np.isfinite(b)
            q = 1.0 - G.gev_cdf(b[ok], 0.0, 1.0, xi)
            target = float(G.pegel(p, 0.0, 1.0, xi))
            out[f"{nm}|{p:.6f}"] = {"fail": int((~ok).sum()), "hit": round(float(np.mean(q <= p)), 6), "gt2": round(float(np.mean(q > 2 * p)), 6),
                                    "medq": round(float(np.median(q) / p), 6), "aufschlag90": round(float(np.quantile(target - b[ok], 0.90)), 6)}
    return xi, m, out


def build(jobs: int) -> dict:
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        cells = {f"{c}|{n}": res for c, n, res in pool.map(run_case, [(c, n, R) for c in CASES for n in NS])}
        short = {f"{xi}|{m}": res for xi, m, res in pool.map(run_short, [(xi, m) for xi in SHORT_XI for m in SHORT_M])}
    return {"stand": STAND, "R": R, "ps": list(PS), "ns": list(NS), "cases": list(CASES), "cells": cells, "short": short, "short_ps": list(SHORT_P),
            "quelle": {"teil1": f"tools/vwr_make_puffer.py, SplitMix64 (numpy-Vektorform), Zufalls-Seed-Basis {BASE}, R = {R} Läufe je Zelle, Stand {STAND}",
                       "vehikel_b": "gemeinsame Tabelle der Linie (1 224 Komponenten), SHA-256 siehe vwr_vehicle.TABLE_SHA256",
                       "teil2": f"m Blockmaxima exakt GEV(0, 1, ξ), R = {R} Läufe, Stand {STAND}"},
            "laufzeit_s": round(time.perf_counter() - t0, 1), "python": sys.version.split()[0], "numpy": np.__version__}


def close(a, b, tol=1e-4):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-12) + 1e-9


def check() -> int:
    data = json.loads(OUT.read_text(encoding="utf-8"))
    bad = 0
    for case, n in (("exp", 2000), ("vehikel_b", 500)):
        _, _, res = run_case((case, n, R))
        old = data["cells"][f"{case}|{n}"]["rows"]
        for key, st in res["rows"].items():
            for k in ("hit", "gt2", "medq", "medr", "f90", "tot"):
                if not close(st[k], old[key][k], 1e-3):
                    bad += 1
                    print(f"ABWEICHUNG {case} n={n} {key} {k}: neu {st[k]} alt {old[key][k]}")
            if abs(st["fail"] - old[key]["fail"]) > 2:
                bad += 1
                print(f"ABWEICHUNG {case} n={n} {key} fail")
    print("vwr_make_puffer --check:", "in Ordnung" if bad == 0 else f"{bad} Abweichungen")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(check())
    if "--write" in sys.argv:
        jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
        data = build(jobs)
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("geschrieben:", OUT, f"({data['laufzeit_s']} s)")
    else:
        print(__doc__)
