"""Erzeuger und Prüfer der eingefrorenen Verfahrens-Matrix `vwr_grid.json` (Experiment 2).

Aufruf (aus dem Demo-Ordner, mit dem Test-venv; OMP_NUM_THREADS=1):
    python tools/vwr_make_grid.py --write [--jobs 4]   # rechnet alles neu (beim Bau 131 s, unter fremder Last mehrfach länger) und schreibt die Datei
    python tools/vwr_make_grid.py --check              # rechnet drei Zellen neu und vergleicht mit der Datei (relative Toleranz 1e-4, Zählungen ±2 Läufe)

Inhalt: (1) Matrix: 4 Verteilungen × 4 Reihenlängen (m = 10, 20, 50, 100 Jahre zu je 100 Beobachtungen) × 2 Ziele (Überschreitungswahrscheinlichkeit 0,01 und 0,001 je Jahr) × acht Verfahren,
R = 400 Läufe je Zelle, alle Verfahren sehen dieselben Reihen, mit drei Zufalls-Seed-Basen (Zählung der Siege als Band); (2) die Matrix-Zelle (m = 50, 100-Jahres-Pegel) mit R = 2 000 für das
Verhältnis Schwelle/Block; (3) die Streuung von ξ̂ bei Exponential-Daten (N = 5 000, R = 2 000). Zufall: SplitMix64 (numpy-Vektorform der Vorlage), Zufalls-Seeds ab 910 000, nicht die der Vormessung.
Zähleinheit: Läufe (je ein ganzer Satz Reihen), Zellen (Verteilung × Länge × Ziel)."""

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
import vwr_gpd as GP  # noqa: E402
import vwr_scenario as S  # noqa: E402
import vwr_vecrng as VR  # noqa: E402

OUT = ROOT / "vwr_grid.json"
SEED_BASES = (910000, 920000, 930000)
R_CELL = 400
R_KOPPLUNG = 2000
KOPPLUNG_BASE = 940000
LENGTHS = (10, 20, 50, 100)
TARGETS = (0.01, 0.001)
STAND = "2026-11-20"


def run_cell(args):
    """Eine Zelle (Verteilung, m): R Läufe, Statistik je Ziel und Verfahren."""
    dist, m, base, R = args
    ny = S.NY
    stream = VR.Stream(S.cell_seed(base, S.MATRIX_DISTS.index(dist), m))
    X = S.draw(dist, stream, (R, ny * m))
    truth = [S.true_level(dist, q) for q in TARGETS]
    methods = E.matrix_methods_for(ny * m)
    err = {(ti, mt): [] for ti in range(len(TARGETS)) for mt in methods}
    fail = {(ti, mt): 0 for ti in range(len(TARGETS)) for mt in methods}
    edge = 0
    for r in range(R):
        info = {}
        lv = E.levels_matrix(X[r], ny, TARGETS, info=info)
        edge += bool(info.get("gev_edge", False))
        for mt in methods:
            for ti in range(len(TARGETS)):
                e = E.log_error(lv[mt][ti], truth[ti])
                if math.isfinite(e):
                    err[(ti, mt)].append(e)
                else:
                    fail[(ti, mt)] += 1
    cell = {"runs": R, "gev_edge": edge, "targets": {}}
    for ti, q in enumerate(TARGETS):
        d = {}
        for mt in methods:
            e = np.array(err[(ti, mt)])
            d[mt] = {"n": int(e.size), "fail": int(fail[(ti, mt)]), "median_ratio": round(float(math.exp(np.median(e))), 6), "rmse": round(float(math.sqrt(np.mean(e * e))), 6),
                     "under": round(float(np.mean(e < 0)), 6), "median_abs": round(float(np.median(np.abs(e))), 6), "sd_ln": round(float(np.std(e, ddof=1)), 6)}
        cell["targets"][str(q)] = d
    return dist, m, base, cell


def run_kopplung(args):
    """Matrix-Zelle m = 50, 100-Jahres-Pegel mit R = 2 000: Verhältnis GPD-ML(k = 50)/GEV-ML; bei Exponential zusätzlich ξ̂ der drei Anpassungen."""
    dist, R = args
    ny, m = S.NY, 50
    stream = VR.Stream(S.cell_seed(KOPPLUNG_BASE, S.MATRIX_DISTS.index(dist), m))
    X = S.draw(dist, stream, (R, ny * m))
    truth = S.true_level(dist, 0.01)
    eg, ep = [], []
    xg, xp, xw = [], [], []
    for r in range(R):
        x = X[r]
        lv = E.levels_matrix(x, ny, (0.01,), methods=("gev_ml", "gpd_ml50"))
        eg.append(E.log_error(lv["gev_ml"][0], truth))
        ep.append(E.log_error(lv["gpd_ml50"][0], truth))
        if dist == "exp":
            xg.append(BF.fit_ml(E.block_maxima(x, ny)).xi)
            xs = np.sort(x)
            y = xs[-50:] - xs[-51]
            xp.append(GP.gpd_ml(y)[0])
            xw.append(GP.gpd_pwm(y)[0])
    eg, ep = np.array(eg), np.array(ep)
    ok = np.isfinite(eg) & np.isfinite(ep)
    eg, ep = eg[ok], ep[ok]
    out = {"runs": int(R), "valid": int(ok.sum()), "rmse_gev": float(math.sqrt(np.mean(eg * eg))), "rmse_gpd": float(math.sqrt(np.mean(ep * ep))),
           "sd_gev": float(np.std(eg, ddof=1)), "sd_gpd": float(np.std(ep, ddof=1))}
    out["ratio_rmse"] = out["rmse_gpd"] / out["rmse_gev"]
    out["ratio_sd"] = out["sd_gpd"] / out["sd_gev"]
    if dist == "exp":
        xg, xp, xw = np.array(xg), np.array(xp), np.array(xw)
        out["xi"] = {"gev_ml": {"mean": float(xg.mean()), "sd": float(xg.std(ddof=1))}, "gpd_ml50": {"mean": float(np.nanmean(xp)), "sd": float(np.nanstd(xp, ddof=1))},
                     "gpd_pwm50": {"mean": float(np.nanmean(xw)), "sd": float(np.nanstd(xw, ddof=1))}}
    for k in list(out):
        if isinstance(out[k], float):
            out[k] = round(out[k], 6)
    return dist, out


def build(jobs: int) -> dict:
    t0 = time.perf_counter()
    tasks = [(d, m, b, R_CELL) for b in SEED_BASES for d in S.MATRIX_DISTS for m in LENGTHS]
    matrix = {str(b): {"base": b, "R": R_CELL, "cells": {}} for b in SEED_BASES}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for dist, m, base, cell in pool.map(run_cell, tasks):
            matrix[str(base)]["cells"][f"{dist}|{m}"] = cell
        kopp = dict(pool.map(run_kopplung, [(d, R_KOPPLUNG) for d in S.MATRIX_DISTS]))
    quelle = {
        "matrix": f"tools/vwr_make_grid.py, SplitMix64 (numpy-Vektorform), Zufalls-Seed-Basen {list(SEED_BASES)}, R = {R_CELL} Läufe je Zelle, Stand {STAND}",
        "kopplung": f"tools/vwr_make_grid.py, Zufalls-Seed-Basis {KOPPLUNG_BASE}, R = {R_KOPPLUNG} Läufe, Stand {STAND}",
        "verteilungen": "exakte Verteilungen, ausgedacht; Jahr = 100 Beobachtungen; Ziele q = 0,01 und 0,001 je Jahr, wahr exakt aus F^100",
    }
    return {"quelle": quelle, "stand": STAND, "ny": S.NY, "lengths": list(LENGTHS), "targets": list(TARGETS), "matrix": matrix, "kopplung": kopp,
            "laufzeit_s": round(time.perf_counter() - t0, 1), "python": sys.version.split()[0], "numpy": np.__version__}


def close(a, b, tol=1e-4):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-12) + 1e-9


def check() -> int:
    data = json.loads(OUT.read_text(encoding="utf-8"))
    bad = 0
    for dist, m, base in (("exp", 10, SEED_BASES[0]), ("pareto", 50, SEED_BASES[1]), ("lognormal", 20, SEED_BASES[2])):
        _, _, _, cell = run_cell((dist, m, base, R_CELL))
        old = data["matrix"][str(base)]["cells"][f"{dist}|{m}"]
        for tk, d in cell["targets"].items():
            for mt, st in d.items():
                o = old["targets"][tk][mt]
                for key in ("median_ratio", "rmse", "under", "median_abs", "sd_ln"):
                    if not close(st[key], o[key], 1e-3 if key == "under" else 1e-4):
                        bad += 1
                        print(f"ABWEICHUNG {dist} m={m} q={tk} {mt} {key}: neu {st[key]} alt {o[key]}")
                if abs(st["fail"] - o["fail"]) > 2:
                    bad += 1
                    print(f"ABWEICHUNG {dist} m={m} q={tk} {mt} fail: neu {st['fail']} alt {o['fail']}")
        if abs(cell["gev_edge"] - old["gev_edge"]) > 2:
            bad += 1
            print(f"ABWEICHUNG {dist} m={m} gev_edge")
    print("vwr_make_grid --check:", "in Ordnung" if bad == 0 else f"{bad} Abweichungen")
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
