"""Erzeuger und Prüfer der eingefrorenen CVaR-Raster `vwr_cvar.json` (Experiment 4).

Aufruf (aus dem Demo-Ordner, mit dem Test-venv; OMP_NUM_THREADS=1):
    python tools/vwr_make_cvar.py --write [--jobs 4]   # rechnet alles neu und schreibt die Datei
    python tools/vwr_make_cvar.py --check              # rechnet eine Zelle neu (relative Toleranz 1e-4)

Inhalt: Median des Verhältnisses (empirischer CVaR aus n gleich wahrscheinlichen Szenarien / exakter CVaR) und Anteil der Läufe unter 1, je Verteilung (fünf ausgedachte), n und α:
(1) n = 600 mit R = 20 000 Läufen; (2) n = 60, 600, 6 000 mit R = 2 000; (3) die Erweiterung durch eine GPD an die oberen 5 % (mindestens 20 Werte) für α = 0,99 und 0,999, n = 600 und 6 000,
R = 2 000 (ungültig: ξ̂ ≥ 1). Das Mittel des Verhältnisses wird nicht geführt (bei der Pareto-Verteilung α = 2 ist die Varianz unendlich). Zufall: SplitMix64 (numpy-Vektorform), Zufalls-Seed-Basis 950 000.
Zähleinheit: Läufe (Stichproben von Szenarien), nicht Szenarien."""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import vwr_cvar as CV  # noqa: E402
import vwr_scenario as S  # noqa: E402
import vwr_vecrng as VR  # noqa: E402

OUT = ROOT / "vwr_cvar.json"
BASE = 950000
STAND = "2026-11-20"
NS = (60, 600, 6000)
R_BIG, R_RASTER, R_GPD = 20000, 2000, 2000


def ratios(dist, n, R, tag, alphas=CV.ALPHAS, gpd=False):
    """Verhältnis-Felder (Läufe × α) des empirischen CVaR (oder der GPD-Erweiterung) zum exakten; in Stücken, damit der Speicher klein bleibt."""
    stream = VR.Stream(S.cell_seed(BASE, CV.CVAR_DISTS.index(dist), n, tag))
    exact = np.array([CV.es_exact(dist, a) for a in alphas])
    chunk = max(1, min(R, 4_000_000 // n))
    out = np.empty((R, len(alphas)))
    done = 0
    while done < R:
        c = min(chunk, R - done)
        X = S.draw(dist, stream, (c, n))
        if gpd:
            k = CV.gpd_k(n)
            for i in range(c):
                for j, a in enumerate(alphas):
                    out[done + i, j] = CV.gpd_es(X[i], k, a) / exact[j]
        else:
            for j, a in enumerate(alphas):
                out[done: done + c, j] = CV.cvar_empirical_rows(X, a) / exact[j]
        done += c
    return out


def summarize(r):
    ok = np.isfinite(r)
    v = r[ok]
    return {"runs": int(r.size), "invalid": int((~ok).sum()), "median": round(float(np.median(v)), 6), "below_one": round(float(np.mean(v < 1.0)), 6)}


def task(args):
    kind, dist, n, R = args
    if kind == "emp":
        r = ratios(dist, n, R, tag=R)
        return kind, dist, n, {str(a): summarize(r[:, j]) for j, a in enumerate(CV.ALPHAS)}
    r = ratios(dist, n, R, tag=7, alphas=(0.99, 0.999), gpd=True)
    return kind, dist, n, {str(a): summarize(r[:, j]) for j, a in enumerate((0.99, 0.999))}


def build(jobs: int) -> dict:
    t0 = time.perf_counter()
    tasks = [("emp", d, 600, R_BIG) for d in CV.CVAR_DISTS] + [("emp", d, n, R_RASTER) for d in CV.CVAR_DISTS for n in NS] + [("gpd", d, n, R_GPD) for d in CV.CVAR_DISTS for n in (600, 6000)]
    data = {"emp_r20000": {}, "emp_r2000": {}, "gpd_r2000": {}}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for kind, dist, n, res in pool.map(task, tasks):
            if kind == "gpd":
                data["gpd_r2000"][f"{dist}|{n}"] = res
            elif res and next(iter(res.values()))["runs"] == R_BIG:
                data["emp_r20000"][f"{dist}|{n}"] = res
            else:
                data["emp_r2000"][f"{dist}|{n}"] = res
    data["exact"] = {f"{d}|{a}": CV.es_exact(d, a) for d in CV.CVAR_DISTS for a in CV.ALPHAS}
    data["quelle"] = {"raster": f"tools/vwr_make_cvar.py, SplitMix64 (numpy-Vektorform), Zufalls-Seed-Basis {BASE}, Läufe R = {R_BIG} (n = 600) bzw. {R_RASTER}, Stand {STAND}",
                      "exakt": "geschlossene Formeln (vwr_cvar.es_exact), Test gegen Quadratur der Quantilfunktion",
                      "gpd": f"GPD-ML an die oberen 5 % (mindestens 20 Werte), R = {R_GPD}, Stand {STAND}"}
    data["stand"] = STAND
    data["laufzeit_s"] = round(time.perf_counter() - t0, 1)
    data["python"], data["numpy"] = sys.version.split()[0], np.__version__
    return data


def check() -> int:
    data = json.loads(OUT.read_text(encoding="utf-8"))
    r = ratios("pareto", 600, R_BIG, tag=R_BIG)
    old = data["emp_r20000"]["pareto|600"]
    bad = 0
    for j, a in enumerate(CV.ALPHAS):
        s = summarize(r[:, j])
        for key in ("median", "below_one"):
            if abs(s[key] - old[str(a)][key]) > 1e-4 * max(abs(old[str(a)][key]), 1e-9) + 1e-9:
                bad += 1
                print(f"ABWEICHUNG pareto n=600 alpha={a} {key}: neu {s[key]} alt {old[str(a)][key]}")
    print("vwr_make_cvar --check:", "in Ordnung" if bad == 0 else f"{bad} Abweichungen")
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
