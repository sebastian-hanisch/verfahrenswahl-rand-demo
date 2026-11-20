"""Eingefrorene Messungen: Aufbau, innere Stimmigkeit, Orakel (das empirische Quantil gegen Beta im Band) und `--check` der drei Werkzeuge (rechnen je eine Auswahl neu und vergleichen als Band)."""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import vwr_buffer as B
import vwr_cvar as CV
import vwr_estimators as E
import vwr_reference as REF
import vwr_scenario as S

ROOT = Path(__file__).resolve().parent.parent


def test_matrix_aufbau_drei_seeds_16_zellen_zwei_ziele():
    g = REF.grid()
    assert g["ny"] == 100 and g["lengths"] == [10, 20, 50, 100] and g["targets"] == [0.01, 0.001] and len(g["matrix"]) == 3
    assert {"matrix", "kopplung", "verteilungen"} <= set(g["quelle"]) and g["stand"] == "2026-11-20"
    for base, mat in g["matrix"].items():
        assert mat["R"] == 400 and len(mat["cells"]) == 16
        for d in S.MATRIX_DISTS:
            for m in (10, 20, 50, 100):
                cell = mat["cells"][f"{d}|{m}"]
                assert cell["runs"] == 400 and 0 <= cell["gev_edge"] <= 400
                for q in ("0.01", "0.001"):
                    methods = set(cell["targets"][q])
                    assert methods == set(E.matrix_methods_for(100 * m)), (d, m)
                    for mt, st in cell["targets"][q].items():
                        assert st["n"] + st["fail"] == 400 and st["n"] >= 360, (d, m, q, mt)       # mindestens 90 % gültige Läufe
                        assert st["rmse"] > 0 and 0.0 <= st["under"] <= 1.0 and st["median_ratio"] > 0 and st["median_abs"] > 0


def test_die_drei_zufalls_seed_basen_unterscheiden_sich_nur_im_rauschen():
    a, b = REF.matrix_cell("exp", 50, 0.01, 0), REF.matrix_cell("exp", 50, 0.01, 1)
    assert a["gumbel"]["rmse"] != b["gumbel"]["rmse"] and abs(a["gumbel"]["rmse"] - b["gumbel"]["rmse"]) < 0.02


def test_kopplung_zelle_aufbau_und_xi_streuung():
    k = REF.grid()["kopplung"]
    assert set(k) == set(S.MATRIX_DISTS) and all(v["runs"] == 2000 and v["valid"] == 2000 for v in k.values())
    xi = k["exp"]["xi"]
    assert abs(xi["gev_ml"]["mean"]) < 0.03 and 0.10 < xi["gev_ml"]["sd"] < 0.14 and xi["gpd_ml50"]["sd"] > xi["gev_ml"]["sd"]


def test_cvar_aufbau_und_exakte_werte():
    d = REF.cvar()
    assert set(d["emp_r20000"]) == {f"{x}|600" for x in CV.CVAR_DISTS}
    assert set(d["emp_r2000"]) == {f"{x}|{n}" for x in CV.CVAR_DISTS for n in (60, 600, 6000)}
    assert set(d["gpd_r2000"]) == {f"{x}|{n}" for x in CV.CVAR_DISTS for n in (600, 6000)}
    for key, v in d["exact"].items():
        dist, alpha = key.split("|")
        assert v == pytest.approx(CV.es_exact(dist, float(alpha)), rel=1e-12)
    for block in (d["emp_r20000"], d["emp_r2000"], d["gpd_r2000"]):
        for cells in block.values():
            for st in cells.values():
                assert 0.0 < st["median"] < 1.2 and 0.0 <= st["below_one"] <= 1.0 and st["invalid"] <= st["runs"]
    assert all(c["runs"] == 20000 for cells in d["emp_r20000"].values() for c in cells.values())


def test_cvar_median_liegt_unter_eins_und_faellt_mit_alpha_im_bereich_der_messung():
    d = REF.cvar()["emp_r20000"]
    for dist in CV.CVAR_DISTS:
        med = [d[f"{dist}|600"][str(a)]["median"] for a in CV.ALPHAS]
        assert med == sorted(med, reverse=True) and med[0] < 1.0


def test_puffer_aufbau():
    p = REF.puffer()
    assert p["R"] == 1000 and p["cases"] == ["vehikel_b", "pareto", "exp", "normal_100_20"] and p["ns"] == [500, 2000, 5000]
    assert len(p["cells"]) == 12 and len(p["short"]) == 8
    for key, cell in p["cells"].items():
        assert len(cell["rows"]) == 6 * 3 and len(cell["btrue"]) == 3
        for row in cell["rows"].values():
            assert row["fail"] <= 1000 and 0.0 <= row["hit"] <= 1.0 and 0.0 <= row["gt2"] <= 1.0


def test_puffer_empirisches_quantil_stimmt_mit_beta_in_allen_zwoelf_zellen_ueberein():
    p = REF.puffer()
    worst = 0.0
    for case in p["cases"]:
        for n in p["ns"]:
            for pp in p["ps"]:
                row = p["cells"][f"{case}|{n}"]["rows"][f"emp|{pp:g}"]
                for sim, ex in ((row["gt2"], B.prob_q_above(n, pp)), (row["hit"], B.prob_q_within(n, pp))):
                    se = math.sqrt(max(ex * (1 - ex), 1e-9) / p["R"])
                    worst = max(worst, abs(sim - ex) / se)
    assert worst < 4.5


def test_puffer_wahre_quantile_aus_der_verteilung():
    p = REF.puffer()
    for case in p["cases"]:
        for n in p["ns"]:
            for pp, b in zip(p["ps"], p["cells"][f"{case}|{n}"]["btrue"]):
                assert b == pytest.approx(S.isf(case, pp), rel=1e-12)


def test_puffer_kurze_reihen_maximum_gegen_beta_1_m():
    p = REF.puffer()
    for key, rows in p["short"].items():
        xi, m = key.split("|")
        for pp in p["short_ps"]:
            ex = (1 - 2 * pp) ** int(m)
            se = math.sqrt(max(ex * (1 - ex), 1e-9) / p["R"])
            assert abs(rows[f"max|{pp:.6f}"]["gt2"] - ex) < 4.5 * se, (key, pp)


@pytest.mark.parametrize("script", ["vwr_make_vehicle.py", "vwr_make_grid.py", "vwr_make_cvar.py", "vwr_make_puffer.py"])
def test_werkzeug_check_rechnet_neu_und_vergleicht_im_band(script):
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "tools" / script), "--check"], capture_output=True, text=True, env=env, cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "in Ordnung" in r.stdout


def test_zaehlung_der_siege_gumbel_12_feste_zellen_hill_nur_bei_der_pareto_verteilung():
    import vwr_evaluation as V
    fixed = V.fixed_gumbel_cells()
    assert len(fixed) == 12 and {d for d, m, q in fixed} == {"exp", "normal"} and sum(1 for d, m, q in fixed if d == "exp") == 8 and all(m <= 20 for d, m, q in fixed if d == "normal")
    lo, hi, per = V.win_band("gumbel")
    assert 12 <= lo <= hi <= 15 and len(per) == 3
    for i in range(3):
        for d in S.MATRIX_DISTS:
            for m in V.LENGTHS:
                for q in V.TARGETS:
                    if V.cell_winner(REF.matrix_cell(d, m, q, i))[1].startswith("hill"):
                        assert d == "pareto"
    assert len(V.gumbel_underestimates(0.9)) == 16 and {d for d, m, q in V.gumbel_underestimates(0.9)} == {"pareto", "lognormal"}
    assert not [1 for d, m, q in V.gumbel_underestimates(0.9) if d in ("normal", "exp")]
