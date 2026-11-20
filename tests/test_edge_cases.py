"""Randfälle: zu wenige Blöcke, k zu groß oder zu klein, ξ̂ am Rand −1, Hill bei Werten ≤ 0, T < m, θ = 0, (1−α)·n < 1 beim CVaR, ξ̂ ≥ 1, Ziele jenseits der Reihe, Regler an den Grenzen."""

import math

import numpy as np
import pytest

import vwr_blockfit as BF
import vwr_buffer as B
import vwr_constants as C
import vwr_cvar as CV
import vwr_estimators as E
import vwr_evaluation as V
import vwr_scenario as S
import vwr_vecrng as VR


def test_zu_wenige_bloecke_geben_nan_statt_einer_zahl():
    x = S.draw("exp", VR.Stream(1), (200,))
    lv = E.levels_matrix(x, 100, (0.01,))             # m = 2 Blöcke
    assert math.isnan(lv["gev_lmom"][0])              # L-Momente brauchen mindestens 3 Maxima
    assert all(math.isnan(v[0]) or v[0] > 0 for v in lv.values())
    lv1 = E.levels_matrix(x[:100], 100, (0.01,))      # m = 1 Block: die ML braucht mindestens 2
    assert math.isnan(lv1["gev_ml"][0]) and math.isnan(lv1["gumbel"][0])


def test_k_ohne_ueberschreitungen_und_k_nicht_kleiner_als_n_geben_nan():
    xs = np.sort(S.draw("exp", VR.Stream(2), (300,)))
    assert math.isnan(E.pot_levels_sorted(xs, 1, [1e-3])[0])          # k < 2
    assert math.isnan(E.pot_levels_sorted(xs, 300, [1e-3])[0])        # k ≥ N
    assert math.isnan(E.hill_levels_sorted(xs, 300, [1e-3])[0])
    assert E.pot_levels_sorted(xs, 3, [1e-3])[0] > 0                  # k = 3 geht noch


def test_hill_bei_nichtpositiver_schwelle_gibt_nan():
    x = np.concatenate([np.linspace(-5, -1, 100), np.array([1.0, 2.0, 3.0])])
    assert math.isnan(E.hill_levels_sorted(np.sort(x), 60, [1e-3])[0])
    assert E.hill_levels_sorted(np.sort(x), 2, [1e-3])[0] > 0


def test_ml_klebt_bei_gleichverteilten_maxima_an_der_grenze_minus_eins():
    hits = 0
    for seed in range(30):
        mx = VR.Stream(seed).random(10)       # Gleichverteilung auf (0, 1): endlicher Rand, ξ = −1
        try:
            hits += int(BF.fit_ml(mx).edge)
        except Exception:
            pass
    assert hits >= 10                                                                     # der Anteil wird angezeigt, nicht verschwiegen (info["gev_edge"])


def test_info_meldet_gev_rand_in_levels_matrix():
    info = {}
    x = VR.Stream(3).random(1000)                    # 10 Jahre Gleichverteilung
    E.levels_matrix(x, 100, (0.01,), info=info)
    assert "gev_edge" in info and isinstance(info["gev_edge"], bool)


def test_ziel_laenger_als_die_reihe_und_theta_null():
    assert float(V.not_reached(10, 1000)) > 0.99 and "extrapolation" in V.hinweise(10, 1000)
    assert float(V.not_reached(100, 10)) < 1e-4 and "extrapolation" not in V.hinweise(100, 10)         # T < m: die Reihe erreicht den Pegel fast sicher
    assert "wenig_cluster" in V.hinweise(50, 100, k=500, theta=0.0)                                    # θ = 0: keine Cluster, keine Zahl
    with pytest.raises(ValueError):
        V.years_for_80(0.0)
    with pytest.raises(ValueError):
        B.p_series_not_reaching(-1, 100)


def test_cvar_maximum_bei_1_minus_alpha_n_unter_1_und_gpd_es_nan_bei_xi_ab_1():
    x = S.draw("pareto", VR.Stream(4), (600,))
    assert CV.cvar_empirical(x, 0.999) == pytest.approx(float(x.max()), rel=1e-12)
    heavy = (1.0 - VR.Stream(5).random(600)) ** (-1 / 0.8)             # Pareto α = 0,8: ξ = 1,25, Mittel nicht endlich
    vals = [CV.gpd_es(heavy, 30, 0.99)]
    assert all(math.isnan(v) for v in vals)
    assert CV.gpd_k(60) == 20 and CV.gpd_k(600) == 30 and CV.gpd_k(6000) == 300


def test_wahrscheinlichkeiten_ausserhalb_des_bereichs_werfen_und_grenzfaelle_stimmen():
    with pytest.raises(ValueError):
        S.isf("exp", 0.0)
    with pytest.raises(ValueError):
        S.isf("exp", 1.0)
    with pytest.raises(ValueError):
        S.isf("unbekannt", 0.5)
    assert B.prob_q_above(500, 0.6) == 0.0                                   # Faktor·p ≥ 1: nichts kann darüber liegen
    assert E.emp_index(10, 0.5) == 6 and E.emp_index(10, 1e-9) == 10         # j = ⌈(1−p)(n+1)⌉, höchstens n
    with pytest.raises(ValueError):
        B.f90_of_maximum("exp", 500, 1e-2)


def test_alle_auswertungen_laufen_an_den_reglergrenzen():
    for dist2 in C.DIST2_OPTIONS:
        s = V.Settings(m1=C.M1_MIN, T1=1000, k1=C.K1_MIN, theta1=0.1, beta1=0.04, rand1="schwer", dist2=dist2, m2=10, target2=1000, R2=C.R2_MIN)
        a = V.analyse_lage(s)
        assert V.verdict_lage(a)[1] and V.verdict_matrix(s, V.analyse_live(s))[1]
    s = V.Settings(m1=C.M1_MAX, T1=30, k1=C.K1_MAX, beta1=0.005)
    assert V.analyse_lage(s).band is None and V.verdict_lage(V.analyse_lage(s))[0] in ("success", "warning")
    for case in C.CASE3_OPTIONS:
        for n in C.N3_OPTIONS:
            for p in C.P3_OPTIONS:
                assert V.verdict_puffer(V.analyse_puffer(V.Settings(case3=case, n3=n, p3=p)))[1]
    for dist in C.DIST4_OPTIONS:
        for n in C.N4_OPTIONS:
            for alpha in C.ALPHA4_OPTIONS:
                a = V.analyse_cvar(V.Settings(dist4=dist, n4=n, alpha4=alpha))
                assert np.isfinite(a.ratios).sum() >= 50 and V.verdict_cvar(a)[1]
