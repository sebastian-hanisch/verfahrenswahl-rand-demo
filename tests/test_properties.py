"""Eigenschaften (je 20 bis 50 Zufallsinstanzen, fester Zufalls-Seed): Skalen- und Verschiebungs-Äquivarianz der Verfahren, Reihenfolge, Monotonie von `hinweise`, des Puffers und des CVaR."""

import math

import numpy as np
import pytest

import vwr_buffer as B
import vwr_cvar as CV
import vwr_estimators as E
import vwr_evaluation as V
import vwr_gpd as GP
import vwr_scenario as S
import vwr_vecrng as VR

QS = (0.01, 0.001)


def _series(dist, m, count, seed):
    return S.draw(dist, VR.Stream(seed), (count, 100 * m))


@pytest.mark.parametrize("dist", ["exp", "pareto", "lognormal"])
def test_skalierung_der_reihe_skaliert_alle_pegel(dist):
    c = 7.5
    for x in _series(dist, 20, 20, 101):
        a = E.levels_matrix(x, 100, QS)
        b = E.levels_matrix(c * x, 100, QS)
        for mt in a:
            assert np.allclose(np.array(b[mt]), c * np.array(a[mt]), rtol=1e-5, equal_nan=True), mt


def test_verschiebung_verschiebt_gev_gumbel_und_gpd_aber_nicht_hill():
    d = 1000.0
    moved_hill = 0
    for x in _series("exp", 20, 20, 102):
        a = E.levels_matrix(x, 100, QS)
        b = E.levels_matrix(x + d, 100, QS)
        for mt in ("gev_ml", "gev_lmom", "gumbel", "gpd_ml50", "gpd_pwm50", "gpd_ml5"):
            assert np.allclose(np.array(b[mt]), np.array(a[mt]) + d, rtol=1e-6, atol=1e-5), mt
        moved_hill += int(abs(b["hill50"][0] - (a["hill50"][0] + d)) > 1.0)       # Hill ist nicht verschiebungsinvariant (Quotient X/X_(k+1))
    assert moved_hill == 20


def test_gleicher_zufalls_seed_gleiche_ausgabe_anderer_seed_andere():
    s = V.Settings(R2=10, m2=10, dist2="exp")
    a, b = V.analyse_live(s), V.analyse_live(s)
    c = V.analyse_live(V.Settings(R2=10, m2=10, dist2="exp", seed=36))
    for mt in a.errors:
        assert np.array_equal(a.errors[mt], b.errors[mt], equal_nan=True)
    assert not np.array_equal(a.errors["gumbel"], c.errors["gumbel"], equal_nan=True)


def test_reihenfolge_innerhalb_eines_blocks_aendert_nichts():
    rng = np.random.default_rng(7)
    for x in _series("lognormal", 10, 20, 103):
        y = x.reshape(10, 100).copy()
        for row in y:
            rng.shuffle(row)
        a = E.levels_matrix(x, 100, QS)
        b = E.levels_matrix(y.reshape(-1), 100, QS)
        for mt in a:
            assert np.allclose(np.array(a[mt]), np.array(b[mt]), rtol=1e-9, equal_nan=True), mt


def test_zusaetzlicher_groesster_wert_senkt_die_gpd_schwelle_nie():
    # Die Schwelle ist der (k+1)-größte Wert: ein neuer größter Wert verschiebt die Rangfolge, der (k+1)-größte Wert kann nur größer oder gleich werden.
    for x in _series("exp", 10, 30, 104):
        k = 50
        xs = np.sort(x)
        u_before = xs[len(xs) - k - 1]
        xs2 = np.sort(np.append(x, x.max() + 1.0))
        u_after = xs2[len(xs2) - k - 1]
        assert u_after >= u_before


def test_hinweise_extrapolation_genau_bei_anteil_ab_einhalb_und_monoton_in_m():
    for T in (10, 30, 100, 300, 1000):
        prev = None
        for m in range(1, 400):
            has = "extrapolation" in V.hinweise(m, T, k=250)
            assert has == (float(V.not_reached(m, T)) >= 0.5)
            if prev is False:
                assert has is False                       # nie wieder an, wenn die Warnung einmal aus ist
            prev = has


def test_nicht_erreicht_waechst_mit_t_und_faellt_mit_m():
    for m in (5, 50, 200):
        vals = [float(V.not_reached(m, T)) for T in V.C.T_OPTIONS]
        assert vals == sorted(vals)
    for T in V.C.T_OPTIONS:
        vals = [float(V.not_reached(m, T)) for m in (5, 50, 200)]
        assert vals == sorted(vals, reverse=True)


def test_warnung_wenig_cluster_genau_bei_theta_k_unter_50():
    for theta in (1.0, 0.5, 0.2, 0.1):
        for k in range(10, 501, 10):
            assert ("wenig_cluster" in V.hinweise(50, 100, k=k, theta=theta)) == (theta * k < 50)


def test_wahrer_pegel_waechst_mit_dem_ziel_und_der_empirische_cvar_mit_dem_niveau():
    for dist in S.MATRIX_DISTS:
        assert S.true_level(dist, 0.01) < S.true_level(dist, 0.001)
    for x in S.draw("lognormal", VR.Stream(5), (30, 300)):
        vals = [CV.cvar_empirical(x, a) for a in CV.ALPHAS]
        assert vals == sorted(vals)


def test_puffer_ueberschreitung_waechst_wenn_n_p_kleiner_wird():
    for p in (1e-2, 1e-3, 1e-4):
        vals = [B.prob_q_above(n, p) for n in (5000, 2000, 500)]
        assert vals == sorted(vals), p
    assert all(0.0 <= B.prob_q_above(n, p) <= 1.0 for n in (500, 5000) for p in (1e-2, 1e-4))


def test_gpd_ml_skaliert_die_skala_und_laesst_xi_gleich():
    for x in _series("exp", 10, 20, 105):
        y = np.sort(x)[-50:] - np.sort(x)[-51]
        xi, sg, _ = GP.gpd_ml(y)
        xi2, sg2, _ = GP.gpd_ml(3.0 * y)
        assert xi2 == pytest.approx(xi, abs=1e-6) and sg2 == pytest.approx(3.0 * sg, rel=1e-5) and math.isfinite(sg)
