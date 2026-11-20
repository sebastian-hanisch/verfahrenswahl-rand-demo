"""Orakel-Tests (ORACLE-PLAYBOOK §5, §6): jede Kernroutine gegen einen anderen Rechenweg. scipy steht nur hier; ξ-Konvention: `genextreme` hat c = −ξ, `genpareto` c = ξ.

Geprüft werden: die exakten wahren Pegel (F^100 und scipy), die Anpassungen (GEV-ML, Gumbel, L-Momente, GPD-ML, PWM, Hill-Weissman) gegen scipy bzw. eine direkte Rechnung, der exakte CVaR gegen Quadratur,
der empirische CVaR gegen eine zweite Fassung, die Beta-Größen des Puffers gegen `scipy.stats.beta` und gegen Simulation im Band."""

import itertools
import math

import numpy as np
import pytest
from scipy import integrate, stats

import vwr_blockfit as BF
import vwr_buffer as B
import vwr_cvar as CV
import vwr_estimators as E
import vwr_gev as G
import vwr_gpd as GP
import vwr_hill as H
import vwr_scenario as S
import vwr_vecrng as VR

SCIPY_DIST = {"exp": stats.expon(), "pareto": stats.pareto(2.0), "normal": stats.norm(), "lognormal": stats.lognorm(1.0), "t3": stats.t(3.0), "normal_100_20": stats.norm(100, 20)}


# --- wahre Pegel, Überlebensfunktion ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dist", list(SCIPY_DIST))
@pytest.mark.parametrize("p", [1e-2, 1e-4, 1.005e-4])
def test_exakter_pegel_gegen_scipy_isf(dist, p):
    assert S.isf(dist, p) == pytest.approx(float(SCIPY_DIST[dist].isf(p)), rel=1e-9)
    assert S.sf(dist, S.isf(dist, p)) == pytest.approx(p, rel=1e-8)


@pytest.mark.parametrize("dist", S.MATRIX_DISTS)
@pytest.mark.parametrize("q", [0.01, 0.001])
def test_wahrer_pegel_erfuellt_f_hoch_100_gleich_eins_minus_q(dist, q):
    x = S.true_level(dist, q)
    assert S.block_cdf(dist, x) == pytest.approx(1 - q, abs=1e-9)
    assert x == pytest.approx(float(SCIPY_DIST[dist].isf(1 - (1 - q) ** 0.01)), rel=1e-8)


def test_t3_verteilungsfunktion_gegen_scipy():
    for x in (-5.0, -0.3, 0.0, 1.1, 8.0):
        assert S.t3_cdf(x) == pytest.approx(float(stats.t(3.0).cdf(x)), abs=1e-13)


# --- GEV ---------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.2, 0.5])
def test_gev_pegel_und_verteilungsfunktion_gegen_scipy_mit_c_gleich_minus_xi(xi):
    mu, sg = 100.0, 20.0
    for p in (0.1, 0.01, 0.001):
        ref = float(stats.genextreme(c=-xi, loc=mu, scale=sg).isf(p))
        assert float(G.pegel(p, mu, sg, xi)) == pytest.approx(ref, rel=1e-9)
    x = np.array([60.0, 90.0, 130.0, 200.0])
    ok = np.isfinite(stats.genextreme(c=-xi, loc=mu, scale=sg).logpdf(x))
    assert np.allclose(G.gev_cdf(x, mu, sg, xi)[ok], stats.genextreme(c=-xi, loc=mu, scale=sg).cdf(x)[ok], atol=1e-12)


@pytest.mark.parametrize("xi", [-0.2, 0.0, 0.3])
def test_gev_ml_erreicht_mindestens_die_likelihood_von_scipy(xi):
    mx = G.gev_ppf(VR.Stream(11).random(60), 100.0, 20.0, xi)
    f = BF.fit_ml(mx)
    mine = float(stats.genextreme.nnlf((-f.xi, f.mu, f.sigma), mx))
    c, loc, scale = stats.genextreme.fit(mx, 0.0 - xi * 0.5)
    theirs = float(stats.genextreme.nnlf((c, loc, scale), mx))
    assert mine <= theirs + 1e-6


def test_gumbel_ml_gegen_scipy_gumbel_r():
    mx = G.gev_ppf(VR.Stream(3).random(50), 100.0, 20.0, 0.0)
    mu, s = BF.gumbel_ml(mx)
    loc, scale = stats.gumbel_r.fit(mx)
    assert mu == pytest.approx(loc, rel=1e-5) and s == pytest.approx(scale, rel=1e-5)


def test_l_momente_der_stichprobe_gegen_paar_und_tripel_aufzaehlung():
    x = np.array([2.0, 3.0, 5.0, 6.0, 9.0, 11.0, 12.0, 20.0])
    m = len(x)
    l1, l2, t3 = G.sample_lmoments(x)
    assert l1 == pytest.approx(x.mean(), abs=1e-12)
    pairs = [abs(a - b) / 2 for a, b in itertools.combinations(x, 2)]
    assert l2 == pytest.approx(float(np.mean(pairs)), abs=1e-12)
    l3 = float(np.mean([(max(t) - 2 * sorted(t)[1] + min(t)) / 3 for t in itertools.combinations(x, 3)]))
    assert t3 == pytest.approx(l3 / l2, abs=1e-12) and m == 8


def test_l_momente_rueckrechnung_trifft_die_population():
    for xi in (-0.3, 0.0, 0.2, 0.5):
        l1, l2, t3 = G.lmoments_pop(100.0, 20.0, xi)
        xi_back = G.xi_from_tau3(t3)
        mu, sg = G.gev_from_lmoments(l1, l2, xi_back)
        assert (xi_back, mu, sg) == pytest.approx((xi, 100.0, 20.0), abs=1e-4)


# --- GPD ---------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("xi", [-0.2, 0.0, 0.3, 0.6])
def test_gpd_ml_erreicht_mindestens_die_likelihood_von_scipy(xi):
    u = VR.Stream(5).random(150)
    y = GP.gpd_isf(1.0 - u, xi, 2.0)
    y = y[y > 0]
    xi_h, sg_h, _ = GP.gpd_ml(y)
    mine = float(stats.genpareto.nnlf((xi_h, 0.0, sg_h), y))
    c, loc, scale = stats.genpareto.fit(y, floc=0.0)
    theirs = float(stats.genpareto.nnlf((c, 0.0, scale), y))
    assert mine <= theirs + 1e-6
    assert xi_h == pytest.approx(c, abs=0.02) and sg_h == pytest.approx(scale, rel=0.05)


def test_gpd_ml_ist_mindestens_so_gut_wie_die_exponential_loesung_mit_skala_gleich_mittel():
    y = -np.log(1.0 - VR.Stream(8).random(200))
    xi, sg, _ = GP.gpd_ml(y)
    nll_exp = float(len(y) * math.log(y.mean()) + y.sum() / y.mean())                       # ξ = 0: ML-Skala = Mittel (geschlossen)
    nll_mine = float(stats.genpareto.nnlf((xi, 0.0, sg), y))
    assert nll_mine <= nll_exp + 1e-9 and abs(xi) < 0.25


def test_gpd_pwm_gegen_direkte_formel():
    y = np.sort(-np.log(1.0 - VR.Stream(2).random(40)))
    k = y.size
    a0 = sum(y) / k
    a1 = sum(y[i] * (1 - (i + 1 - 0.35) / k) for i in range(k)) / k
    xi, sg = GP.gpd_pwm(y)
    assert xi == pytest.approx(2.0 - a0 / (a0 - 2 * a1), abs=1e-12) and sg == pytest.approx(2 * a0 * a1 / (a0 - 2 * a1), abs=1e-12)


def test_pot_pegel_gegen_scipy_genpareto():
    xi, sg, u, zeta, p = 0.3, 2.0, 5.0, 0.05, 1e-4
    ref = u + float(stats.genpareto(c=xi, scale=sg).isf(p / zeta))
    assert GP.pot_level(xi, sg, u, zeta, p) == pytest.approx(ref, rel=1e-12)
    assert GP.pot_level(0.0, sg, u, zeta, p) == pytest.approx(u + sg * math.log(zeta / p), rel=1e-12)


# --- Hill-Weissman -----------------------------------------------------------------------------------------------------------------------------
def test_hill_gegen_direkte_rechnung_und_weissman_ohne_extrapolation():
    x = 1.0 + np.abs(np.sin(np.arange(1, 61) * 1.7)) * 5 + np.arange(60) * 0.1
    xs = np.sort(x)[::-1]
    k = 12
    h = sum(math.log(xs[i] / xs[k]) for i in range(k)) / k
    top = xs[None, : k + 1]
    assert float(H.hill(top, k)[0]) == pytest.approx(h, abs=1e-12)
    n = x.size
    assert float(H.weissman(top, k, n, k / n)[0]) == pytest.approx(xs[k], abs=1e-12)     # k/(n p) = 1: keine Extrapolation
    assert E.hill_levels_sorted(np.sort(x), k, [1e-3])[0] == pytest.approx(xs[k] * (k / (n * 1e-3)) ** h, rel=1e-12)


def test_hill_bei_reiner_pareto_verteilung_mittel_und_streuung_im_band():
    # H_k ~ Gamma(k, ξ/k): Mittel ξ, Streuung ξ/√k; Pareto α = 2 hat ξ = 0,5
    R, n, k = 2000, 1000, 100
    X = S.draw("pareto", VR.Stream(21), (R, n))
    top = H.top_sorted(X, k)
    h = H.hill(top, k)
    assert abs(h.mean() - 0.5) < 4 * 0.05 / math.sqrt(R) and abs(h.std(ddof=1) - 0.05) < 4 * 0.05 / math.sqrt(2 * R)


# --- CVaR --------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dist", CV.CVAR_DISTS)
@pytest.mark.parametrize("alpha", CV.ALPHAS)
def test_exakter_cvar_gegen_quadratur_der_quantilfunktion(dist, alpha):
    ppf = {"normal": stats.norm.ppf, "exp": stats.expon.ppf, "lognormal": stats.lognorm(1.0).ppf, "pareto": stats.pareto(2.0).ppf, "t3": stats.t(3.0).ppf}[dist]
    # u = 1 − s²: die Endpunkt-Singularität verschwindet
    f = lambda s: float(ppf(1 - s * s)) * 2 * s
    val = integrate.quad(f, 0.0, math.sqrt(1 - alpha), limit=500, epsabs=1e-12, epsrel=1e-10)[0] / (1 - alpha)
    assert CV.es_exact(dist, alpha) == pytest.approx(val, rel=1e-6)


def test_empirischer_cvar_gegen_zweite_fassung_bei_ganzzahligem_1_minus_alpha_n():
    x = S.draw("lognormal", VR.Stream(4), (50, 200))
    for alpha, top in ((0.9, 20), (0.95, 10), (0.99, 2)):
        second = np.sort(x, axis=1)[:, -top:].mean(axis=1)
        assert np.allclose(CV.cvar_empirical_rows(x, alpha), second, rtol=1e-12)


# --- Puffer ------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [500, 2000, 5000])
@pytest.mark.parametrize("p", [1e-2, 1e-3, 1e-4])
def test_beta_groessen_gegen_scipy(n, p):
    a, b = B.beta_params(n, p)
    assert (a, b) == (n - E.emp_index(n, p) + 1, E.emp_index(n, p))
    assert B.prob_q_above(n, p) == pytest.approx(float(stats.beta.sf(2 * p, a, b)), abs=1e-10)
    assert B.prob_q_within(n, p) == pytest.approx(float(stats.beta.cdf(p, a, b)), abs=1e-10)
    assert B.expected_q(n, p) == pytest.approx(float(stats.beta.mean(a, b)), abs=1e-12)


@pytest.mark.parametrize("n,p", [(2000, 1e-4), (500, 1e-4), (5000, 1e-4), (100, 1e-3)])
def test_maximum_hat_die_geschlossene_form_eins_minus_zwei_p_hoch_n(n, p):
    assert E.emp_index(n, p) == n and B.prob_q_above(n, p) == pytest.approx((1 - 2 * p) ** n, rel=1e-9)


@pytest.mark.parametrize("dist,n,p", [("exp", 500, 1e-2), ("pareto", 2000, 1e-3), ("vehikel_b", 500, 1e-2)])
def test_empirisches_quantil_wahre_ueberschreitung_ist_beta_in_simulation_im_band(dist, n, p):
    R = 2000
    X = S.draw(dist, VR.Stream(1234), (R, n))
    j = E.emp_index(n, p)
    b = np.sort(X, axis=1)[:, j - 1]
    q = np.array([S.sf(dist, v) for v in b])
    for sim, ex in ((float(np.mean(q > 2 * p)), B.prob_q_above(n, p)), (float(np.mean(q <= p)), B.prob_q_within(n, p))):
        se = math.sqrt(max(ex * (1 - ex), 1e-9) / R)
        assert abs(sim - ex) < 4.5 * se, (sim, ex)


def test_aufschlag_des_maximums_exakt_gegen_simulation_im_band():
    n, p, R = 2000, 1e-4, 3000
    X = S.draw("exp", VR.Stream(99), (R, n))
    mx = X.max(axis=1)
    f90_sim = float(np.quantile(S.isf("exp", p) / mx, 0.9))
    assert f90_sim == pytest.approx(B.f90_of_maximum("exp", n, p), rel=0.03)
    assert B.f90_of_maximum("vehikel_b", 2000, 1e-4) == pytest.approx(2.2048, abs=5e-4)        # Aufschlag 120,5 % (Korrektur U4, siehe README)


def test_f90_verlangt_dass_das_empirische_quantil_das_maximum_ist():
    with pytest.raises(ValueError):
        B.f90_of_maximum("exp", 500, 1e-2)


def test_kurze_blockreihe_maximum_ist_beta_1_m():
    m, p = 6, 1 / 12
    R = 6000
    U = VR.Stream(5).random(R * m).reshape(R, m)
    top = G.gev_ppf(U, 0.0, 1.0, 0.2).max(axis=1)
    q = 1.0 - G.gev_cdf(top, 0.0, 1.0, 0.2)
    ex = float((1 - 2 * p) ** m)
    assert abs(float(np.mean(q > 2 * p)) - ex) < 4.5 * math.sqrt(ex * (1 - ex) / R) and ex == pytest.approx((5 / 6) ** 6, rel=1e-12)
    assert float(np.mean(q)) == pytest.approx(1 / (m + 1), abs=4.5 * float(np.std(q)) / math.sqrt(R))
