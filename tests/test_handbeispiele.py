"""Die Handbeispiele des Plans (H 1 bis H 6) als exakte Tests: Brüche, Ganzzahlen, Aufzählung, Tafelwerte; jede Prüfung rechnet einen zweiten Weg (scipy nur hier)."""

import itertools
import math
from fractions import Fraction as Fr

import numpy as np
import pytest
from scipy import integrate, optimize, stats

import vwr_buffer as B
import vwr_cvar as CV
import vwr_estimators as E
import vwr_evaluation as V
import vwr_gpd as GP
import vwr_scenario as S


# --- H1: Reihenlänge gegen Wiederkehrperiode ------------------------------------------------------------------------------------------------------
def test_h1_reihenlaenge_exakt():
    p = B.p_series_not_reaching
    assert p(50, 100) == Fr(99, 100) ** 50 and abs(float(p(50, 100)) - 0.6050) < 5e-5
    assert abs(float(1 - p(50, 100)) - 0.3950) < 5e-5 and abs(float(1 - p(50, 1000)) - 0.0488) < 5e-5
    assert abs(float(1 - p(100, 100)) - 0.6340) < 5e-5 and float(1 - p(100, 100)) > 1 - math.exp(-1)
    assert abs(float(1 - p(30, 100)) - 0.2603) < 5e-5
    assert abs(float(p(100, 100)) - 0.366) < 5e-4
    assert abs(1 - math.exp(-1) - 0.63212) < 5e-6


def test_h1_enumeration_27_64():
    tot = Fr(0)
    for bits in itertools.product((0, 1), repeat=3):
        pr = Fr(1)
        for b in bits:
            pr *= Fr(1, 4) if b else Fr(3, 4)
        if sum(bits) == 0:
            tot += pr
    assert tot == Fr(27, 64) == B.p_series_not_reaching(3, 4)


# --- H2: Block gegen Schwelle an zwölf Beobachtungen --------------------------------------------------------------------------------------------
def test_h2_zwoelf_beobachtungen_von_hand():
    x = np.array([3, 4, 7, 2, 9, 1, 5, 8, 6, 2, 10, 3], float)
    assert E.block_maxima(x, 4).tolist() == [7, 9, 10]                      # drei Blöcke zu 4 geben drei Maxima
    xs = np.sort(x)
    k = 4
    u = xs[len(xs) - k - 1]
    y = xs[len(xs) - k:] - u
    assert u == 6 and sorted(y.tolist(), reverse=True) == [4, 3, 2, 1]      # vier Überschreitungen aus denselben zwölf Werten
    sigma = y.mean()
    assert sigma == 2.5                                                      # Exponential-Rand: ML-Skala = Mittel
    nll = lambda s: len(y) * math.log(s) + y.sum() / s
    assert abs(optimize.minimize_scalar(nll, bounds=(0.1, 20), method="bounded", options={"xatol": 1e-10}).x - 2.5) < 1e-5   # zweiter Weg: numerische Minimierung
    level = GP.pot_level(0.0, sigma, u, k / 12, 1 / 120)
    assert level == pytest.approx(6 + 2.5 * math.log(40), abs=1e-12) and level == pytest.approx(15.2222, abs=1e-3)


# --- H3: geschlossene Zahlen der Tabelle ------------------------------------------------------------------------------------------------------
def _u_ks(n):
    u = np.linspace(0, min(n, 60), 400001)
    return float(np.max(np.abs(np.exp(-u) - (1 - u / n) ** n)))


def test_h3a_exponential_abstand_zum_gumbel_gesetz():
    assert abs(_u_ks(365) - 0.00074) < 5e-6
    assert abs(365 * _u_ks(365) - 2 / math.e ** 2) < 1e-3 and abs(2 / math.e ** 2 - 0.27067) < 5e-6
    assert next(n for n in range(2, 200) if _u_ks(n) < 0.01) == 28


def test_h3b_normal_abstand_beide_normierungen_und_pegel():
    n = 365
    b_n = float(stats.norm.isf(1.0 / n))
    a_n = 1.0 / (n * float(stats.norm.pdf(b_n)))
    xx = np.linspace(b_n - 3 * a_n, b_n + 12 * a_n, 400001)
    d_b = float(np.max(np.abs(stats.norm.cdf(xx) ** n - np.exp(-np.exp(-(xx - b_n) / a_n)))))
    assert abs(d_b - 0.02133) < 2e-4
    l2 = math.sqrt(2 * math.log(n))
    a_A, b_A = 1.0 / l2, l2 - (math.log(math.log(n)) + math.log(4 * math.pi)) / (2 * l2)
    xa = np.linspace(b_A - 3 * a_A, b_A + 12 * a_A, 400001)
    d_A = float(np.max(np.abs(stats.norm.cdf(xa) ** n - np.exp(-np.exp(-(xa - b_A) / a_A)))))
    assert abs(d_A - 0.0524) < 5e-4 and 70 < d_A / _u_ks(365) < 72 and 28 < d_b / _u_ks(365) < 29.5
    x_ex = float(stats.norm.isf(1 - 0.99 ** (1 / 365)))
    assert abs(x_ex - 4.033) < 5e-4 and abs((b_n - a_n * math.log(-math.log(0.99))) - 4.272) < 5e-4
    assert abs((b_A - a_A * math.log(-math.log(0.99))) - 4.147) < 1e-3
    assert abs(S.true_level("normal", 0.01, 365) - x_ex) < 1e-9


def test_h3c_hill_und_gpd_streuung_bei_xi_halb_k_100():
    xi, k = 0.5, 100
    assert abs(xi / math.sqrt(k) - 0.05) < 1e-12 and abs((1 + xi) / math.sqrt(k) - 0.15) < 1e-12
    g = stats.gamma(a=k, scale=xi / k)
    assert abs(g.mean() - 0.5) < 1e-12 and abs(g.std() - 0.05) < 1e-12


def test_h3d_extremalindex_bruchwerte_und_blockskala():
    assert [(1 - Fr(r)) ** 2 for r in (Fr(1, 2), Fr(4, 5), Fr(9, 10))] == [Fr(1, 4), Fr(1, 25), Fr(1, 100)]
    assert Fr(1, 5) * 150 == 30 and 1 / Fr(1, 25) == 25 and 1 / Fr(1, 100) == 100
    phi, b = Fr(4, 5), 50
    s_b = 1 + (b - 1) * (1 - phi)
    assert s_b == Fr(54, 5) and abs(float(s_b) / (-math.log(0.99)) - 1074.6) < 0.5


def test_h3e_trennschaerfe_und_risiko():
    assert abs(V.trend_power(0.02, 50) - 0.532) < 1e-3
    assert abs(float(1 - Fr(99, 100) ** 4) - 0.0394) < 5e-5
    assert (V.years_for_80(0.01), V.years_for_80(0.02), V.years_for_80(0.04)) == (99, 62, 39)
    assert V.trend_power(0.02, 62) >= 0.8 > V.trend_power(0.02, 61)


# --- H4: CVaR von Hand und exakt ----------------------------------------------------------------------------------------------------------------
def test_h4_cvar_von_hand():
    assert CV.cvar_empirical([1, 2, 3, 10], 0.5) == pytest.approx(6.5, abs=1e-12)          # Mittel der zwei größten
    assert CV.cvar_empirical([1, 2, 3, 10], 0.999) == pytest.approx(10.0, abs=1e-12)       # (1 − α) n < 1: das Maximum
    assert CV.cvar_empirical(list(range(1, 101)), 0.95) == pytest.approx(98.0, abs=1e-12)  # Mittel von 96 bis 100
    assert abs(0.999 ** 600 - 0.5486) < 5e-5 and CV.prob_no_scenario_above_quantile(0.999, 600) == 0.999 ** 600


@pytest.mark.parametrize("alpha", [0.9, 0.95, 0.99])
def test_h4_exakte_es_exponential_und_pareto_gegen_quadratur(alpha):
    num_exp = integrate.quad(lambda u: -math.log1p(-u), alpha, 1, epsabs=1e-12)[0] / (1 - alpha)
    num_par = integrate.quad(lambda u: (1 - u) ** -0.5, alpha, 1, epsabs=1e-11, epsrel=1e-11, limit=400)[0] / (1 - alpha)
    assert CV.es_exact("exp", alpha) == pytest.approx(num_exp, abs=1e-8)
    assert CV.es_exact("pareto", alpha) == pytest.approx(num_par, abs=1e-8)
    assert CV.es_exact("exp", 0.95) == pytest.approx(3.995732, abs=1e-6) and CV.es_exact("pareto", 0.95) == pytest.approx(8.944272, abs=1e-6)


# --- H5: die reine Funktion hinweise ------------------------------------------------------------------------------------------------------------
def test_h5_hinweise_acht_handfaelle():
    h = V.hinweise
    assert h(50, 100, k=250) == ["extrapolation"]
    assert h(50, 30, k=250) == []
    assert h(20, 1000, k=60, theta=0.2) == ["extrapolation", "kurze_reihe", "k_klein", "wenig_cluster"]
    assert h(50, 100, k=150, theta=0.2) == ["extrapolation", "wenig_cluster"]
    assert h(50, 100, k=150, theta=1.0) == ["extrapolation"]
    assert h(50, 100, k=250, beta=0.01) == ["extrapolation", "trend_nicht_erkennbar"]
    assert h(100, 100, k=250, beta=0.01) == []
    assert h(10, 10, k=None) == ["kurze_reihe"] and float(V.not_reached(10, 10)) < 0.5


def test_h5_die_schwelle_der_extrapolation_liegt_zwischen_m_68_und_69():
    assert float(V.not_reached(68, 100)) == pytest.approx(0.505, abs=5e-4) and float(V.not_reached(69, 100)) == pytest.approx(0.4998, abs=5e-5)
    assert V.hinweise(68, 100, k=250) == ["extrapolation"] and V.hinweise(69, 100, k=250) == []
    assert all(float(V.not_reached(m, 100)) > float(V.not_reached(m + 1, 100)) for m in range(1, 300))


# --- H6: Puffer jenseits der Daten (exakt) ------------------------------------------------------------------------------------------------------
def test_h6_maximum_von_n_beobachtungen():
    assert Fr(5, 6) ** 6 == Fr(15625, 46656) and abs(float(Fr(5, 6) ** 6) - 0.3349) < 5e-5
    assert abs(float(Fr(59, 60) ** 6) - 0.90408) < 1e-5
    assert abs(B.max_not_above_target(2000, 1e-4) - 0.6703) < 1e-4 and abs(B.prob_q_above(2000, 1e-4) - 0.6703) < 1e-4
    assert abs(B.expected_q(2000, 1e-4) / 1e-4 - 4.9975) < 1e-4
    val = integrate.quad(lambda x: 2000 * (1 - x) ** 1999, 2e-4, 1, epsabs=1e-12, epsrel=1e-12)[0]
    assert abs(val - (1 - 2e-4) ** 2000) < 1e-9


def test_h6_empirisches_quantil_n_500_p_001():
    j = E.emp_index(500, 0.01)
    assert j == 496 and abs(B.prob_q_above(500, 0.01) - 0.028) < 1e-3
    assert B.prob_q_above(500, 0.01) == pytest.approx(float(stats.beta.sf(0.02, 500 - j + 1, j)), abs=1e-12)


def test_h6_aufzaehlung_drei_stufen_27_64():
    cnt = sum(1 for tup in itertools.product(range(4), repeat=3) if max(tup) < 3)
    assert Fr(cnt, 64) == Fr(27, 64)
