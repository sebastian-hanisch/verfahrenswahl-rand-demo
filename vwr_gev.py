"""GEV-Kern: Verteilungsfunktion, Pegel, negative Log-Likelihood, L-Momente und ihre Rückrechnung. Nur Standardbibliothek und numpy (kein scipy in der App).

Vorzeichen-Konvention der Linie: ξ > 0 Fréchet-Typ, ξ = 0 Gumbel, ξ < 0 Weibull-Typ (oberer Endpunkt μ - σ/ξ).
`scipy.stats.genextreme` hat c = -ξ; Hosking schreibt k = -ξ. Hier steht überall ξ; der Test `tests/test_oracle_gev.py` prüft gegen scipy mit c = -ξ.
G(x) = exp(-[1 + ξ (x - μ)/σ]^(-1/ξ)) auf 1 + ξ (x - μ)/σ > 0, ξ = 0 als Grenzwert exp(-exp(-(x - μ)/σ)).
Pegel zur Überschreitungswahrscheinlichkeit p je Block: x_p = μ + σ/ξ ((-ln(1 - p))^(-ξ) - 1)."""

import math

import numpy as np

EULER = 0.5772156649015329
XI_MIN = -1.0            # Nebenbedingung der ML-Anpassung: für ξ < -1 ist die Likelihood unbeschränkt (Wahl dieser Demo, keine Eigenschaft von scipy)
XI_GUMBEL = 1e-9         # darunter gelten die Formeln als Gumbel-Grenzfall (Umschaltgrenze)
HUGE = 1e300
KLAMMER_LO, KLAMMER_HI = -0.999, 0.99   # Klammer der L-Moment-Rückrechnung (Wahl dieser Demo)
OHNE_KLAMMER_LO = -60.0                  # „ohne Klammer“: ξ̂ bis -60 zugelassen (darunter liegt τ3 innerhalb 1e-8 von -1)


class FitError(ValueError):
    """Anpassung nicht möglich (zu wenige Block-Maxima, alle gleich, ...): die App zeigt eine Meldung statt einer Zahl."""


def gev_cdf(x, mu, sigma, xi):
    z = (np.asarray(x, float) - mu) / sigma
    if abs(xi) < XI_GUMBEL:
        return np.exp(-np.exp(-z))
    t = 1 + xi * z
    with np.errstate(over="ignore"):
        return np.where(t > 0, np.exp(-np.power(np.maximum(t, 1e-300), -1 / xi)), 0.0 if xi > 0 else 1.0)


def pegel(p, mu, sigma, xi):
    """Pegel (Wiederkehrwert) zur Überschreitungswahrscheinlichkeit p je Block, 0 < p < 1 (Coles 2001, Gl. 3.4)."""
    with np.errstate(divide="ignore", over="ignore"):
        y = -np.log1p(-np.asarray(p, float))
        if abs(xi) < XI_GUMBEL:
            return mu - sigma * np.log(y)
        return mu + sigma * (np.power(y, -xi) - 1) / xi


def gev_ppf(u, mu, sigma, xi):
    """Quantilfunktion zur Wahrscheinlichkeit u (0 < u < 1): y = -ln u direkt, ohne den Umweg 1 - (1 - u)."""
    with np.errstate(divide="ignore", over="ignore"):
        y = -np.log(np.asarray(u, float))
        if abs(xi) < XI_GUMBEL:
            return mu - sigma * np.log(y)
        return mu + sigma * (np.power(y, -xi) - 1) / xi


def endpoint(mu, sigma, xi):
    """Oberer Endpunkt μ - σ/ξ (nur ξ < 0), sonst unendlich."""
    return mu - sigma / xi if xi < -XI_GUMBEL else math.inf


def nll(theta, x):
    """Negative Log-Likelihood, theta = (μ, ln σ, ξ); HUGE außerhalb des Trägers oder für ξ < XI_MIN."""
    mu, ls, xi = theta
    if xi < XI_MIN:
        return HUGE
    z = (x - mu) * math.exp(-ls)
    if abs(xi) < 1e-7:
        return x.size * ls + float(np.sum(z) + np.sum(np.exp(-z)))
    t = 1 + xi * z
    if np.any(t <= 0):
        return HUGE
    lt = np.log(t)
    return x.size * ls + (1 + 1 / xi) * float(np.sum(lt)) + float(np.sum(np.exp(-lt / xi)))


def nll_values(x, mu, sigma, xi):
    """Wie `nll` mit (μ, σ, ξ) statt (μ, ln σ, ξ), für die Handbeispiele; ohne Prüfung des Trägers (ξ ≠ 0)."""
    x = np.asarray(x, float)
    t = 1 + xi * (x - mu) / sigma
    return x.size * math.log(sigma) + (1 + 1 / xi) * float(np.sum(np.log(t))) + float(np.sum(t ** (-1 / xi)))


# ---------- L-Momente ----------
def sample_lmoments(x):
    """(l1, l2, τ3) aus den Wahrscheinlichkeits-gewichteten Momenten b0, b1, b2; braucht m >= 3 (b2 hat (m - 2) im Nenner)."""
    x = np.sort(np.asarray(x, float))
    m = x.size
    if m < 3:
        raise FitError("L-Momente brauchen mindestens 3 Block-Maxima.")
    j = np.arange(1, m + 1)
    b0 = x.mean()
    b1 = float(np.sum((j - 1) / (m - 1) * x)) / m
    b2 = float(np.sum((j - 1) * (j - 2) / ((m - 1) * (m - 2)) * x)) / m
    l1, l2, l3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
    if not l2 > 0:
        raise FitError("Alle Block-Maxima sind gleich (l2 = 0): keine Anpassung möglich.")
    return float(l1), float(l2), float(l3 / l2)


def lmoments_pop(mu, sigma, xi):
    """Populations-L-Momente (l1, l2, τ3) der GEV (Hosking-Form mit k = -ξ)."""
    k = -xi
    if abs(k) < 1e-9:
        return mu + sigma * EULER, sigma * math.log(2), 2 * math.log(3) / math.log(2) - 3
    g = math.gamma(1 + k)
    return mu + sigma * (1 - g) / k, sigma * (-math.expm1(-k * math.log(2))) * g / k, tau3_of_xi(xi)


def tau3_of_xi(xi):
    """τ3 der GEV als Funktion von ξ: wächst streng mit ξ (τ3(-1) = -1/3, τ3(0) = 0,16993, τ3(1) = 1)."""
    k = -xi
    if abs(k) < 1e-9:
        return 2 * math.log(3) / math.log(2) - 3
    return 2 * math.expm1(-k * math.log(3)) / math.expm1(-k * math.log(2)) - 3


def xi_from_tau3(t3, lo=KLAMMER_LO, hi=KLAMMER_HI):
    """ξ aus τ3 exakt (Bisektion, τ3 wächst streng in ξ), auf [lo, hi] geklammert: liegt τ3 außerhalb, bleibt ξ am Rand."""
    if tau3_of_xi(lo) > t3:
        return lo
    if tau3_of_xi(hi) < t3:
        return hi
    a, b = lo, hi
    for _ in range(200):
        mid = 0.5 * (a + b)
        if tau3_of_xi(mid) < t3:
            a = mid
        else:
            b = mid
        if b - a < 1e-14:
            break
    return 0.5 * (a + b)


def xi_from_tau3_hosking(t3):
    """Hosking-Näherung (Hosking/Wallis/Wood 1985, nach Zusammenfassung): k = 7,8590 c + 2,9554 c², c = 2/(3 + τ3) - ln 2/ln 3; ξ = -k."""
    c = 2 / (3 + t3) - math.log(2) / math.log(3)
    return -(7.8590 * c + 2.9554 * c * c)


def gev_from_lmoments(l1, l2, xi):
    """(μ, σ) zu gegebenem ξ aus l1 und l2 (Hosking-Form)."""
    k = -xi
    if abs(k) < 1e-9:
        sigma = l2 / math.log(2)
        return l1 - sigma * EULER, sigma
    sigma = l2 * k / ((-math.expm1(-k * math.log(2))) * math.gamma(1 + k))
    return l1 - sigma * (1 - math.gamma(1 + k)) / k, sigma


def fit_lmoments(x, klammer=True):
    """GEV-Anpassung nach L-Momenten mit exakt invertiertem ξ̂. klammer=True: ξ̂ ∈ [-0,999; 0,99]; False: ξ̂ ∈ [-60; 0,99].
    Rückgabe (μ̂, σ̂, ξ̂)."""
    l1, l2, t3 = sample_lmoments(x)
    xi = xi_from_tau3(t3, KLAMMER_LO if klammer else OHNE_KLAMMER_LO, KLAMMER_HI)
    mu, sigma = gev_from_lmoments(l1, l2, xi)
    if not (math.isfinite(mu) and math.isfinite(sigma) and sigma > 0):
        raise FitError("Die Rückrechnung der L-Momente gibt keine gültige Anpassung.")
    return mu, sigma, xi
