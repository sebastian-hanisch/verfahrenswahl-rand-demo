"""Puffer jenseits der Daten: exakte Größen (Beta-Verteilung der wahren Überschreitungswahrscheinlichkeit, verteilungsfrei) und die Reihenlänge gegen die Wiederkehrperiode.

Der Puffer b ist das Quantil zur Überschreitungswahrscheinlichkeit p je Beobachtung; n·p ist die erwartete Zahl der Beobachtungen oberhalb des Ziels. Das empirische Quantil X_(j), j = ⌈(1−p)(n+1)⌉ (höchstens n),
hat für JEDE stetige Verteilung F die wahre Überschreitungswahrscheinlichkeit q = 1 − F(X_(j)) ~ Beta(n−j+1, j). Für ganze Parameter ist das eine Binomialsumme: P(q ≤ x) = P(Bin(n, x) ≥ n−j+1).
Das Maximum von m Blockmaxima hat q ~ Beta(1, m). Nur Standardbibliothek (`math`, `fractions`); die Tests rechnen mit `scipy.stats.beta` und Quadratur nach.
Zähleinheiten: Beobachtungen n, Blöcke m (Jahre), Läufe nur in den gemessenen Tabellen."""

import math
from fractions import Fraction

from vwr_estimators import emp_index


def p_series_not_reaching(m_years: int, period: int) -> Fraction:
    """Anteil der Reihen aus m iid-Jahresmaxima, die den Pegel der Wiederkehrperiode T nicht erreichen: (1 − 1/T)^m, exakt als Bruch."""
    if m_years < 0 or period < 1:
        raise ValueError("m ≥ 0 und T ≥ 1 verlangt")
    return (1 - Fraction(1, period)) ** m_years


def beta_params(n: int, p: float) -> tuple:
    """(a, b) = (n−j+1, j) der Beta-Verteilung der wahren Überschreitungswahrscheinlichkeit des empirischen Quantils."""
    j = emp_index(n, p)
    return n - j + 1, j


def prob_q_above(n: int, p: float, factor: float = 2.0) -> float:
    """P(q > factor·p) für das empirische Quantil, exakt: P(Bin(n, factor·p) ≤ n−j) (Summe über i = 0 … n−j)."""
    j = emp_index(n, p)
    x = factor * p
    if x >= 1.0:
        return 0.0
    if x <= 0.0:
        return 1.0
    total = 0.0
    lx, l1x = math.log(x), math.log1p(-x)
    for i in range(0, n - j + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lx + (n - i) * l1x)
    return min(1.0, total)


def prob_q_within(n: int, p: float) -> float:
    """P(q ≤ p): Anteil der Läufe, in denen das empirische Quantil sein Ziel einhält (exakt, Binomialsumme)."""
    return 1.0 - prob_q_above(n, p, 1.0)


def expected_q(n: int, p: float) -> float:
    """E[q] = (n−j+1)/(n+1); bei j = n (n·p < 1) ist das 1/(n+1)."""
    a, _ = beta_params(n, p)
    return a / (n + 1.0)


def max_not_above_target(n: int, p: float) -> float:
    """Das Maximum von n Beobachtungen überschreitet 2p mit q ~ Beta(1, n): P(q > 2p) = (1 − 2p)^n (exakt, geschlossene Form)."""
    return (1.0 - 2.0 * p) ** n


def blocks_max_prob(m: int, p: float, factor: float = 2.0) -> float:
    """Maximum von m Blockmaxima: q ~ Beta(1, m), P(q > factor·p) = (1 − factor·p)^m; bei p = 1/12 und factor = 2 als Bruch (5/6)^m."""
    return (1.0 - factor * p) ** m


def f90_of_maximum(dist: str, n: int, p: float) -> float:
    """Aufschlag f90 des Puffers „Maximum der n Beobachtungen“ (gilt, wenn das empirische Quantil das Maximum ist, also j = n, n·p unter etwa 1): das 90-%-Quantil von b_wahr / Maximum.
    Exakt aus F^n: das 10-%-Quantil des Maximums ist x mit F(x)^n = 0,1, also P(X > x) = 1 − 0,1^(1/n); f90 = b_wahr / x. Die Simulation (R = 1 000 Läufe) streut um diesen Wert."""
    import vwr_scenario as S
    if emp_index(n, p) != n:
        raise ValueError("Das empirische Quantil ist hier nicht das Maximum (n·p zu groß)")
    p_x = -math.expm1(math.log(0.1) / n)
    return S.isf(dist, p) / S.isf(dist, p_x)
