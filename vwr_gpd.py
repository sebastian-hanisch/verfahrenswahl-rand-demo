"""GPD-Anpassung an Überschreitungen (Auszug aus `gpd-ueberschreitungen-demo`, Demo der Linie „Extremwerte erklärt“): Maximum Likelihood über das Profil in t = θ·mean(y), PWM und Pegel.

Die Funktionen sind wörtliche Abschriften aus `gpu_algorithm.py` („Überschreitungen und die GPD“; `tests/test_copies.py` vergleicht sie, solange der Nachbarordner da ist). Nur numpy und Standardbibliothek.
Konvention ξ wie in der ganzen Linie: ξ > 0 Fréchet-Typ, ξ = 0 Gumbel, ξ < 0 Weibull-Typ; `scipy.stats.genpareto` hat c = ξ, `scipy.stats.genextreme` c = −ξ (nur in den Tests)."""

import math

import numpy as np

XI_MIN = -1.0            # Nebenbedingung der ML: für ξ < −1 ist die Likelihood unbeschränkt (Wahl der Umsetzung, README)


def gpd_sf(y, xi: float, sigma: float):
    """Überlebensfunktion P(Y > y) der GPD(ξ, σ); jenseits des Endpunkts (ξ < 0) null."""
    y = np.asarray(y, dtype=float)
    if abs(xi) < 1e-12:
        return np.exp(-y / sigma)
    t = 1.0 + xi * y / sigma
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(t > 0, np.power(np.where(t > 0, t, 1.0), -1.0 / xi), 0.0 if xi < 0 else 1.0)
    return out


def gpd_isf(q, xi: float, sigma: float):
    """y mit P(Y > y) = q (0 < q ≤ 1)."""
    q = np.asarray(q, dtype=float)
    if abs(xi) < 1e-12:
        return -sigma * np.log(q)
    return sigma / xi * (np.power(q, -xi) - 1.0)


def pot_level(xi: float, sigma: float, u: float, zeta: float, p: float) -> float:
    """Pegel mit Überschreitungswahrscheinlichkeit p je Beobachtung: u + (σ/ξ)((ζ/p)^ξ − 1), für ξ = 0 u + σ ln(ζ/p); ζ = Anteil der Beobachtungen über u."""
    ratio = zeta / p
    if abs(xi) < 1e-8:
        return u + sigma * math.log(ratio)
    return u + sigma / xi * (ratio ** xi - 1.0)


def _golden_min(f, a: float, b: float, tol: float = 1e-10) -> float:
    """Minimum auf [a, b] (Goldener Schnitt); Stellen mit +inf sind erlaubt (Wand der Nebenbedingung). Gibt den besten wirklich ausgewerteten Punkt zurück,
    nicht die Mitte des letzten Intervalls (die kann hinter der Wand liegen)."""
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    best_x, best_f = (c, fc) if fc <= fd else (d, fd)
    while abs(b - a) > tol * (abs(c) + abs(d) + 1e-12):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
            if fc < best_f:
                best_x, best_f = c, fc
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
            if fd < best_f:
                best_x, best_f = d, fd
    return best_x


def gpd_ml(y, xi_min: float = XI_MIN):
    """Maximum Likelihood der GPD (ξ, σ) an Überschreitungen y > 0 über das Profil in t = θ·mean(y), θ = ξ/σ.

    Gibt (ξ̂, σ̂, Randtreffer) zurück. Randtreffer = die Lösung liegt am unteren Rand ξ = ξ_min oder am oberen Ende des Suchgitters (t = 40).
    Ohne die Nebenbedingung ξ ≥ −1 wäre die Likelihood für ξ < −1 unbeschränkt (der Endpunkt rückt an die größte Beobachtung)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 2:
        raise ValueError("Die Anpassung braucht mindestens zwei Überschreitungen")
    yb = float(y.mean())
    ymax = float(y.max())
    tlo = -yb / ymax * (1.0 - 1e-9)

    def xi_of(t):
        if abs(t) <= 1e-12:
            return 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(np.mean(np.log1p(t * y / yb)))

    def ll(t):
        if abs(t) < 1e-9:
            return -n * math.log(yb) - n
        xi = xi_of(t)
        if not math.isfinite(xi) or xi < xi_min or xi == 0.0 or t / xi <= 0:
            return -math.inf
        return n * math.log(t / xi) - n * math.log(yb) - n * (xi + 1.0)

    grid = np.concatenate([np.linspace(tlo, -1e-3, 70), [0.0], np.geomspace(1e-3, 40.0, 90)])
    vals = np.array([ll(float(t)) for t in grid])
    i = int(np.argmax(vals))
    if not np.isfinite(vals[i]):
        return math.nan, math.nan, True
    a, b = float(grid[max(i - 1, 0)]), float(grid[min(i + 1, len(grid) - 1)])
    t = float(grid[i])
    if a < b:
        # feineres Gitter im Klammerintervall (die Profil-Likelihood kann dort zwei Gipfel haben, einer davon an der Wand ξ = ξ_min), dann Goldener Schnitt im besten Teilintervall
        fine = np.linspace(a, b, 41)
        fvals = np.array([ll(float(s)) for s in fine])
        j = int(np.argmax(fvals))
        if fvals[j] > vals[i]:
            t, best = float(fine[j]), float(fvals[j])
        else:
            best = float(vals[i])
        a2, b2 = float(fine[max(j - 1, 0)]), float(fine[min(j + 1, len(fine) - 1)])
        t_opt = _golden_min(lambda s: -ll(s), a2, b2)
        if ll(t_opt) >= best - 1e-12:
            t = t_opt
    if abs(t) < 1e-9:
        return 0.0, yb, False
    xi = xi_of(t)
    sigma = xi * yb / t
    lower_edge = xi <= xi_min + 1e-3
    if xi_min == -1.0 and (lower_edge or -n * math.log(ymax) > ll(t) + 1e-12):
        # Am Rand ξ = −1 ist die Dichte (1/σ) je Wert, solange σ ≥ größte Überschreitung; das beschränkte Maximum liegt dort bei σ = max(y), nicht beim Profilpunkt.
        # Hat die Profil-Likelihood einen inneren Gipfel UND die Wand, gewinnt der höhere (bei kleinem k kommt das vor).
        xi, sigma, lower_edge = -1.0, ymax, True
    edge = lower_edge or (i >= len(grid) - 2)
    return xi, sigma, bool(edge)


def gpd_pwm(y):
    """Wahrscheinlichkeitsgewichtete Momente (Hosking/Wallis 1987, Plotting Position (i − 0,35)/k); gültig für ξ < 1. Gibt (ξ̂, σ̂) oder (nan, nan)."""
    y = np.sort(np.asarray(y, dtype=float))
    k = y.size
    p = (np.arange(1, k + 1) - 0.35) / k
    a0 = float(y.mean())
    a1 = float(np.mean(y * (1.0 - p)))
    den = a0 - 2.0 * a1
    if den <= 0:
        return math.nan, math.nan
    return 2.0 - a0 / den, 2.0 * a0 * a1 / den
