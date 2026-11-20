"""CVaR (Erwartungsfehlbetrag, Expected Shortfall) aus gleich wahrscheinlichen Szenarien gegen den exakten: Einordnung für den Tail (Experiment 4). Nur numpy und Standardbibliothek.

Empirisch in der Form von Rockafellar und Uryasev (die Form der LPs der `cvar-demo`): ζ = das ⌈α n⌉-kleinste Szenario, CVaR = ζ + Σ max(x_i − ζ, 0) / ((1−α) n). Ist (1−α)·n < 1, ist der empirische CVaR das
Maximum der Szenarien. Exakt: geschlossene Formeln für Normal, Exponential, Lognormal σ = 1, Pareto-Verteilung α = 2 und Student-t ν = 3 (die Tests rechnen sie per Quadratur der Quantilfunktion nach).
Die fünf Verteilungen sind ausgedacht; die Szenarien der `cvar-demo` sind hier nicht nachgebaut."""

import math

import numpy as np

import vwr_gpd as GP
import vwr_scenario as S

CVAR_DISTS = ("normal", "exp", "lognormal", "pareto", "t3")
CVAR_LABELS = {"normal": "Normal", "exp": "Exponential", "lognormal": "Lognormal σ = 1", "pareto": "Pareto-Verteilung α = 2", "t3": "Student-t, ν = 3"}
ALPHAS = (0.90, 0.95, 0.99, 0.999)


def es_exact(dist: str, alpha: float) -> float:
    """Erwartungsfehlbetrag zum Niveau α aus F: E[X | X > VaR_α]."""
    a = 1.0 - alpha
    if dist == "normal":
        z = S.norm_isf(a)
        return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) / a
    if dist == "exp":
        return 1.0 - math.log(a)
    if dist == "lognormal":
        z = S.norm_isf(a)
        return math.exp(0.5) * S.norm_sf(z - 1.0) / a
    if dist == "pareto":
        return 2.0 * a ** -0.5
    if dist == "t3":
        t = S.t3_isf(a)
        dens = 2.0 / (math.pi * math.sqrt(3.0) * (1.0 + t * t / 3.0) ** 2)
        return dens / a * (3.0 + t * t) / 2.0
    raise ValueError(f"unbekannte Verteilung {dist!r}")


def cvar_index(n: int, alpha: float) -> int:
    """⌈α n⌉ (1-basiert), gegen Gleitkomma-Rauschen gerundet."""
    return int(math.ceil(round(alpha * n, 9)))


def cvar_empirical_rows(x, alpha: float) -> np.ndarray:
    """Rockafellar/Uryasev-CVaR je Zeile einer Matrix (Läufe × Szenarien); ein Vektor gilt als eine Zeile."""
    x = np.atleast_2d(np.asarray(x, float))
    n = x.shape[1]
    xs = np.sort(x, axis=1)
    zeta = xs[:, cvar_index(n, alpha) - 1]
    return zeta + np.sum(np.maximum(xs - zeta[:, None], 0.0), axis=1) / ((1.0 - alpha) * n)


def cvar_empirical(x, alpha: float) -> float:
    return float(cvar_empirical_rows(x, alpha)[0])


def prob_no_scenario_above_quantile(alpha: float, n: int) -> float:
    """Wahrscheinlichkeit, dass n Szenarien nicht einmal das α-Quantil überschreiten: α^n (exakt, verteilungsfrei)."""
    return alpha ** n


def gpd_es(x, k: int, alpha: float) -> float:
    """CVaR mit einer GPD an die k größten Szenarien: VaR/(1−ξ) + (σ − ξ u)/(1−ξ); nan bei ξ̂ ≥ 1 (Mittel nicht schätzbar) oder ohne Anpassung."""
    xs = np.sort(np.asarray(x, float))
    n = xs.size
    u = float(xs[n - k - 1])
    y = xs[n - k:] - u
    if not np.all(y > 0):
        y = np.maximum(y, 1e-12 * max(abs(u), 1.0))
    xi, sig, _ = GP.gpd_ml(y)
    if not math.isfinite(xi) or xi >= 1.0:
        return math.nan
    var = GP.pot_level(xi, sig, u, k / n, 1.0 - alpha)
    return float(var / (1.0 - xi) + (sig - xi * u) / (1.0 - xi))


def gpd_k(n: int) -> int:
    """GPD an die oberen 5 % der Szenarien, mindestens 20 Werte."""
    return max(20, int(math.ceil(0.05 * n)))
