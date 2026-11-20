"""Verteilungen der Matrix, des CVaR und des Puffers: Ziehen (SplitMix64, numpy-Vektorform), exakte Überlebensfunktion und exakter Pegel. Alles ausgedacht oder Lehrbuch, kein Betriebsbezug.

Die Matrix hat vier Verteilungen: Exponential (ξ = 0, GPD exakt), Pareto-Verteilung α = 2 (ξ = 0,5), Normal (leichter Rand, ξ_u < 0) und Lognormal σ = 1 (ξ = 0 im Grenzwert, lokal schwerer).
Der CVaR kommt mit Student-t ν = 3 dazu, der Puffer mit Normal(100, 20) und Vehikel B. ‚Wahr‘ ist immer exakt aus F bzw. F^n, nie aus GEV-Parametern.
Bezugsmenge des Pegels: je Beobachtung (Überschreitungswahrscheinlichkeit p) oder je Block (q); die Umrechnung für Blöcke zu n_y Beobachtungen ist p = 1 − (1 − q)^(1/n_y), also F^{n_y}(x) = 1 − q."""

import math
from statistics import NormalDist

import numpy as np

import vwr_vecrng as VR

_ND = NormalDist()
NY = 100                                   # Beobachtungen je Jahr (Block) in der Matrix; „Wiederkehrwert“ zählt 365 Tageswerte je Jahr

MATRIX_DISTS = ("exp", "pareto", "normal", "lognormal")
DIST_LABELS = {"exp": "Exponential (ξ = 0)", "pareto": "Pareto-Verteilung α = 2 (ξ = 0,5)", "normal": "Normal", "lognormal": "Lognormal σ = 1",
               "t3": "Student-t, ν = 3", "normal_100_20": "Normal(100, 20)", "vehikel_b": "Vehikel B (Verweilzeit-Mischung)"}
DIST_SHORT = {"exp": "Exponential", "pareto": "Pareto-Verteilung", "normal": "Normal", "lognormal": "Lognormal"}


# --- Überlebensfunktion und exakter Pegel (stdlib) ---------------------------------------------------------------------------------------------
def norm_isf(q: float) -> float:
    """z mit P(Z > z) = q (stdlib `inv_cdf`); für kleine q über die Symmetrie, damit 1 − q nicht auf 1 gerundet wird."""
    return -_ND.inv_cdf(q) if q < 0.5 else _ND.inv_cdf(1.0 - q)


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def sf(dist: str, x: float) -> float:
    """Überlebensfunktion P(X > x)."""
    if dist == "exp":
        return math.exp(-x) if x > 0 else 1.0
    if dist == "pareto":
        return x ** -2.0 if x >= 1.0 else 1.0
    if dist == "normal":
        return norm_sf(x)
    if dist == "lognormal":
        return norm_sf(math.log(x)) if x > 0 else 1.0
    if dist == "normal_100_20":
        return norm_sf((x - 100.0) / 20.0)
    if dist == "t3":
        return 1.0 - t3_cdf(x)
    if dist == "vehikel_b":
        import vwr_vehicle as VB
        return VB.sf(x) if x > 0 else 1.0
    raise ValueError(f"unbekannte Verteilung {dist!r}")


def isf(dist: str, p: float) -> float:
    """x mit P(X > x) = p (exakt bzw. mit der Genauigkeit der stdlib)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p muss zwischen 0 und 1 liegen")
    if dist == "exp":
        return -math.log(p)
    if dist == "pareto":
        return p ** -0.5
    if dist == "normal":
        return norm_isf(p)
    if dist == "lognormal":
        return math.exp(norm_isf(p))
    if dist == "normal_100_20":
        return 100.0 + 20.0 * norm_isf(p)
    if dist == "t3":
        return t3_isf(p)
    if dist == "vehikel_b":
        import vwr_vehicle as VB
        return VB.quantile(p)
    raise ValueError(f"unbekannte Verteilung {dist!r}")


def t3_cdf(x: float) -> float:
    """Verteilungsfunktion der Student-t-Verteilung mit ν = 3 in geschlossener Form: 1/2 + (1/π)(x/(√3 (1 + x²/3)) + arctan(x/√3))."""
    s3 = math.sqrt(3.0)
    return 0.5 + (x / (s3 * (1.0 + x * x / 3.0)) + math.atan(x / s3)) / math.pi


def t3_isf(p: float) -> float:
    """x mit P(T > x) = p (Bisektion auf der Symmetrie, Genauigkeit 1e-13 relativ)."""
    if p > 0.5:
        return -t3_isf(1.0 - p)
    lo, hi = 0.0, 1.0
    while t3_sf_pos(hi) > p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t3_sf_pos(mid) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def t3_sf_pos(x: float) -> float:
    """P(T > x) für x ≥ 0 ohne Auslöschung: 1/2 − (1/π)(x/(√3 (1 + x²/3)) + arctan(x/√3)) = (1/π)(arctan(√3/x) − √3 x/(3 + x²)) für x > 0."""
    if x <= 0.0:
        return 0.5 - (x / (math.sqrt(3.0) * (1.0 + x * x / 3.0)) + math.atan(x / math.sqrt(3.0))) / math.pi
    return (math.atan(math.sqrt(3.0) / x) - math.sqrt(3.0) * x / (3.0 + x * x)) / math.pi


def p_per_obs(q: float, ny: int = NY) -> float:
    """Überschreitungswahrscheinlichkeit je Beobachtung zu q je Block aus ny Beobachtungen: p = 1 − (1 − q)^(1/ny) (log-stabil)."""
    return -math.expm1(math.log1p(-q) / ny)


def true_level(dist: str, q: float, ny: int = NY) -> float:
    """Wahrer Pegel zur Überschreitungswahrscheinlichkeit q je Block (Block = ny Beobachtungen): F^ny(x) = 1 − q."""
    return isf(dist, p_per_obs(q, ny))


def block_cdf(dist: str, x: float, ny: int = NY) -> float:
    """F^ny(x), die exakte Verteilung des Blockmaximums (Orakel der wahren Pegel)."""
    return (1.0 - sf(dist, x)) ** ny


# --- Ziehen ------------------------------------------------------------------------------------------------------------------------------------
def draw(dist: str, stream: VR.Stream, shape) -> np.ndarray:
    """Zufallsfeld der Form `shape` aus dem Strom `stream` (SplitMix64, numpy-Vektorform; Gleichverteilung bitgleich zur Vorlage, Normalzahlen auf 1e-12)."""
    n = int(np.prod(shape))
    if dist == "exp":
        x = -np.log1p(-stream.random(n))
    elif dist == "pareto":
        x = (1.0 - stream.random(n)) ** -0.5
    elif dist == "normal":
        x = stream.normal(n)
    elif dist == "lognormal":
        x = np.exp(stream.normal(n))
    elif dist == "normal_100_20":
        x = 100.0 + 20.0 * stream.normal(n)
    elif dist == "t3":
        z = stream.normal(3 * n).reshape(n, 3)
        x = stream.normal(n) / np.sqrt(np.sum(z * z, axis=1) / 3.0)
    elif dist == "vehikel_b":
        import vwr_vehicle as VB
        u = stream.random(n)
        x = VB.draw(u, stream.normal(n))
    else:
        raise ValueError(f"unbekannte Verteilung {dist!r}")
    return x.reshape(shape)


def cell_seed(base: int, *tags: int) -> int:
    """Zufalls-Seed einer Zelle aus einer Basis und ganzzahligen Marken (reine Arithmetik, auf jeder Plattform gleich)."""
    s = int(base)
    for t in tags:
        s = (s * 1_000_003 + int(t) + 1) % (1 << 62)
    return s
