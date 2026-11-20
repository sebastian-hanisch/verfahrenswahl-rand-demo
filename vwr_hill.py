"""Hill-Schätzer und Weissman-Pegel aus den k größten Werten (numpy; Einheit des Index: ξ = 1/α, nicht der Dichte-Exponent 1 + 1/H).

Konvention: `top` ist eine Matrix (Läufe × (K+1)) der K+1 größten Werte je Lauf, absteigend geordnet; die Schwelle bei k Überschreitungen ist der (k+1)-größte
Wert `top[:, k]`. Der Hill-Schätzer H_k = (1/k) Σ_{i=1..k} ln(X_(i) / X_(k+1)) verlangt positive Werte (Meldung statt Zahl bei X ≤ 0).
Weissman-Pegel: x̂_p = X_(k+1) · (k / (n p))^{H_k}, der Pegel mit der Überschreitungswahrscheinlichkeit p je Beobachtung (n Beobachtungen insgesamt)."""

import math

import numpy as np


def top_sorted(x: np.ndarray, kmax: int) -> np.ndarray:
    """Die kmax+1 größten Werte je Zeile, absteigend. x: (Läufe, n) oder (n,)."""
    x = np.atleast_2d(np.asarray(x, float))
    n = x.shape[1]
    if not 1 <= kmax <= n - 1:
        raise ValueError(f"kmax muss zwischen 1 und n-1 = {n - 1} liegen, ist {kmax}")
    part = np.partition(x, n - kmax - 1, axis=1)[:, n - kmax - 1:]
    return np.sort(part, axis=1)[:, ::-1]


def check_positive(top: np.ndarray, k: int) -> None:
    if not np.all(top[:, :k + 1] > 0):
        raise ValueError("Der Hill-Schätzer verlangt positive Werte (X > 0); die Schwelle liegt bei einem Wert <= 0.")


def hill_all(top: np.ndarray, kmax: int | None = None) -> np.ndarray:
    """H_k für k = 1 ... kmax, je Zeile (Läufe × kmax); Spalte k-1 gehört zu k."""
    kmax = top.shape[1] - 1 if kmax is None else kmax
    check_positive(top, kmax)
    lg = np.log(top[:, :kmax + 1])
    k = np.arange(1, kmax + 1)
    return np.cumsum(lg[:, :kmax], axis=1) / k - lg[:, 1:kmax + 1]


def hill(top: np.ndarray, k: int) -> np.ndarray:
    """H_k je Zeile."""
    if k < 1 or k >= top.shape[1]:
        raise ValueError("k muss zwischen 1 und K liegen (K+1 = Spaltenzahl von top)")
    check_positive(top, k)
    return np.mean(np.log(top[:, :k]), axis=1) - np.log(top[:, k])


def weissman(top: np.ndarray, k: int, n: int, p: float) -> np.ndarray:
    """Weissman-Pegel je Zeile: X_(k+1) (k/(n p))^{H_k}. Bei k/(n p) = 1 genau X_(k+1) (keine Extrapolation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p muss zwischen 0 und 1 liegen")
    return top[:, k] * (k / (n * p)) ** hill(top, k)


def weissman_all(top: np.ndarray, n: int, p: float, kmax: int | None = None) -> np.ndarray:
    """Weissman-Pegel für k = 1 ... kmax (Läufe × kmax)."""
    kmax = top.shape[1] - 1 if kmax is None else kmax
    k = np.arange(1, kmax + 1)
    h = hill_all(top, kmax)
    return top[:, 1:kmax + 1] * (k[None, :] / (n * p)) ** h


def spannweite_dichte_exponent(top: np.ndarray, n: int, anteile: np.ndarray | None = None, min_werte: int = 100) -> np.ndarray:
    """Spannweite (max − min) von 1 + 1/H_k über die Tail-Anteile q = 2 % bis 50 % je Zeile: das Stabilitätsmaß der Nachbar-Demo `potenzgesetz-test-demo`
    (10 log-gleiche Stufen mit je mindestens `min_werte` Werten; ihre Konvention: „stabil“ bis 0,3). `top` braucht mindestens k_max + 1 Spalten."""
    if anteile is None:
        anteile = np.geomspace(0.02, 0.5, 10)
    ks = np.round(anteile * n).astype(int)
    ks = ks[ks >= min_werte]
    h = hill_all(top, int(ks.max()))[:, ks - 1]
    d = 1.0 + 1.0 / h
    return d.max(axis=1) - d.min(axis=1)


def L_extrapolation(k: int, n: int, p: float) -> float:
    """L = ln(k/(n p)): der Abstand der Extrapolation, mit dem der Index-Fehler multipliziert wird."""
    return math.log(k / (n * p))
