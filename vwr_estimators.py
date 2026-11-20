"""Die acht Verfahren der Matrix (numpy und Standardbibliothek, E18: scipy nur in den Tests) und die Verfahren des Puffers.

Matrix (diese Demo vergleicht, was die Demos „GEV an Blockmaxima“, „Überschreitungen und die GPD“ und „Pegel statt Index“ bauen): GEV-ML (ξ frei), GEV-L-Momente (ξ̂ geklammert) und Gumbel-ML (ξ = 0 fest) an die Blockmaxima; GPD-ML und GPD-PWM an die k = 50
größten Werte; GPD-ML mit k = 5 % von N; Hill-Weissman mit k = 50 und k = 5 % von N. Die Anpassungen selbst sind wörtliche Abschriften aus den Nachbar-Demos (`vwr_gev`, `vwr_blockfit`, `vwr_gpd`,
`vwr_hill`; `tests/test_copies.py` vergleicht sie). ξ-Konvention: ξ > 0 Fréchet-Typ; `scipy.stats.genextreme` hat c = −ξ, `genpareto` c = ξ.
Alle Pegel in Einheiten der Beobachtung; p ist die Überschreitungswahrscheinlichkeit je Beobachtung, q je Block."""

import math

import numpy as np

import vwr_blockfit as BF
import vwr_gev as G
import vwr_gpd as GP
import vwr_hill as H

MATRIX_METHODS = ("gev_ml", "gev_lmom", "gumbel", "gpd_ml50", "gpd_pwm50", "gpd_ml5", "hill50", "hill5")
METHOD_LABELS = {
    "gev_ml": "GEV-ML an die Blockmaxima (ξ frei)",
    "gev_lmom": "GEV-L-Momente an die Blockmaxima",
    "gumbel": "Gumbel-ML an die Blockmaxima (ξ = 0 fest)",
    "gpd_ml50": "GPD-ML, k = 50 Überschreitungen",
    "gpd_pwm50": "GPD-PWM, k = 50",
    "gpd_ml5": "GPD-ML, k = 5 % der Beobachtungen",
    "hill50": "Hill-Weissman, k = 50",
    "hill5": "Hill-Weissman, k = 5 % der Beobachtungen",
}
METHOD_SHORT = {"gev_ml": "GEV-ML", "gev_lmom": "GEV-L-Momente", "gumbel": "Gumbel (ξ = 0)", "gpd_ml50": "GPD-ML k=50", "gpd_pwm50": "GPD-PWM k=50", "gpd_ml5": "GPD-ML k=5 %",
                "hill50": "Hill k=50", "hill5": "Hill k=5 %"}
K_FIX = 50                  # feste eigene Festlegung (die Demo „Schwellenwahl“ behandelt die Schwellenwahl), gleich der Blockzahl bei m = 50
K_SHARE = 0.05              # Anteil der Beobachtungen für die zweite Schwelle


def k_share(n_obs: int) -> int:
    """k = 5 % der Beobachtungen (N = 100 m: 50, 100, 250, 500 bei m = 10, 20, 50, 100)."""
    return int(round(K_SHARE * n_obs))


def matrix_methods_for(n_obs: int) -> tuple:
    """Die Verfahren einer Zelle: bei m = 10 sind k = 50 und k = 5 % von N (= 50) dasselbe Verfahren, die Zeilen k = 5 % entfallen dort."""
    if k_share(n_obs) == K_FIX:
        return tuple(m for m in MATRIX_METHODS if m not in ("gpd_ml5", "hill5"))
    return MATRIX_METHODS


def block_maxima(x, ny: int) -> np.ndarray:
    """Jahresmaxima einer Reihe aus m Jahren zu je ny Beobachtungen (Block = aufeinanderfolgende ny Werte)."""
    x = np.asarray(x, float)
    return x[: (x.size // ny) * ny].reshape(-1, ny).max(axis=1)


def levels_matrix(x, ny: int, qs, methods=None, info=None) -> dict:
    """Pegel aller Verfahren der Matrix für die Ziele q (Überschreitungswahrscheinlichkeit je Block); nan, wenn ein Verfahren keine gültige Anpassung liefert.
    Rückgabe: {Verfahren: [Pegel zu q_1, q_2, ...]}. Alle Verfahren sehen dieselbe Reihe x (m Jahre zu ny Beobachtungen).
    `info` (ein dict, optional) bekommt `gev_edge`: ξ̂ der ML klebt an der Grenze −1."""
    x = np.asarray(x, float)
    n_obs = x.size
    methods = matrix_methods_for(n_obs) if methods is None else methods
    ps = [-math.expm1(math.log1p(-q) / ny) for q in qs]
    nan = [math.nan] * len(qs)
    out = {}
    mx = block_maxima(x, ny)
    if "gev_ml" in methods:
        try:
            f = BF.fit_ml(mx)
            if info is not None:
                info["gev_edge"] = bool(f.edge)          # ξ̂ klebt an der Grenze −1 (Nebenbedingung der ML)
            out["gev_ml"] = [float(G.pegel(q, f.mu, f.sigma, f.xi)) for q in qs]
        except (G.FitError, ValueError, OverflowError, ZeroDivisionError):
            out["gev_ml"] = nan
    if "gev_lmom" in methods:
        try:
            mu, s, xi = G.fit_lmoments(mx)
            out["gev_lmom"] = [float(G.pegel(q, mu, s, xi)) for q in qs]
        except (G.FitError, ValueError, OverflowError, ZeroDivisionError):
            out["gev_lmom"] = nan
    if "gumbel" in methods:
        try:
            mu, s = BF.gumbel_ml(mx)
            out["gumbel"] = [float(G.pegel(q, mu, s, 0.0)) for q in qs]
        except (G.FitError, ValueError, OverflowError, ZeroDivisionError):
            out["gumbel"] = nan
    xs = np.sort(x)
    for name, k, how in (("gpd_ml50", K_FIX, "ml"), ("gpd_pwm50", K_FIX, "pwm"), ("gpd_ml5", k_share(n_obs), "ml")):
        if name in methods:
            out[name] = pot_levels_sorted(xs, k, ps, how)
    for name, k in (("hill50", K_FIX), ("hill5", k_share(n_obs))):
        if name in methods:
            out[name] = hill_levels_sorted(xs, k, ps)
    return out


def pot_levels_sorted(xs, k: int, ps, how: str = "ml") -> list:
    """GPD an die k größten Werte (Schwelle = (k+1)-größter Wert), Pegel zu den Wahrscheinlichkeiten ps je Beobachtung; nan ohne gültige Anpassung."""
    n = xs.size
    if k < 2 or k >= n:
        return [math.nan] * len(ps)
    u = float(xs[n - k - 1])
    y = xs[n - k:] - u
    if not np.all(y > 0):
        y = np.maximum(y, 1e-12 * max(abs(u), 1.0))     # Gleichstände an der Schwelle: winziger positiver Abstand statt Division durch 0
    try:
        if how == "ml":
            xi, sig, _ = GP.gpd_ml(y)
        else:
            xi, sig = GP.gpd_pwm(y)
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return [math.nan] * len(ps)
    if not (math.isfinite(xi) and math.isfinite(sig) and sig > 0):
        return [math.nan] * len(ps)
    return [float(GP.pot_level(xi, sig, u, k / n, p)) for p in ps]


def hill_levels_sorted(xs, k: int, ps) -> list:
    """Hill-Weissman-Pegel aus den k größten Werten: X_(k+1) (k/(n p))^{H_k}; nan, wenn die Schwelle nicht positiv ist (der Hill-Schätzer verlangt X > 0)."""
    n = xs.size
    if k < 1 or k >= n or not xs[n - k - 1] > 0:
        return [math.nan] * len(ps)
    top = xs[::-1][: k + 1][None, :]
    return [float(H.weissman(top, k, n, p)[0]) for p in ps]


def log_error(level: float, truth: float) -> float:
    """ln(Pegel-Dach/wahr): 0 heißt richtig, ln 0,5 = −0,69 die Hälfte; nan bei ungültigem Pegel."""
    if not (math.isfinite(level) and level > 0):
        return math.nan
    return math.log(level / truth)


# --- Puffer jenseits der Daten (Experiment 3) ----------------------------------------------------------------------------------------------------
PUFFER_METHODS = ("emp", "normal", "lognormal", "gpd", "hill", "gev")
PUFFER_LABELS = {"emp": "empirisches Quantil", "normal": "Normal-Formel (Mittel + z·SD)", "lognormal": "Lognormal-Formel", "gpd": "GPD an die größten 5 %",
                 "hill": "Hill-Weissman an die größten 5 %", "gev": "GEV an Blockmaxima (Blöcke zu 100)"}
PUFFER_BLOCK = 100


def emp_index(n: int, p: float) -> int:
    """Ordnungsstatistik j = ⌈(1−p)(n+1)⌉, höchstens das Maximum (j = n)."""
    return min(n, int(math.ceil((1.0 - p) * (n + 1) - 1e-9)))


def buffer_levels(x, ps) -> dict:
    """Puffer (Quantile zur Überschreitungswahrscheinlichkeit p je Beobachtung) der sechs Verfahren; nan bei Fehlschlag."""
    from vwr_scenario import norm_isf
    x = np.asarray(x, float)
    n = x.size
    xs = np.sort(x)
    out = {"emp": [float(xs[emp_index(n, p) - 1]) for p in ps]}
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    out["normal"] = [mu + norm_isf(p) * sd for p in ps]
    if np.all(x > 0):
        lx = np.log(x)
        lm, ls = float(lx.mean()), float(lx.std(ddof=1))
        out["lognormal"] = [math.exp(lm + norm_isf(p) * ls) for p in ps]
    else:
        out["lognormal"] = [math.nan] * len(ps)
    k = max(10, int(round(0.05 * n)))
    out["gpd"] = pot_levels_sorted(xs, k, ps, "ml")
    out["hill"] = hill_levels_sorted(xs, k, ps)
    try:
        f = BF.fit_ml(block_maxima(x, PUFFER_BLOCK))
        out["gev"] = [float(G.pegel(-math.expm1(PUFFER_BLOCK * math.log1p(-p)), f.mu, f.sigma, f.xi)) for p in ps]
    except (G.FitError, ValueError, OverflowError, ZeroDivisionError):
        out["gev"] = [math.nan] * len(ps)
    return out
