"""Anpassungen an Block-Maxima: ML (eigener Nelder-Mead, nur numpy), L-Momente, Gumbel mit ξ = 0 fest. Kein scipy.

ML: Nebenbedingung ξ ≥ -1 (Wahl dieser Demo), KEINE obere Grenze für ξ, Start aus den L-Momenten (Hosking-Näherung, ξ auf [-0,6; 0,6] begrenzt),
zwei Nelder-Mead-Durchgänge. Die Streuung von ξ̂ bei kleinem m hängt an dieser fehlenden oberen Grenze (Ausreißer-Maß, siehe README)."""

import math
from dataclasses import dataclass

import numpy as np

import vwr_gev as G
from vwr_gev import FitError

METHODS = ("ML", "L-Momente", "Gumbel")
METHOD_LABELS = {"ML": "ML (GEV, ξ frei)", "L-Momente": "L-Momente (GEV, ξ frei)", "Gumbel": "Gumbel (ξ = 0 fest, ML)"}


@dataclass(frozen=True)
class Fit:
    method: str
    mu: float
    sigma: float
    xi: float
    edge: bool = False      # ML: ξ̂ klebt an der Nebenbedingung ξ = -1 (Optimierer)

    def level(self, p):
        return float(G.pegel(p, self.mu, self.sigma, self.xi))

    @property
    def endpoint(self):
        return G.endpoint(self.mu, self.sigma, self.xi)


def nm_minimize(f, x0, scale, tol_x=1e-8, tol_f=1e-11, maxit=1500):
    """Nelder-Mead mit numpy (Anfangssimplex: Startpunkt plus halbe Schrittweiten je Achse). Rückgabe (Minimalstelle, Minimum)."""
    n = len(x0)
    sim = np.vstack([x0] + [x0 + scale * 0.5 * np.eye(n)[i] for i in range(n)])
    fv = np.array([f(s) for s in sim])
    for _ in range(maxit):
        o = np.argsort(fv)
        sim, fv = sim[o], fv[o]
        if np.max(np.abs(sim[1:] - sim[0])) <= tol_x and np.max(np.abs(fv[1:] - fv[0])) <= tol_f:
            break
        c = sim[:-1].mean(axis=0)
        xr = c + (c - sim[-1])
        fr = f(xr)
        if fr < fv[0]:
            xe = c + 2 * (c - sim[-1])
            fe = f(xe)
            sim[-1], fv[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fv[-2]:
            sim[-1], fv[-1] = xr, fr
        else:
            xc = c + 0.5 * (xr - c) if fr < fv[-1] else c + 0.5 * (sim[-1] - c)
            fc = f(xc)
            if fc < min(fr, fv[-1]):
                sim[-1], fv[-1] = xc, fc
            else:
                sim[1:] = sim[0] + 0.5 * (sim[1:] - sim[0])
                fv[1:] = [f(s) for s in sim[1:]]
    i = int(np.argmin(fv))
    return sim[i], float(fv[i])


def _start(x):
    """Startwert (μ, ln σ, ξ): L-Momente mit der Hosking-Näherung, sonst Gumbel-Momente."""
    try:
        l1, l2, t3 = G.sample_lmoments(x)
        xi = float(np.clip(G.xi_from_tau3_hosking(t3), -0.6, 0.6))
        mu, s = G.gev_from_lmoments(l1, l2, xi)
        if s > 0 and math.isfinite(mu) and math.isfinite(s):
            t = np.array([mu, math.log(s), xi])
            if G.nll(t, x) < G.HUGE:
                return t
    except (FitError, ValueError, OverflowError, ZeroDivisionError):
        pass
    s = float(x.std(ddof=1)) * math.sqrt(6) / math.pi
    if not s > 0:
        raise FitError("Alle Block-Maxima sind gleich: keine Anpassung möglich.")
    return np.array([float(x.mean()) - G.EULER * s, math.log(s), 0.0])


def fit_ml(x, start=None):
    """ML-GEV (μ̂, σ̂, ξ̂) mit ξ ≥ -1; `edge` heißt: ξ̂ ≤ -0,999 laut Optimierer. Braucht m >= 2 und nicht alle Werte gleich."""
    x = np.asarray(x, float)
    if x.size < 2:
        raise FitError("Die ML-Anpassung braucht mindestens 2 Block-Maxima.")
    p0 = _start(x) if start is None else np.asarray(start, float)
    sc = np.array([float(x.std()), 0.1, 0.1])
    p1, f1 = nm_minimize(lambda t: G.nll(t, x), p0, sc, 1e-7, 1e-10)
    p2, f2 = nm_minimize(lambda t: G.nll(t, x), p1, sc * 0.1, 1e-8, 1e-11)
    p, f = (p2, f2) if f2 <= f1 else (p1, f1)
    if not f < G.HUGE / 10:
        raise FitError("Die ML-Anpassung fand keinen gültigen Punkt.")
    mu, ls, xi = p
    return Fit("ML", float(mu), float(math.exp(ls)), float(xi), edge=bool(xi <= G.XI_MIN + 1e-3))


def gumbel_ml(x):
    """Gumbel-ML (ξ = 0 fest): Newton auf σ in s - x̄ + E_w[x] = 0 mit Gewichten w ∝ exp(-x/s); Rückgabe (μ̂, σ̂)."""
    x = np.asarray(x, float)
    if x.size < 2:
        raise FitError("Die Gumbel-Anpassung braucht mindestens 2 Block-Maxima.")
    sd = float(x.std(ddof=1))
    if not sd > 0:
        raise FitError("Alle Block-Maxima sind gleich: keine Anpassung möglich.")
    xc = x - x.mean()
    s = sd * math.sqrt(6) / math.pi
    for _ in range(200):
        w = np.exp(-(xc - xc.min()) / s)
        sw = w.sum()
        e = float(np.sum(w * xc) / sw)
        var = float(np.sum(w * xc * xc) / sw) - e * e
        f = s + e
        d = 1 + var / (s * s)
        s_new = s - f / d
        if s_new <= 0:
            s_new = s / 2
        if abs(s_new - s) < 1e-13 * s:
            s = s_new
            break
        s = s_new
    mu = float(x.mean()) - s * math.log(float(np.mean(np.exp(-xc / s))) )
    return float(mu), float(s)


def fit(method, x, klammer=True):
    """Anpassung nach Verfahren: ML, L-Momente (klammer wie in vwr_gev.fit_lmoments) oder Gumbel."""
    x = np.asarray(x, float)
    if method == "ML":
        return fit_ml(x)
    if method == "L-Momente":
        mu, s, xi = G.fit_lmoments(x, klammer)
        return Fit("L-Momente", mu, s, xi)
    if method == "Gumbel":
        mu, s = gumbel_ml(x)
        return Fit("Gumbel", mu, s, 0.0)
    raise ValueError(f"unbekanntes Verfahren {method!r}")
