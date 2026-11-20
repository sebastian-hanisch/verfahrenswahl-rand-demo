"""Vehikel B: die Verweilzeit-Mischung der Linie „Extremwerte erklärt“ (ausgedacht, kein Echtdatenbezug) als eingefrorene Gewichtstabelle, nur für den Puffer jenseits der Daten (Experiment 3).

Quelle: gemeinsame Definition der Linie (Demo „Überschreitungen und die GPD“, `gpd-ueberschreitungen-demo`, Stand 2026-11-20): exakte Aufzählung aus Art (Import 0,35 / Export 0,30 / Umschlag 0,25 /
Leer 0,10) × 12 Kunden × Zoll × Wochenende × Vorlauf (Gauß-Legendre mit 48 Knoten), je Komponente ln(Verweilzeit in Tagen) ~ N(μ, s²). 4 704 Aufzählungsschritte, davon 2 328 mit Gewicht 0,
2 376 mit Gewicht, nach dem Zusammenfassen gleicher (μ, s) 1 224 Komponenten (48 Import, 576 Export, 576 Umschlag, 24 Leer). `vwr_vehikel_b_tabelle.json` ist eine byte-gleiche Kopie der
gemeinsamen Tabelle (SHA-256 unten); `tools/vwr_make_vehicle.py --check` zählt sie aus der Definition neu auf. Hier gebraucht: Überlebensfunktion und Ziehen (Quantile je Beobachtung
zu p = 10⁻² bis 10⁻⁴), nicht die Zielgrößen (a) und (b) der Demos davor."""

import json
import math
from pathlib import Path

import numpy as np

TABLE_FILE = Path(__file__).with_name("vwr_vehikel_b_tabelle.json")
TABLE_SHA256 = "669f821c78e0bf38c47fa846232f27cd2ff126f4fceadca5369a5c9af5551c9f"
QUELLE = {"komponenten": "gemeinsame Definition der Linie „Extremwerte erklärt“, Demo „Überschreitungen und die GPD“ (Vehikel B), Stand 2026-11-20",
          "gewichte": "Aufzählung Art × Kunde × Zoll × Wochenende × Vorlauf (48 Gauß-Legendre-Knoten), Stand 2026-11-20",
          "mu_s": "log-Median und Streuung je Komponente wie `verweilzeit-demo` (nur gelesen), Stand 2026-11-20",
          "sha256": TABLE_SHA256}

_CACHE: dict = {}
_ERFC = np.vectorize(math.erfc, otypes=[float])


def norm_sf_vec(z) -> np.ndarray:
    """P(Z > z) für Standardnormal, elementweise (stdlib `math.erfc`)."""
    return 0.5 * _ERFC(np.asarray(z, float) / math.sqrt(2.0))


def table() -> np.ndarray:
    """(Gewicht, μ, s) je Komponente als Feld 1224 × 3."""
    if "t" not in _CACHE:
        data = json.loads(TABLE_FILE.read_text(encoding="utf-8"))
        arr = np.array(data["komponenten"], float)
        if arr.shape != (data["n_komponenten"], 3):
            raise ValueError("Tabelle des Vehikels B ist beschädigt")
        _CACHE["t"] = arr
        _CACHE["cw"] = np.cumsum(arr[:, 0]) / float(np.sum(arr[:, 0]))
    return _CACHE["t"]


def sf(x: float) -> float:
    """Überlebensfunktion P(Verweilzeit > x) der Mischung."""
    t = table()
    z = (math.log(x) - t[:, 1]) / t[:, 2]
    return float(np.sum(t[:, 0] * norm_sf_vec(z)))


def quantile(q: float) -> float:
    """x mit P(Verweilzeit > x) = q (Bisektion in ln x, Genauigkeit 1e-13)."""
    key = ("q", q)
    if key in _CACHE:
        return _CACHE[key]
    lq = math.log(q)
    lo, hi = -3.0, 14.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if math.log(sf(math.exp(mid))) > lq:
            lo = mid
        else:
            hi = mid
    _CACHE[key] = math.exp(0.5 * (lo + hi))
    return _CACHE[key]


def _grid():
    if "grid" not in _CACHE:
        lx = np.linspace(math.log(0.02), math.log(5000.0), 4001)
        ls = np.array([math.log(max(sf(math.exp(v)), 1e-300)) for v in lx])
        _CACHE["grid"] = (lx, ls)
    return _CACHE["grid"]


def sf_vec(x) -> np.ndarray:
    """Überlebensfunktion für viele Pegel auf einmal: lineare Interpolation von ln sf über ln x (4 001 Stützstellen, 0,02 bis 5 000 Tage); darüber 0, darunter 1.
    Die Genauigkeit gegen `sf` prüft `tests/test_vehikel_b.py` (größter relativer Fehler unter 1e-3)."""
    x = np.asarray(x, float)
    lx, ls = _grid()
    out = np.ones_like(x)
    pos = x > 0
    v = np.log(np.where(pos, x, 1.0))
    inside = pos & (v >= lx[0]) & (v <= lx[-1])
    out[inside] = np.exp(np.interp(v[inside], lx, ls))
    out[pos & (v > lx[-1])] = 0.0
    return out


def draw(u_comp: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Verweilzeiten aus gleichverteilten u_comp (Komponente über die Summenverteilung der Gewichte) und Normalzahlen z."""
    t = table()
    c = np.minimum(np.searchsorted(_CACHE["cw"], u_comp), len(t) - 1)
    return np.exp(t[c, 1] + t[c, 2] * z)


def mean() -> float:
    t = table()
    return float(np.sum(t[:, 0] * np.exp(t[:, 1] + t[:, 2] ** 2 / 2.0)))
