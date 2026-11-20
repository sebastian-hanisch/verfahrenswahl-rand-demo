"""Die eingefrorenen Messungen laden: `vwr_grid.json` (Verfahrens-Matrix), `vwr_cvar.json` (CVaR-Raster), `vwr_puffer.json` (Puffer-Tabellen). Jede Datei trägt Quelle und Stand;
erzeugt und geprüft werden sie von `tools/vwr_make_grid.py`, `tools/vwr_make_cvar.py`, `tools/vwr_make_puffer.py` (je mit `--check`)."""

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CACHE: dict = {}


def _load(name: str) -> dict:
    if name not in _CACHE:
        _CACHE[name] = json.loads((_HERE / name).read_text(encoding="utf-8"))
    return _CACHE[name]


def grid() -> dict:
    return _load("vwr_grid.json")


def cvar() -> dict:
    return _load("vwr_cvar.json")


def puffer() -> dict:
    return _load("vwr_puffer.json")


def seed_keys() -> list:
    """Die Zufalls-Seed-Basen der Matrix in der Reihenfolge der Datei (die erste ist die geführte)."""
    return list(grid()["matrix"])


def matrix_cell(dist: str, m: int, target: float, seed_index: int = 0) -> dict:
    """{Verfahren: Statistik} der Zelle (Verteilung, Länge m, Ziel q) für eine der drei Zufalls-Seed-Basen."""
    g = grid()
    key = seed_keys()[seed_index]
    return g["matrix"][key]["cells"][f"{dist}|{m}"]["targets"][str(target)]


def matrix_edge(dist: str, m: int, seed_index: int = 0) -> tuple:
    """(Läufe mit ξ̂ der GEV-ML an der Grenze −1, Läufe) in der Zelle."""
    g = grid()
    c = g["matrix"][seed_keys()[seed_index]]["cells"][f"{dist}|{m}"]
    return c["gev_edge"], c["runs"]
