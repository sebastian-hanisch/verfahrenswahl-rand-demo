"""Kopierte und abgeschriebene Dateien: Vergleich mit den Quellen in den Nachbar-Demos (Demos „GEV an Blockmaxima“, „Überschreitungen und die GPD“, „Pegel statt Index“) und der Zufalls-Vorlage, solange sie im Portfolio-Ordner liegen (sonst übersprungen);
dazu feste Anker, damit die Kopien auch allein laufen."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PORTFOLIO = ROOT.parent


def _source(rel: str) -> Path:
    p = PORTFOLIO / rel
    if not p.exists():
        pytest.skip(f"{rel} liegt nicht im Portfolio-Ordner")
    return p


def _funcs(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}


def test_rng_ist_wortlaut_der_vorlage_bis_auf_die_kopfzeile():
    src = _source("sebastianhanisch-website/tools/templates/rng_splitmix64_exact.py")
    mine = (ROOT / "vwr_rng.py").read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith("# Wörtliche Kopie von sebastianhanisch-website/tools/templates/rng_splitmix64_exact.py")
    assert mine[1:] == src.read_text(encoding="utf-8").splitlines(), "die Kopie weicht von der Vorlage ab: Änderungen gehören in die Vorlage, danach neu kopieren"


def test_rng_feste_anker_ohne_die_vorlage():
    from vwr_rng import SplitMix64
    r = SplitMix64(0)
    assert [r.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    r = SplitMix64(42)
    assert [r.random().hex() for _ in range(2)] == ["0x1.7bae644c5fd6dp-1", "0x1.477f199d93378p-3"]


def test_gev_ist_wortlaut_von_gvb_gev():
    src = _source("gev-blockmaxima-demo/gvb_gev.py")
    assert (ROOT / "vwr_gev.py").read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


def test_blockfit_ist_gvb_fit_mit_umbenanntem_import():
    src = _source("gev-blockmaxima-demo/gvb_fit.py")
    assert (ROOT / "vwr_blockfit.py").read_text(encoding="utf-8") == src.read_text(encoding="utf-8").replace("gvb_gev", "vwr_gev")


def test_hill_ist_wortlaut_von_hlp_hill():
    src = _source("hill-pegel-demo/hlp_hill.py")
    assert (ROOT / "vwr_hill.py").read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


def test_vecrng_ist_hlp_vecrng_mit_umbenanntem_verweis():
    src = _source("hill-pegel-demo/hlp_vecrng.py")
    assert (ROOT / "vwr_vecrng.py").read_text(encoding="utf-8") == src.read_text(encoding="utf-8").replace("hlp_rng.py", "vwr_rng.py")


def test_gpd_funktionen_sind_wortlaut_von_gpu_algorithm():
    src = _funcs(_source("gpd-ueberschreitungen-demo/gpu_algorithm.py"))
    mine = _funcs(ROOT / "vwr_gpd.py")
    assert set(mine) == {"gpd_sf", "gpd_isf", "pot_level", "_golden_min", "gpd_ml", "gpd_pwm"}
    for name, dump in mine.items():
        assert dump == src[name], f"{name} weicht von gpu_algorithm.py ab"


def test_vehikel_b_tabelle_ist_byte_gleich_der_gemeinsamen_tabelle():
    src = _source("extremwert-planung/vormessung/ev_v13_vehikel_b_tabelle.json")
    assert (ROOT / "vwr_vehikel_b_tabelle.json").read_bytes() == src.read_bytes()
