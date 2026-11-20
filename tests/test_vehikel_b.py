"""Vehikel B: die kopierte Gewichtstabelle gegen eine eigene Aufzählung aus der Definition (tools/vwr_make_vehicle.py), wahre Pegel, Überlebensfunktion und Ziehung;
die Konstanten gegen `verweilzeit-demo` (nur gelesen, kein Import; Test überspringt sich, wenn der Ordner fehlt)."""

import ast
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import vwr_make_vehicle as TOOL  # noqa: E402
import vwr_scenario as S  # noqa: E402
import vwr_vecrng as VR  # noqa: E402
import vwr_vehicle as VB  # noqa: E402


def test_tabelle_hat_den_festen_sha256_und_1224_komponenten_mit_summe_1():
    data = (ROOT / "vwr_vehikel_b_tabelle.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == VB.TABLE_SHA256 == VB.QUELLE["sha256"] == "669f821c78e0bf38c47fa846232f27cd2ff126f4fceadca5369a5c9af5551c9f"
    t = VB.table()
    assert t.shape == (1224, 3) and abs(t[:, 0].sum() - 1.0) < 1e-12 and np.all(t[:, 0] > 0)
    assert {s: int(np.sum(t[:, 2] == s)) for s in (0.55, 0.25, 0.35, 0.9)} == {0.55: 48, 0.25: 576, 0.35: 576, 0.9: 24}


def test_eigene_aufzaehlung_aus_der_definition_gleich_der_kopierten_tabelle():
    comps, steps, zero = TOOL.aufzaehlung()
    assert steps == 4704 and zero == 2328 and steps - zero == 2376 and len(comps) == 1224
    tab = VB.table()
    a = comps[np.lexsort((comps[:, 1], comps[:, 2]))]
    b = tab[np.lexsort((tab[:, 1], tab[:, 2]))]
    assert np.max(np.abs(a - b)) < 1e-12


def test_kundeneffekte_sind_default_rng_12345_normal_0_0_3_12():
    assert np.array_equal(np.array(TOOL.UK), np.random.default_rng(12345).normal(0, 0.30, 12))


def test_konstanten_gleich_verweilzeit_demo_falls_vorhanden():
    path = ROOT.parent / "verweilzeit-demo" / "vwz_constants.py"
    if not path.exists():
        pytest.skip("verweilzeit-demo liegt nicht im Portfolio-Ordner")
    names = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                names[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    assert names["ART_P"] == TOOL.ART_P and names["K_KUNDEN"] == TOOL.K_KUNDEN and names["U_SCALE"] == TOOL.U_SCALE
    assert names["S_BASE"] == TOOL.S_BASE and names["P_ZOLL"] == TOOL.P_ZOLL


def test_wahre_pegel_der_zielgroessen_der_linie():
    # (b) Tagesmaximum aus 100 Containern, einmal in 1 000 Tagen; (a) je Container 1e-5; 100-Tage-Pegel des Tagesmaximums (n_y = 100 Container je Tag, q = 0,01)
    assert S.true_level("vehikel_b", 0.001) == pytest.approx(287.123, abs=5e-4)
    assert S.isf("vehikel_b", 1e-5) == pytest.approx(287.158, abs=5e-4)
    assert S.true_level("vehikel_b", 0.01) == pytest.approx(155.128, abs=5e-4)


def test_ueberlebensfunktion_per_interpolation_gegen_die_direkte_mischung():
    rng = np.random.default_rng(5)
    xt = np.exp(rng.uniform(math.log(0.5), math.log(1500), 200))
    direct = np.array([VB.sf(v) for v in xt])
    rel = np.abs(VB.sf_vec(xt) / direct - 1)[direct > 1e-12]
    assert rel.max() < 1e-3
    assert VB.sf_vec(np.array([0.0, 1e9])).tolist() == [1.0, 0.0]


def test_ziehung_trifft_ueberlebensfunktion_und_log_median_im_band():
    n = 400_000
    s = VR.Stream(77)
    x = VB.draw(s.random(n), s.normal(n))
    for t in (3.0, 10.0, 30.0, 100.0):
        p = VB.sf(t)
        assert abs(float(np.mean(x > t)) - p) < 4 * math.sqrt(p * (1 - p) / n), t
    tab = VB.table()
    assert abs(float(np.mean(np.log(x))) - float(np.sum(tab[:, 0] * tab[:, 1]))) < 4 * float(np.std(np.log(x))) / math.sqrt(n)
