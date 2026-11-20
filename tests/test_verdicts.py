"""Verdikte als reine Funktionen (`vwr_evaluation.verdict_*`): je Preset die erwartete Verdikt-Zeile, Schwellen an handgesetzten Werten, kein Effekt wird gemeldet, wo keiner ist (§11a Punkt 7)."""

import dataclasses

import pytest

import vwr_constants as C
import vwr_evaluation as V


def _settings(name, **extra):
    return V.Settings(**{**C.PRESETS[name], **extra})


def _codes(name):
    return [c for c, _ in V.analyse_lage(_settings(name)).hints]


def test_hinweise_je_preset():
    assert _codes("50 Jahre, 100er-Pegel") == ["extrapolation"]
    assert _codes("Kurze Reihe, 10 Jahre") == ["extrapolation", "kurze_reihe", "k_klein"]
    assert _codes("50 Jahre, 1000er-Pegel") == ["extrapolation"]
    assert _codes("Abhängig, θ = 0,2") == ["extrapolation", "wenig_cluster"]
    assert _codes("Trend 0,02") == ["extrapolation", "trend_nicht_erkennbar"]
    assert _codes("Schwerer Rand (Pareto-Verteilung)") == ["extrapolation"]
    assert _codes("Normal") == ["extrapolation"]                      # kein Preset ohne Effekt meldet einen Effekt
    assert _codes("Puffer jenseits der Daten") == ["extrapolation"]


def test_lage_verdikt_standard_nennt_exakten_anteil_und_band_von_stueck_3():
    kind, text = V.verdict_lage(V.analyse_lage(V.Settings()))
    assert kind == "warning" and "60,5 %" in text and "39,5 %" in text and "0,73- und 1,49-Fachen" in text and "„Wiederkehrwert“" in text and "Extrapolation" in text


def test_lage_verdikt_1000er_und_kurze_reihe():
    _, text = V.verdict_lage(V.analyse_lage(_settings("50 Jahre, 1000er-Pegel")))
    assert "95,1 %" in text and "0,56- und 2,21-Fachen" in text
    _, text = V.verdict_lage(V.analyse_lage(_settings("Kurze Reihe, 10 Jahre")))
    assert "90,4 %" in text and "nicht gemessen" in text and "Kurze Reihe" in text


def test_lage_verdikt_ohne_hinweis_ist_success():
    kind, text = V.verdict_lage(V.analyse_lage(V.Settings(m1=100, T1=100, k1=250)))
    assert kind == "success" and "Kein Hinweis" in text and "36,6 %" in text


def test_hinweistexte_nennen_ihre_zahl():
    texts = dict(V.analyse_lage(_settings("Abhängig, θ = 0,2")).hints)
    assert "30 Cluster" in texts["wenig_cluster"] and "eigene Festlegung" in texts["wenig_cluster"] and "„Spitzen-Cluster“" in texts["wenig_cluster"]
    texts = dict(V.analyse_lage(_settings("Trend 0,02")).hints)
    assert "62 Jahre" in texts["trend_nicht_erkennbar"] and "50" in texts["trend_nicht_erkennbar"]


@pytest.mark.parametrize("name,dist,years,share", [("50 Jahre, 100er-Pegel", "Pareto-Verteilung", "100", "98,8 %"), ("Schwerer Rand (Pareto-Verteilung)", "Pareto-Verteilung", "1000", "99,8 %"),
                                                   ("Kurze Reihe, 10 Jahre", "Exponential", "100", "63,5 %"), ("Normal", "Normal", "100", "11,2 %")])
def test_matrix_verdikt_je_preset(name, dist, years, share):
    s = _settings(name, R2=10)
    kind, text = V.verdict_matrix(s, V.analyse_live(s))
    assert kind == "info" and f"{dist}, {s.m2} Jahre, {years}-Jahres-Pegel" in text and share in text and "12 bis 14" in text and "kein Verfahren gewinnt durchgehend" in text


def test_matrix_verdikt_vehikel_b_sagt_dass_es_nicht_in_der_matrix_ist():
    s = V.Settings(dist2="vehikel_b", R2=10)
    _, text = V.verdict_matrix(s, V.analyse_live(s))
    assert "Vehikel B ist nicht in der eingefrorenen Matrix" in text and "gewinnt Gumbel 12 bis 14" in text


def test_puffer_verdikt_schwellen():
    kind, text = V.verdict_puffer(V.analyse_puffer(_settings("Puffer jenseits der Daten")))
    assert kind == "warning" and "0,670" in text and "n·p = 0,20" in text and "ein kleinerer Aufschlag ist dann ein höherer Grundwert" in text
    kind, text = V.verdict_puffer(V.analyse_puffer(V.Settings(case3="exp", n3=5000, p3=1e-3)))
    assert kind == "success" and "Ab n·p ≈ 5 genügt das empirische Quantil" in text
    kind, text = V.verdict_puffer(V.analyse_puffer(V.Settings()))
    assert kind == "info" and "ist kleiner" in text                     # Pareto-Verteilung, n·p = 2: der Gesamtpuffer der GPD ist kleiner (1,41 gegen 1,57), das steht so da


def test_puffer_verdikt_behauptet_keinen_effekt_der_gpd_wenn_er_nicht_da_ist():
    a = V.analyse_puffer(V.Settings())
    emp, gpd = a.rows["emp"], a.rows["gpd"]
    _, text = V.verdict_puffer(a)
    assert (gpd["tot"] >= emp["tot"]) == ("wird aber nicht kleiner" in text)


def test_cvar_verdikt_ampel_und_randfall():
    kind, text = V.verdict_cvar(V.analyse_cvar(V.Settings()))
    assert kind == "warning" and "0,847" in text and "0,2 %" in text and "cvar-demo" in text
    kind, text = V.verdict_cvar(V.analyse_cvar(_settings("Normal")))
    assert kind == "info" and "0,985" in text
    _, text = V.verdict_cvar(V.analyse_cvar(V.Settings(alpha4=0.999)))
    assert "unter 1" in text and "das Maximum der Szenarien" in text and "54,9 %" in text
    _, text = V.verdict_cvar(V.analyse_cvar(V.Settings(alpha4=0.9)))
    assert "das Maximum der Szenarien" not in text


def test_jedes_preset_hat_in_allen_vier_verdikten_text_und_gueltige_art():
    for name in C.PRESETS:
        s = dataclasses.replace(_settings(name), R2=10)
        for kind, text in (V.verdict_lage(V.analyse_lage(s)), V.verdict_matrix(s, V.analyse_live(s)), V.verdict_puffer(V.analyse_puffer(s)), V.verdict_cvar(V.analyse_cvar(s))):
            assert kind in ("success", "info", "warning") and len(text) > 40
