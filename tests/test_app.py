"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- eigene Tests (die Vorlage oben zählt nur `st.slider`; diese Demo hat auch Auswahl-Regler) -------------------------------------------------------------------
import vwr_constants as C  # noqa: E402


OPTIONS = {"T1": C.T_OPTIONS, "theta1": C.THETA_OPTIONS, "beta1": C.BETA_OPTIONS, "m2": C.M2_OPTIONS, "target2": C.TARGET2_OPTIONS, "n3": C.N3_OPTIONS, "p3": C.P3_OPTIONS,
           "n4": C.N4_OPTIONS, "alpha4": C.ALPHA4_OPTIONS}


def test_every_select_slider_at_its_first_and_last_option_does_not_raise():
    n = len(_fresh_app().select_slider)
    assert n == len(OPTIONS)
    for index in range(n):
        for edge in (0, -1):
            at = _fresh_app()
            sl = at.select_slider[index]
            sl.set_value(OPTIONS[sl.key][edge]).run(timeout=TIMEOUT)
            assert not at.exception, (sl.label, edge, [str(e) for e in at.exception])


def test_every_option_of_every_selectbox_and_radio_does_not_raise():
    n_box, n_radio = len(_fresh_app().selectbox), len(_fresh_app().radio)
    assert n_box == 3 and n_radio >= 1
    for index in range(n_box):
        for option in _fresh_app().selectbox[index].options:
            at = _fresh_app()
            at.selectbox[index].select(option).run(timeout=TIMEOUT)
            assert not at.exception, (index, option, [str(e) for e in at.exception])
    for index in range(n_radio):
        for option in _fresh_app().radio[index].options:
            at = _fresh_app()
            at.radio[index].set_value(option).run(timeout=TIMEOUT)
            assert not at.exception, (index, option, [str(e) for e in at.exception])


def test_every_preset_shows_four_experiments_four_verdicts_and_the_table_expander():
    for name in C.PRESETS:
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        titles = [h.value for h in at.markdown if h.value.startswith("## ")] + [s.value for s in at.subheader]
        assert sum("Experiment" in t for t in titles) == 4, (name, titles)
        assert len([m for m in at.markdown if m.value.startswith("**") and m.value[2:3] in "1234" and m.value[3:4] == "."]) == 4
        assert any(e.label == "📋 Die ganze Entscheidungstabelle (alle Zeilen)" for e in at.expander)
        assert any(e.label == "📐 Mathematische Formulierung" for e in at.expander)


def test_preset_trend_meldet_den_hinweis_und_normal_nur_die_extrapolation():
    at = _fresh_app()
    at.sidebar.button(key="preset_Trend 0,02").click().run(timeout=TIMEOUT)
    assert any("Trend nicht erkennbar" in w.value for w in at.warning)
    at.sidebar.button(key="preset_Normal").click().run(timeout=TIMEOUT)
    hint_warnings = [w.value for w in at.warning if w.value.startswith(("Extrapolation", "Kurze Reihe", "Wenige Überschreitungen", "Wenig Cluster", "Trend nicht"))]
    assert len(hint_warnings) == 1 and hint_warnings[0].startswith("Extrapolation")


def test_permalink_uebernimmt_werte_und_begrenzt_sie():
    at = AppTest.from_file(APP_PATH)
    at.query_params["m1"] = "100000"
    at.query_params["T1"] = "55"
    at.query_params["rand1"] = "unsinn"
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert at.slider(key="m1").value == C.M1_MAX and at.select_slider(key="T1").value == C.DEFAULT_T and at.radio(key="rand1").value == C.DEFAULT_RAND


def _live_table(at):
    return next(d.value for d in at.dataframe if "Median Pegel-Dach/wahr" in d.value.columns)


def test_live_laeufe_zeigen_acht_verfahren_bei_m_20_und_sechs_bei_m_10():
    at = _fresh_app()
    at.select_slider(key="m2").set_value(10).run(timeout=TIMEOUT)
    assert not at.exception and len(_live_table(at)) == 6
    at.select_slider(key="m2").set_value(20).run(timeout=TIMEOUT)
    assert not at.exception and len(_live_table(at)) == 8
