"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import vwr_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None     # nur zulässige Werte (Auswahl-Regler)


SETTING_SPECS = {
    "m1": SettingSpec("m1", int, C.DEFAULT_M1, C.M1_MIN, C.M1_MAX),
    "T1": SettingSpec("T1", int, C.DEFAULT_T, choices=C.T_OPTIONS),
    "k1": SettingSpec("k1", int, C.DEFAULT_K1, C.K1_MIN, C.K1_MAX),
    "theta1": SettingSpec("th1", float, C.DEFAULT_THETA, choices=C.THETA_OPTIONS),
    "beta1": SettingSpec("b1", float, C.DEFAULT_BETA, choices=C.BETA_OPTIONS),
    "rand1": SettingSpec("r1", str, C.DEFAULT_RAND, choices=C.RAND_OPTIONS),
    "dist2": SettingSpec("d2", str, C.DEFAULT_DIST2, choices=C.DIST2_OPTIONS),
    "m2": SettingSpec("m2", int, C.DEFAULT_M2, choices=C.M2_OPTIONS),
    "target2": SettingSpec("t2", int, C.DEFAULT_TARGET2, choices=C.TARGET2_OPTIONS),
    "R2": SettingSpec("R2", int, C.DEFAULT_R2, C.R2_MIN, C.R2_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "case3": SettingSpec("c3", str, C.DEFAULT_CASE3, choices=C.CASE3_OPTIONS),
    "n3": SettingSpec("n3", int, C.DEFAULT_N3, choices=C.N3_OPTIONS),
    "p3": SettingSpec("p3", float, C.DEFAULT_P3, choices=C.P3_OPTIONS),
    "dist4": SettingSpec("d4", str, C.DEFAULT_DIST4, choices=C.DIST4_OPTIONS),
    "n4": SettingSpec("n4", int, C.DEFAULT_N4, choices=C.N4_OPTIONS),
    "alpha4": SettingSpec("a4", float, C.DEFAULT_ALPHA4, choices=C.ALPHA4_OPTIONS),
}
PRESET_KEYS = {"m1": "m1", "T1": "T1", "k1": "k1", "theta1": "theta1", "beta1": "beta1", "rand1": "rand1", "dist2": "dist2", "m2": "m2", "target2": "target2", "R2": "R2",
               "seed": "seed_input", "case3": "case3", "n3": "n3", "p3": "p3", "dist4": "dist4", "n4": "n4", "alpha4": "alpha4"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def limit_value(spec: SettingSpec, value):
    """Wert auf die Reglergrenzen begrenzen bzw. auf eine zulässige Auswahl prüfen; None, wenn unzulässig."""
    if spec.choices is not None:
        return value if value in spec.choices else None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            value = limit_value(spec, value)
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Alle Regler setzen: was das Preset nicht nennt, geht auf den Standardwert zurück (so liest `aktives_preset` es auch); sonst bliebe der Rest des vorigen Presets stehen."""
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name].get(key, SETTING_SPECS[state_key].default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
