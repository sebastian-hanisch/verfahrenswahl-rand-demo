"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset einzigartig; Permalink begrenzt Werte."""

import vwr_constants as C
import vwr_evaluation as ev
import vwr_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 8
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_or_choices():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            assert P.limit_value(spec, value) == value, (name, key, value)


def test_default_preset_is_the_first_and_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert first == "50 Jahre, 100er-Pegel" and C.PRESETS[first] == {} and ev.Settings(**C.PRESETS[first]) == ev.Settings()


def test_all_presets_are_distinct_settings():
    settings = [ev.Settings(**p) for p in C.PRESETS.values()]
    assert len(set(settings)) == len(settings)


def test_setting_specs_cover_every_preset_key_and_the_defaults_are_allowed():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    for key, spec in P.SETTING_SPECS.items():
        assert P.limit_value(spec, spec.default) == spec.default, key


def test_limit_value_clamps_numbers_and_rejects_unknown_choices():
    assert P.limit_value(P.SETTING_SPECS["m1"], 10_000) == C.M1_MAX and P.limit_value(P.SETTING_SPECS["m1"], -3) == C.M1_MIN
    assert P.limit_value(P.SETTING_SPECS["T1"], 55) is None and P.limit_value(P.SETTING_SPECS["T1"], 300) == 300
    assert P.limit_value(P.SETTING_SPECS["rand1"], "unsinn") is None


def test_every_option_of_every_choice_is_a_valid_input_of_the_analyses():
    for name, spec in P.SETTING_SPECS.items():
        if spec.choices:
            assert len(set(spec.choices)) == len(spec.choices) and spec.default in spec.choices, name
