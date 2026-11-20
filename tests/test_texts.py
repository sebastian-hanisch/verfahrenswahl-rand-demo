"""Wortwahl der sichtbaren Texte (App, Tabelle, Presets, README): kein „trägt/tragen“, kein „Saatwert“ und „Schwanz“, „Pareto“ nur als Pareto-Verteilung, „Weibull“ nur als Weibull-Typ,
Extremwertindex ξ und Extremalindex θ getrennt benannt, „Zufalls-Seed“."""

import re
from pathlib import Path

import pytest

import vwr_constants as C
import vwr_table as TB
import vwr_texts as T

ROOT = Path(__file__).resolve().parent.parent


def _texts() -> dict:
    out = {"vwr_texts.py": " ".join([T.INTRO, T.HOW_IT_WORKS, T.CAPTION_EXP1, T.CAPTION_EXP2, T.CAPTION_EXP3, T.CAPTION_EXP4, T.ASSUMPTIONS, T.SYMBOLS, T.FORMULAS_OUTRO, *T.FORMULA_BLOCKS]),
           "Presets": " ".join(C.PRESET_HELP.values()), "Tabelle": " ".join(" ".join(r[k] for k in ("situation", "messung", "verfahren", "herkunft")) for r in TB.rows()),
           "app.py": (ROOT / "app.py").read_text(encoding="utf-8")}
    if (ROOT / "README.md").exists():
        out["README.md"] = (ROOT / "README.md").read_text(encoding="utf-8")
    return out


@pytest.mark.parametrize("quelle", ["vwr_texts.py", "Presets", "Tabelle", "app.py", "README.md"])
def test_verbotene_woerter_kommen_nicht_vor(quelle):
    text = _texts().get(quelle)
    if text is None:
        pytest.skip("README noch nicht da")
    assert not re.search(r"\btr(ä|ae)g(t|en)\b", text, re.I), "trägt/tragen: reicht/gilt/zeigt"
    assert "Saatwert" not in text and "Schwanz" not in text and "Seed-Wert" not in text
    assert not re.search(r"Weibull(?!-Typ)", text), "Weibull nur als Weibull-Typ"
    assert "Zufallsseed" not in text and "Random Seed" not in text


def test_pareto_nie_allein():
    for quelle, text in _texts().items():
        text = text.replace("`", "").replace("potenzgesetz", "")
        bad = [m.group(0) for m in re.finditer(r"Pareto.{0,14}", text) if not re.match(r"Pareto(-Verteilung|-artig|-Mischung)", m.group(0))]
        assert not bad, (quelle, bad[:3])


def test_extremwertindex_xi_und_extremalindex_theta_werden_getrennt_benannt():
    t = _texts()["vwr_texts.py"]
    assert "Extremalindex" in t and "Extremwertindex" in t and "**nicht** mit $\\xi$ verwandt" in t
