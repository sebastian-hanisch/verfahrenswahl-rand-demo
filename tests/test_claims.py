"""Claims: jede Zahl der README wird hier gegen die Daten bzw. exakt nachgerechnet (DEMO-PLAYBOOK §4, §11a). Die Tabellen der README entstehen aus `tools/vwr_readme_tables.py`; dieser Test
prüft, dass die README sie unverändert enthält, und rechnet die wichtigsten Zahlen auf einem zweiten Weg (roher JSON-Wert, `scipy.stats.beta`, Brüche) nach. Zitierte Zahlen anderer Demos
werden im Wortlaut in den READMEs der Nachbar-Demos gesucht (übersprungen, wenn der Ordner fehlt)."""

import json
import math
import re
from fractions import Fraction as Fr
from pathlib import Path

import pytest
from scipy import stats

import vwr_buffer as B
import vwr_cvar as CV
import vwr_estimators as E
import vwr_evaluation as V
import vwr_reference as REF
import vwr_scenario as S
import vwr_table as TB
import vwr_readme_tables as RT

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def de(x, nd):
    return f"{x:.{nd}f}".replace(".", ",")


def _raw_grid():
    return json.loads((ROOT / "vwr_grid.json").read_text(encoding="utf-8"))


def _readme_table(heading: str) -> dict:
    """Zeilen der ersten Markdown-Tabelle nach der Überschrift als {erste Zelle: [übrige Zellen]}."""
    start = README.index(heading)
    rows = {}
    for line in README[start:].splitlines()[2:]:
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows[cells[0]] = cells[1:]
    return rows


# --- exakte Zahlen ---------------------------------------------------------------------------------------------------------------------------------
def test_reihenlaenge_exakt():
    assert (Fr(99, 100) ** 50, Fr(999, 1000) ** 50) == (V.not_reached(50, 100), V.not_reached(50, 1000))
    for text, val in (("60,5 %", float(Fr(99, 100) ** 50)), ("95,1 %", float(Fr(999, 1000) ** 50)), ("36,6 %", float(Fr(99, 100) ** 100)), ("63,4 %", 1 - float(Fr(99, 100) ** 100)),
                      ("39,5 %", 1 - float(Fr(99, 100) ** 50)), ("4,9 %", 1 - float(Fr(999, 1000) ** 50))):
        assert text == f"{100 * val:.1f} %".replace(".", ","), text
        assert text in README
    assert "63,2 %" in README and abs(1 - math.exp(-1) - 0.632) < 1e-3


def test_puffer_exakt_beta_tabelle_gegen_scipy():
    assert "{{" not in README                                                     # alle Marker der Vorlage sind ersetzt
    for n, p in ((5000, 1e-2), (2000, 1e-2), (500, 1e-2), (5000, 1e-3), (5000, 1e-4), (2000, 1e-4), (500, 1e-4)):
        j = E.emp_index(n, p)
        ex = float(stats.beta.sf(2 * p, n - j + 1, j))
        zeile = f"| {n:,} | ".replace(",", ".")
        line = next(l for l in README.splitlines() if l.startswith(zeile) and f"| {de(n * p, 2)} | {j} |" in l)
        assert f"| {de(ex, 3)} |" in line, (n, p, ex)
    assert "0,670" in README and "5,00-fache" in README


def test_puffer_exakt_zahlen_der_kurzen_reihe():
    assert Fr(5, 6) ** 6 == Fr(15625, 46656) and f"{float(Fr(5, 6) ** 6):.3f}".replace(".", ",") == "0,335" and "(5/6)^6 = 0,335" in README
    assert abs(B.expected_q(2000, 1e-4) / 1e-4 - 4.9975) < 1e-4 and abs(B.prob_q_above(2000, 1e-4) - (1 - 2e-4) ** 2000) < 1e-12


def test_aufschlag_120_5_prozent_ist_exakt_und_die_simulation_streut_darum():
    f90 = B.f90_of_maximum("vehikel_b", 2000, 1e-4)
    assert f"{f90:.3f}".replace(".", ",") == "2,205" and f"{100 * (f90 - 1):.1f}".replace(".", ",") == "120,5"
    assert "f90 = 2,205" in README and "Aufschlag 120,5 %" in README and "**120,5 %**" in README
    sim = REF.puffer()["cells"]["vehikel_b|2000"]["rows"]["emp|0.0001"]["f90"]
    assert 2.0 < sim < 2.4 and f"{sim:.2f}".replace(".", ",") == "2,16"


def test_cvar_exakt_54_86_prozent():
    assert f"{100 * 0.999 ** 600:.2f}".replace(".", ",") == "54,86" and "54,86 %" in README and (1 - 0.999) * 600 < 1


def test_hinweise_schwellen_der_tabelle():
    assert (V.years_for_80(0.01), V.years_for_80(0.02), V.years_for_80(0.04)) == (99, 62, 39) and "99 / 62 / 39" in README
    assert "36,6 %" in README and V.hinweise(100, 100, k=250) == []


# --- eingefrorene Messungen --------------------------------------------------------------------------------------------------------------------
def test_readme_enthaelt_alle_tabellen_unveraendert():
    for name, block in RT.tables().items():
        assert block in README, name


def test_matrix_tabelle_gegen_rohe_json_werte():
    g = _raw_grid()
    cells = g["matrix"][next(iter(g["matrix"]))]["cells"]
    rows = _readme_table("RMSE von ln(Pegel-Dach/wahr); alle Verfahren sehen dieselben Reihen:")
    for mt in E.MATRIX_METHODS:
        got = rows[E.METHOD_LABELS[mt]]
        want = [de(cells[f"{d}|50"]["targets"]["0.01"][mt]["rmse"], 2) for d in S.MATRIX_DISTS]
        assert got == want, mt
    for d_i, d in enumerate(S.MATRIX_DISTS):
        c = cells[f"{d}|50"]["targets"]["0.01"]
        assert rows["*Median Pegel-Dach/wahr, Gumbel*"][d_i] == de(c["gumbel"]["median_ratio"], 2)
        assert rows["*Anteil der Läufe, in denen Gumbel unterschätzt*"][d_i] == de(100 * c["gumbel"]["under"], 1) + " %"
    assert rows["*Anteil der Läufe, in denen Gumbel unterschätzt*"] == ["54,5 %", "98,8 %", "11,2 %", "99,5 %"]


def test_matrix_zaehlung_unabhaengig_nachgezaehlt():
    g = _raw_grid()
    per_seed = []
    for mat in g["matrix"].values():
        wins = {}
        for d in S.MATRIX_DISTS:
            for m in (10, 20, 50, 100):
                for q in ("0.01", "0.001"):
                    cell = mat["cells"][f"{d}|{m}"]["targets"][q]
                    best = min((st["rmse"], mt) for mt, st in cell.items() if st["n"] >= 360)
                    wins[best[1]] = wins.get(best[1], 0) + 1
        per_seed.append(wins)
    assert [w.get("gumbel", 0) for w in per_seed] == [12, 13, 14] and "12 bis 14" in README and "12 / 13 / 14" in README
    assert [w.get("hill50", 0) + w.get("hill5", 0) for w in per_seed] == [8, 8, 8] and [w.get("gpd_ml5", 0) for w in per_seed] == [9, 9, 9]
    assert sum(w["gumbel"] for w in per_seed) / 3 == 13
    assert sum(1 for mat in g["matrix"].values() for d in S.MATRIX_DISTS for m in (10, 20, 50, 100) for q in ("0.01", "0.001")) == 96


def test_kurze_reihe_tabelle_gegen_rohe_json_werte():
    g = _raw_grid()
    cells = g["matrix"][next(iter(g["matrix"]))]["cells"]
    rows = _readme_table("(unempfindlich gegen Ausreißer):")
    for mt in ("gev_ml", "gev_lmom", "gumbel", "gpd_ml50", "gpd_pwm50", "hill50"):
        assert rows[E.METHOD_LABELS[mt]] == [de(cells[f"{d}|10"]["targets"]["0.01"][mt]["median_abs"], 2) for d in S.MATRIX_DISTS], mt
    assert rows["*ML klebt an der Grenze ξ = −1 (Läufe von 400)*"] == [str(cells[f"{d}|10"]["gev_edge"]) for d in S.MATRIX_DISTS]
    assert "1,8 bis 12,5 %" in README and min(cells[f"{d}|10"]["gev_edge"] for d in S.MATRIX_DISTS) / 400 == 0.0175 and max(cells[f"{d}|10"]["gev_edge"] for d in S.MATRIX_DISTS) / 400 == 0.125


def test_kopplung_zelle_und_xi_streuung():
    k = _raw_grid()["kopplung"]
    assert [de(k[d]["ratio_rmse"], 2) for d in S.MATRIX_DISTS] == ["1,04", "1,09", "1,03", "1,06"] and "**R = 2 000 1,04 / 1,09 / 1,03 / 1,06**" in README
    assert [de(k[d]["ratio_sd"], 2) for d in S.MATRIX_DISTS] == ["1,01", "1,08", "1,01", "1,06"]
    xi = k["exp"]["xi"]
    assert (de(xi["gev_ml"]["mean"], 3), de(xi["gev_ml"]["sd"], 3), de(xi["gpd_ml50"]["mean"], 3), de(xi["gpd_ml50"]["sd"], 3)) == ("-0,007", "0,121", "-0,059", "0,171")


def test_prosa_zahlen_der_matrix():
    c = lambda d, m, q, mt: REF.matrix_cell(d, m, q, 0)[mt]
    assert [de(c(d, 50, 0.01, "hill5")["median_ratio"], 2) for d in ("exp", "normal", "lognormal")] == ["1,66", "1,66", "1,69"]
    assert [de(c(d, 50, 0.001, "hill5")["median_ratio"], 2) for d in ("exp", "normal", "lognormal")] == ["2,43", "2,38", "2,56"]
    assert [de(c("pareto", m, 0.01, "gumbel")["median_ratio"], 2) for m in (10, 20, 50, 100)] == ["0,39", "0,43", "0,45", "0,47"]
    assert [de(c("pareto", m, 0.001, "gumbel")["median_ratio"], 2) for m in (10, 20, 50, 100)] == ["0,16", "0,18", "0,20", "0,20"]
    assert [de(100 * (1 - c("normal", m, 0.01, "gumbel")["under"]), 1) for m in (20, 50, 100)] == ["75,5", "88,8", "97,8"]
    assert de(c("pareto", 50, 0.01, "gpd_ml5")["rmse"], 2) == "0,35" and de(c("pareto", 50, 0.01, "gev_ml")["rmse"], 2) == "0,48"
    for text in ("1,66-, 1,66- und 1,69-Fachen", "2,43-, 2,38- und 2,56-Fachen", "0,39- bis 0,47-Fachen", "0,16- bis 0,20-Fachen", "75,5 %", "88,8 %", "97,8 %", "Pareto-Verteilung 0,35 gegen 0,48"):
        assert text in README, text


def test_gumbel_feste_zellen_und_unterschaetzung():
    assert len(V.fixed_gumbel_cells()) == 12 and len(V.gumbel_underestimates(0.9)) == 16
    assert "12 feste Zellen" in README and "16 von 32 Zellen" in README


def test_puffer_tabelle_und_prosa_gegen_rohe_json_werte():
    raw = json.loads((ROOT / "vwr_puffer.json").read_text(encoding="utf-8"))
    rows = raw["cells"]["vehikel_b|2000"]["rows"]
    got = {mt: rows[f"{mt}|0.0001"] for mt in E.PUFFER_METHODS}
    assert [de(got[mt]["gt2"], 3) for mt in ("emp", "gpd", "hill", "gev")] == ["0,672", "0,220", "0,004", "0,318"]
    assert [de(got[mt]["tot"], 2) for mt in ("emp", "gpd", "hill", "gev")] == ["1,48", "1,68", "1,41", "1,80"]
    assert "Vehikel B 0,220 / 0,004 / 0,318 gegen 0,672" in README and "(Vehikel B: GPD 1,68 gegen 1,48 empirisch)" in README
    n_p_5 = [raw["cells"][f"{c}|{n}"]["rows"][f"{mt}|{p:g}"]["gt2"] for c in raw["cases"] for n in raw["ns"] for p in raw["ps"] if n * p >= 5 - 1e-9 for mt in ("emp", "gpd", "hill")]
    assert len(n_p_5) == 48 and max(n_p_5) <= 0.034 + 1e-12
    emp, gpd, hill = (max(raw["cells"][f"{c}|{n}"]["rows"][f"{mt}|{p:g}"]["gt2"] for c in raw["cases"] for n in raw["ns"] for p in raw["ps"] if n * p >= 5 - 1e-9) for mt in ("emp", "gpd", "hill"))
    assert (de(emp, 3), de(gpd, 3), de(hill, 3)) == ("0,033", "0,032", "0,034") and "0,033 empirisch, 0,032 GPD, 0,034 Hill-Weissman" in README
    gev = max(raw["cells"][f"{c}|{n}"]["rows"][f"gev|{p:g}"]["gt2"] for c in raw["cases"] for n in raw["ns"] for p in raw["ps"] if n * p >= 5 - 1e-9)
    assert de(gev, 3) == "0,162" and "bis 0,162" in README
    nf = raw["cells"]["vehikel_b|2000"]["rows"]["normal|0.0001"]
    assert nf["gt2"] == 1.0 and de(nf["medq"], 1) == "107,6" and "107,6-Fache" in README
    ng = raw["cells"]["normal_100_20|2000"]["rows"]
    assert (de(ng["gpd|0.0001"]["gt2"], 3), de(ng["normal|0.0001"]["gt2"], 3)) == ("0,451", "0,003") and "(0,451 gegen 0,003)" in README
    ex = raw["cells"]["exp|2000"]["rows"]
    assert (de(ex["lognormal|0.001"]["medr"], 2), de(ex["lognormal|0.0001"]["medr"], 2)) == ("4,26", "7,15")
    assert de(raw["cells"]["normal_100_20|2000"]["rows"]["normal|0.01"]["f90"], 2) == "1,01"


def test_kurze_blockreihen_gegen_rohe_json_werte():
    sh = REF.puffer()["short"]
    assert de(sh["0.0|6"]["gev_ml|0.083333"]["gt2"], 3) == "0,346" and de(sh["0.2|6"]["gev_ml|0.083333"]["gt2"], 3) == "0,337" and "(0,346 / 0,337)" in README
    assert (de(sh["0.0|6"]["max|0.083333"]["aufschlag90"], 2), de(sh["0.2|6"]["max|0.083333"]["aufschlag90"], 2)) == ("1,44", "2,07") and "1,44 σ bei ξ = 0, 2,07 σ bei ξ = 0,2" in README
    assert (de(sh["0.2|24"]["gumbel|0.008333"]["gt2"], 2), de(sh["0.2|24"]["max|0.008333"]["gt2"], 2)) == ("0,78", "0,65") and "0,78 gegen 0,65" in README
    assert (de(sh["0.0|6"]["max|0.083333"]["gt2"], 3), de(sh["0.2|6"]["max|0.083333"]["gt2"], 3)) == ("0,305", "0,352") and "0,305 / 0,352" in README


def test_cvar_tabellen_gegen_rohe_json_werte_und_exakt():
    raw = json.loads((ROOT / "vwr_cvar.json").read_text(encoding="utf-8"))
    rows = _readme_table("{{CVAR}}") if "{{CVAR}}" in README else _readme_table("### D. CVaR aus Szenarien (Median des Verhältnisses empirisch/exakt)")
    for d in CV.CVAR_DISTS:
        assert rows[CV.CVAR_LABELS[d]] == [de(raw["emp_r20000"][f"{d}|600"][str(a)]["median"], 3) for a in CV.ALPHAS], d
    assert [de(raw["emp_r20000"]["pareto|600"][str(a)]["median"], 3) for a in CV.ALPHAS] == ["0,961", "0,940", "0,847", "0,467"]
    assert (de(raw["emp_r20000"]["normal|600"]["0.9"]["median"], 3), de(raw["emp_r20000"]["pareto|600"]["0.9"]["median"], 3)) == ("0,997", "0,961") and "(0,961 bis 0,997)" in README
    assert raw["gpd_r2000"]["pareto|600"]["0.99"]["invalid"] == 54 and "ungültig (ξ̂ ≥ 1) bei der Pareto-Verteilung 58 → 54 von 2 000 Läufen" in README
    for d in CV.CVAR_DISTS:
        for a in CV.ALPHAS:
            assert raw["exact"][f"{d}|{a}"] == pytest.approx(CV.es_exact(d, a), rel=1e-12)


def test_entscheidungstabelle_enthaelt_15_zeilen_und_jede_steht_in_der_readme():
    ids = [r["id"] for r in TB.rows()]
    assert len(ids) == 15 and len(set(ids)) == 15
    for r in TB.rows():
        for key in ("situation", "messung", "verfahren", "herkunft"):
            assert r[key] and r[key] in README, (r["id"], key)
        assert re.search(r"\d", r["messung"]) or r["id"] == "r_mehrdim"          # jede Zeile außer „nicht gerechnet“ nennt eine Zahl


def test_entscheidungstabelle_eigene_zahlen_gegen_die_daten():
    row = {r["id"]: r["messung"] for r in TB.rows()}
    assert "60,5 % der 50-Jahres-Reihen" in row["r_reihe"] and "95,1 %" in row["r_reihe"] and "36,6 %" in row["r_reihe"]
    assert "RMSE von ln 0,06 (Gumbel, ξ = 0 fest) gegen 0,13 (GEV-ML)" in row["r_gumbel_fest"] and "in allen 12 festen Zellen" in row["r_gumbel_fest"]
    assert "im Median bei 0,45" in row["r_schwer"] and "98,8 %" in row["r_schwer"] and "16 von 32 Zellen" in row["r_schwer"]
    assert "1,04 / 1,09 / 1,03 / 1,06" in row["r_block_schwelle"]
    assert "0,37 bei n·p = 0,5, 0,67 bei 0,2, 0,90 bei 0,05" in row["r_puffer"] and "0,335" in row["r_puffer"]
    assert "0,961 (Pareto-Verteilung) bis 0,997 (Normal)" in row["r_cvar"]


def _sibling_readme(repo: str) -> str:
    p = ROOT.parent / repo / "README.md"
    if not p.exists():
        pytest.skip(f"{repo} liegt nicht im Portfolio-Ordner")
    return p.read_text(encoding="utf-8")


@pytest.mark.parametrize("key", list(TB.ZITATE))
def test_zitate_stehen_im_wortlaut_der_nachbar_demos_und_in_der_readme(key):
    z = TB.ZITATE[key]
    assert z["text"] in README and f"**„{z['demo']}“**" in README
    text = _sibling_readme(z["repo"])
    for snippet in z["snippets"]:
        assert snippet in text, (key, snippet)


def test_zitate_stueck_3_band_tabelle_gegen_die_nachbar_readme():
    text = _sibling_readme("wiederkehrwert-demo")
    for (m, T), (lo, hi, med) in V.BAND_STUECK3.items():
        assert f"{de(lo, 2)} bis {de(hi, 2)} (Median {de(med, 2)})" in text, (m, T)


# --- Zweite Prüfung (2026-11-20): Zählungen über die drei Zufalls-Seeds, Puffer unter n·p < 1, CVaR bei anderem n, „Spitzen-Cluster“ nach der Korrektur dort -------------
def _seeds():
    return range(len(REF.seed_keys()))


def _puffer_row(raw, case, n, p, mt):
    return raw["cells"][f"{case}|{n}"]["rows"][f"{mt}|{p:g}"]


def test_gumbel_unterschaetzung_je_zufalls_seed_und_lognormal_zelle():
    assert [len(V.gumbel_underestimates(0.9, i)) for i in _seeds()] == [16, 16, 15]
    logn = [REF.matrix_cell("lognormal", 10, 0.01, i)["gumbel"]["under"] for i in _seeds()]
    assert (de(100 * min(logn), 1), de(100 * max(logn), 1)) == ("89,5", "93,8")
    leicht = max(REF.matrix_cell(d, m, q, i)["gumbel"]["under"] for i in _seeds() for d in ("exp", "normal") for m in (10, 20, 50, 100) for q in (0.01, 0.001))
    assert de(100 * leicht, 1) == "64,5"
    for text in ("15 bis 16 Zellen", "89,5 bis 93,8 %", "höchstens 64,5 %"):
        assert text in README, text


def test_ml_an_der_grenze_und_gumbel_normal_ueber_drei_zufalls_seeds():
    shares = [100 * a / b for i in _seeds() for d in S.MATRIX_DISTS for a, b in [REF.matrix_edge(d, 10, i)]]
    assert (de(min(shares), 2), de(max(shares), 1)) == ("0,75", "12,5") and "0,75 bis 12,5 %" in README
    high = [100 * (1 - REF.matrix_cell("normal", 50, 0.01, i)["gumbel"]["under"]) for i in _seeds()]
    assert (de(min(high), 1), de(max(high), 1)) == ("88,8", "94,0") and "88,8 bis 94,0 %" in README


def test_puffer_unter_einer_beobachtung_kein_rand_verfahren_haelt_in_allen_vier_faellen():
    raw = json.loads((ROOT / "vwr_puffer.json").read_text(encoding="utf-8"))
    cases = ("vehikel_b", "pareto", "exp", "normal_100_20")
    hit = {mt: {c: _puffer_row(raw, c, 2000, 1e-4, mt)["hit"] for c in cases} for mt in ("gpd", "gev", "hill")}
    both = list(hit["gpd"].values()) + list(hit["gev"].values())
    assert (round(100 * min(both)), round(100 * max(both))) == (36, 60)                         # GPD und GEV: 36 bis 60 %
    assert [de(hit["hill"][c], 3) for c in cases] == ["0,956", "0,501", "1,000", "0,999"]       # Hill: Vehikel B, Pareto, Exponential, Normal
    assert [de(_puffer_row(raw, c, 2000, 1e-4, "hill")["medr"], 2) for c in ("vehikel_b", "exp", "normal_100_20")] == ["1,56", "1,65", "1,11"]
    for text in ("36 bis 60 %", "95,6 / 100 / 99,9 %", "1,56 / 1,65 / 1,11", "nur in 50,1 %", "1,1- bis 1,7-fachen Puffer"):
        assert text in README, text
    for n in (2000, 500):                                                                        # n·p = 0,2 und 0,05: die Fehlgriffe streuen weniger
        for c in cases:
            emp = _puffer_row(raw, c, n, 1e-4, "emp")["gt2"]
            assert all(_puffer_row(raw, c, n, 1e-4, mt)["gt2"] < emp for mt in ("gpd", "hill", "gev")), (c, n)


def test_puffer_gesamtpuffer_je_np_und_ab_np_5():
    raw = json.loads((ROOT / "vwr_puffer.json").read_text(encoding="utf-8"))
    cases = ("vehikel_b", "pareto", "exp", "normal_100_20")
    tot = lambda c, n, p, mt: _puffer_row(raw, c, n, p, mt)["tot"]
    assert all(tot(c, 2000, 1e-4, "gpd") >= tot(c, 2000, 1e-4, "emp") for c in cases)           # n·p = 0,2: GPD nie kleiner
    assert all(tot(c, 5000, 1e-4, "gpd") < tot(c, 5000, 1e-4, "emp") for c in cases)            # n·p = 0,5: in allen vier Fällen kleiner
    assert [(de(tot(c, 5000, 1e-4, "gpd"), 2), de(tot(c, 5000, 1e-4, "emp"), 2)) for c in ("vehikel_b", "pareto")] == [("1,38", "1,48"), ("1,53", "1,87")]
    assert [(de(tot(c, 2000, 1e-4, "hill"), 2), de(tot(c, 2000, 1e-4, "emp"), 2)) for c in ("vehikel_b", "pareto")] == [("1,41", "1,48"), ("1,50", "1,87")]
    ge5 = [(c, n, p) for c in cases for n in raw["ns"] for p in raw["ps"] if n * p >= 5 - 1e-9]
    assert len(ge5) == 16 and all(tot(c, n, p, mt) <= tot(c, n, p, "emp") + 1e-9 for c, n, p in ge5 for mt in ("gpd", "hill"))
    gev5 = [_puffer_row(raw, c, n, p, "gev")["gt2"] for c, n, p in ge5 if n == 500]                # 5 Blöcke
    gev20_50 = [_puffer_row(raw, c, n, p, "gev")["gt2"] for c, n, p in ge5 if n != 500]            # 20 und 50 Blöcke
    emp20_50 = [_puffer_row(raw, c, n, p, "emp")["gt2"] for c, n, p in ge5 if n != 500]
    assert de(max(gev5), 3) == "0,162" and (de(max(gev20_50), 3), de(max(emp20_50), 3)) == ("0,038", "0,033")
    for text in ("höchstens 0,038 gegen 0,033", "(Vehikel B 1,38 gegen 1,48, Pareto-Verteilung 1,53 gegen 1,87)", "(1,41 gegen 1,48, 1,50 gegen 1,87)"):
        assert text in README, text


def test_normal_formel_bei_schwerem_rand_und_f90_des_maximums():
    raw = json.loads((ROOT / "vwr_puffer.json").read_text(encoding="utf-8"))
    cells = [(c, n, p) for c in ("vehikel_b", "pareto") for n in raw["ns"] for p in raw["ps"]]
    assert len(cells) == 9 * 2 and all(_puffer_row(raw, c, n, p, "normal")["gt2"] > _puffer_row(raw, c, n, p, "emp")["gt2"] for c, n, p in cells)
    f90 = [de(B.f90_of_maximum(c, 2000, 1e-4), 3) for c in ("normal_100_20", "exp", "vehikel_b", "pareto")]
    assert f90 == ["1,083", "1,361", "2,205", "3,392"] and "1,083 bei Normal, 1,361 bei Exponential, 2,205 bei Vehikel B, 3,392 bei der Pareto-Verteilung" in README
    row = {r["id"]: r for r in TB.rows()}["r_puffer"]
    assert "die Normal-Formel nicht bei schwerem Rand" in row["verfahren"] and "nie die Normal-Formel" not in row["verfahren"]


def test_cvar_abstand_bei_n_60_600_und_6000():
    d = REF.cvar()
    med = lambda n, dist, a: d["emp_r20000" if n == 600 else "emp_r2000"][f"{dist}|{n}"][str(a)]["median"]
    for n in (60, 600):
        for dist in CV.CVAR_DISTS:
            m = [med(n, dist, a) for a in CV.ALPHAS]
            assert all(x < 1 for x in m) and m == sorted(m, reverse=True), (n, dist)           # unter dem echten, wächst mit dem Niveau
    gap = {dist: 1 - med(6000, dist, 0.95) for dist in CV.CVAR_DISTS}
    assert max(1 - med(6000, dist, a) for dist in CV.CVAR_DISTS for a in (0.9, 0.95)) <= 0.02 and de(100 * gap["pareto"], 1) == "1,5"
    assert "bei n = 60 und 600 im Median unter dem echten" in README and "höchstens 2 % (2 000 Läufe; Pareto-Verteilung 1,5 %)" in README


def test_stueck7_zahlen_nach_der_korrektur_dort_und_keine_alten_zahlen_im_zitat():
    z = TB.ZITATE["z7_abh"]["text"]
    for neu in ("89,0 / 62,4 / 32,6 %", "89,0 / 75,6 / 60,5 %", "35,4 / 4,9 %", "59,1 / 22,9 %", "61,7 / 25,6 %", "10,2 bis 74,2 %", "88,9 / 86,1 / 84,3 %"):
        assert neu in z, neu
    for alt in ("95,3 / 78,3 / 49,1", "95,3 / 89,5 / 82,0", "73,9", "17,9", "94,8", "97,4", "70,2", "86,9"):
        assert alt not in z and alt not in {r["id"]: r for r in TB.rows()}["r_abhaengig"]["verfahren"], alt
    row = {r["id"]: r for r in TB.rows()}["r_abhaengig"]
    assert z in row["messung"] and "Warnung bei θ·k < 50, eigene Festlegung" in row["verfahren"] and "Schwelle mit θ-Korrektur" not in row["verfahren"]
    assert "Vorzeichenfehler in der Kovarianz" in README and "nicht mehr gedeckt" in README
    assert "73,9 / 94,8 / 97,4 und 17,9 / 70,2 / 86,9 % → **35,4 / 59,1 / 61,7 und 4,9 / 22,9 / 25,6 %**" in README
