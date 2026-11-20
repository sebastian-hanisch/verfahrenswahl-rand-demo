"""Tabellen der README, aus den eingefrorenen Messungen und den exakten Größen erzeugt (eine Quelle für README und Test).

Aufruf (aus dem Demo-Ordner, mit dem Test-venv):
    python tools/vwr_readme_tables.py --write    # füllt die Marker {{NAME}} der Vorlage tools/vwr_readme_vorlage.md und schreibt README.md
    python tools/vwr_readme_tables.py --check    # prüft, dass jede Tabelle der README genau so aus den Daten entsteht

`tests/test_claims.py` ruft `tables()` auf und sucht jeden Block in der README: die Tabellen können nicht von den Messungen abweichen."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import vwr_buffer as B  # noqa: E402
import vwr_cvar as CV  # noqa: E402
import vwr_estimators as E  # noqa: E402
import vwr_evaluation as V  # noqa: E402
import vwr_reference as REF  # noqa: E402
import vwr_scenario as S  # noqa: E402
import vwr_table as TB  # noqa: E402
from vwr_evaluation import band, de, pct  # noqa: E402

DISTS = S.MATRIX_DISTS


def _md(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _c(d, m, q, mt, i=0):
    return REF.matrix_cell(d, m, q, i)[mt]


def matrix_table() -> str:
    head = ["Verfahren"] + [S.DIST_SHORT[d] for d in DISTS]
    rows = [[E.METHOD_LABELS[mt]] + [de(_c(d, 50, 0.01, mt)["rmse"]) for d in DISTS] for mt in E.MATRIX_METHODS]
    rows.append(["*Median Pegel-Dach/wahr, Gumbel*"] + [de(_c(d, 50, 0.01, "gumbel")["median_ratio"]) for d in DISTS])
    rows.append(["*Anteil der Läufe, in denen Gumbel unterschätzt*"] + [pct(_c(d, 50, 0.01, "gumbel")["under"]) + " %" for d in DISTS])
    rows.append(["*Median Pegel-Dach/wahr, Hill k = 5 %*"] + [de(_c(d, 50, 0.01, "hill5")["median_ratio"]) for d in DISTS])
    return _md(head, rows)


def m10_table() -> str:
    head = ["Verfahren (m = 10, k = 50)"] + [S.DIST_SHORT[d] for d in DISTS]
    ms = ("gev_ml", "gev_lmom", "gumbel", "gpd_ml50", "gpd_pwm50", "hill50")
    rows = [[E.METHOD_LABELS[mt]] + [de(_c(d, 10, 0.01, mt)["median_abs"]) for d in DISTS] for mt in ms]
    rows.append(["*ML klebt an der Grenze ξ = −1 (Läufe von 400)*"] + [str(REF.matrix_edge(d, 10)[0]) for d in DISTS])
    return _md(head, rows)


def wins_table() -> str:
    head = ["Verfahren", "Siege mit Hill (Zufalls-Seed 0 / 1 / 2)", "Siege ohne Hill"]
    rows = []
    for mt in E.MATRIX_METHODS:
        a = V.win_band(mt)[2]
        b = V.win_band(mt, with_hill=False)[2]
        if sum(a) == 0 and sum(b) == 0:
            continue
        rows.append([E.METHOD_LABELS[mt], " / ".join(map(str, a)), "-" if mt.startswith("hill") else " / ".join(map(str, b))])
    return _md(head, rows)


def kopplung_table() -> str:
    k = REF.grid()["kopplung"]
    head = ["Größe (m = 50, 100-Jahres-Pegel, R = 2 000 Läufe)"] + [S.DIST_SHORT[d] for d in DISTS]
    rows = [["RMSE GEV-ML (50 Blöcke)"] + [de(k[d]["rmse_gev"], 3) for d in DISTS], ["RMSE GPD-ML (k = 50 = n/b)"] + [de(k[d]["rmse_gpd"], 3) for d in DISTS],
            ["Verhältnis RMSE Schwelle/Block"] + [de(k[d]["ratio_rmse"]) for d in DISTS], ["Verhältnis der ln-Streuung Schwelle/Block"] + [de(k[d]["ratio_sd"]) for d in DISTS]]
    return _md(head, rows)


def xi_table() -> str:
    xi = REF.grid()["kopplung"]["exp"]["xi"]
    rows = [[lab, de(xi[key]["mean"], 3), de(xi[key]["sd"], 3)] for key, lab in (("gev_ml", "GEV-ML (50 Blöcke)"), ("gpd_ml50", "GPD-ML (k = 50)"), ("gpd_pwm50", "GPD-PWM (k = 50)"))]
    return _md(["ξ̂ bei Exponential (ξ = 0), N = 5 000, R = 2 000", "Mittel", "Standardabweichung"], rows)


def puffer_table() -> str:
    pf = REF.puffer()
    head = ["Fall, n = 2 000, p = 10⁻⁴ (n·p = 0,2)"] + [E.PUFFER_LABELS[mt] for mt in E.PUFFER_METHODS]
    rows = []
    for case in pf["cases"]:
        rows.append([f"{S.DIST_LABELS[case]}: P(q > 2p)"] + [de(pf["cells"][f"{case}|2000"]["rows"][f"{mt}|0.0001"]["gt2"], 3) for mt in E.PUFFER_METHODS])
    vb = pf["cells"]["vehikel_b|2000"]["rows"]
    rows.append(["Vehikel B: Aufschlag f90"] + [de(vb[f"{mt}|0.0001"]["f90"], 2) for mt in E.PUFFER_METHODS])
    rows.append(["Vehikel B: Gesamtpuffer (Median Puffer/wahr × f90)"] + [de(vb[f"{mt}|0.0001"]["tot"], 2) for mt in E.PUFFER_METHODS])
    return _md(head, rows)


def puffer_exact_table() -> str:
    rows = []
    for n, p in ((5000, 1e-2), (2000, 1e-2), (500, 1e-2), (5000, 1e-3), (5000, 1e-4), (2000, 1e-4), (500, 1e-4)):
        rows.append([f"{n:,}".replace(",", "."), V.p_text(p), de(n * p, 2), str(E.emp_index(n, p)), de(B.prob_q_above(n, p), 3), de(B.expected_q(n, p) / p, 2)])
    return _md(["n", "p", "n·p", "j", "P(q > 2p), exakt", "E[q]/p, exakt"], rows)


def cvar_table() -> str:
    cv = REF.cvar()["emp_r20000"]
    head = ["Verteilung (n = 600, 20 000 Läufe)"] + [f"α = {de(a, 3)}" for a in CV.ALPHAS]
    rows = [[CV.CVAR_LABELS[d]] + [de(cv[f"{d}|600"][str(a)]["median"], 3) for a in CV.ALPHAS] for d in CV.CVAR_DISTS]
    return _md(head, rows)


def cvar_gpd_table() -> str:
    g = REF.cvar()["gpd_r2000"]
    head = ["Verteilung (GPD an die oberen 5 %, 2 000 Läufe)", "n = 600, α = 0,99", "n = 600, α = 0,999", "ungültig (ξ̂ ≥ 1) von 2 000", "n = 6 000, α = 0,99", "n = 6 000, α = 0,999"]
    rows = [[CV.CVAR_LABELS[d], de(g[f"{d}|600"]["0.99"]["median"], 3), de(g[f"{d}|600"]["0.999"]["median"], 3), str(g[f"{d}|600"]["0.99"]["invalid"]), de(g[f"{d}|6000"]["0.99"]["median"], 3),
             de(g[f"{d}|6000"]["0.999"]["median"], 3)] for d in CV.CVAR_DISTS]
    return _md(head, rows)


def decision_table() -> str:
    return _md(["Situation", "Was die Messung sagt", "Verfahren: Hinweis", "Herkunft"], [[r["situation"], r["messung"], r["verfahren"], r["herkunft"]] for r in TB.rows()])


def zitate_list() -> str:
    return "\n".join(f"* **„{z['demo']}“** ({'`' + z['repo'] + '`' if z['repo'] else 'Plan'}): {z['text']}." for z in TB.ZITATE.values())


def stueck3_band_table() -> str:
    rows = [[mm] + [f"{de(V.BAND_STUECK3[(mm, tt)][0])} bis {de(V.BAND_STUECK3[(mm, tt)][1])} (Median {de(V.BAND_STUECK3[(mm, tt)][2])})" for tt in (10, 100, 1000)] for mm in (20, 50, 100, 200, 500)]
    return _md(["m (Jahre)", "T = 10", "T = 100", "T = 1000"], rows)


def tables() -> dict:
    lo, hi, _ = V.win_band("gumbel")
    return {"GUMBEL_BAND": band(lo, hi), "MATRIX": matrix_table(), "M10": m10_table(), "WINS": wins_table(), "KOPPLUNG": kopplung_table(), "XI": xi_table(), "PUFFER": puffer_table(), "PUFFER_EXAKT": puffer_exact_table(),
            "CVAR": cvar_table(), "CVAR_GPD": cvar_gpd_table(), "ENTSCHEIDUNG": decision_table(), "ZITATE": zitate_list(), "BAND3": stueck3_band_table()}


def build(vorlage: str, **extra) -> str:
    out = vorlage
    for k, v in {**tables(), **{k.upper(): str(v) for k, v in extra.items()}}.items():
        out = out.replace("{{" + k + "}}", v)
    return out


if __name__ == "__main__":
    tpl = (ROOT / "tools" / "vwr_readme_vorlage.md").read_text(encoding="utf-8")
    if "--write" in sys.argv:
        extra = {}
        for a in sys.argv:
            if a.startswith("--") and "=" in a:
                k, v = a[2:].split("=", 1)
                extra[k] = v
        (ROOT / "README.md").write_text(build(tpl, **extra), encoding="utf-8", newline="\n")
        print("README.md geschrieben")
    elif "--check" in sys.argv:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        bad = [k for k, v in tables().items() if v not in readme]
        print("Tabellen in der README:", "in Ordnung" if not bad else f"abweichend: {bad}")
        sys.exit(1 if bad else 0)
    else:
        print(__doc__)
