"""Die Entscheidungstabelle: je Zeile eine Situation, was die Messung sagt (mit Zähleinheit und Bezugsmenge), ein Hinweis zum Verfahren und die Herkunft der Zahl.

Eigene Zeilen (Matrix, CVaR, Puffer, Reihenlänge) holen ihre Zahlen aus den eingefrorenen Dateien bzw. rechnen sie exakt; Zeilen anderer Demos zitieren die Demo, die die Zahl führt (`ZITATE`),
und leiten sie nicht neu her. Die Tabelle sagt, wann man ein Verfahren **nicht** nehmen sollte, nicht, welches das richtige ist. Alles gilt für ausgedachte Verteilungen, iid-Beobachtungen (außer in den
Zeilen zu Abhängigkeit und Trend), feste eigene Wahl der k (50 und 5 % der Beobachtungen) und die Zähleinheiten Läufe, Zellen, Jahre (Blöcke zu 100 Beobachtungen).
Über den Nutzen in Logistik, Lager oder Hafen sagt keine Zeile etwas."""

import vwr_buffer as B
import vwr_evaluation as V
import vwr_reference as REF
import vwr_scenario as S
from vwr_evaluation import band, de, pct

# Zitierte Zahlen: Demo, Repo, Wortlaut (so steht er in der Nachbar-Demo bzw. im Plan), Textstelle zum Wiederfinden (`tests/test_copies.py` sucht sie, wenn der Nachbarordner da ist).
ZITATE = {
    "z3_band": {"demo": "Wiederkehrwert", "repo": "wiederkehrwert-demo", "text": "GEV(μ = 100, σ = 20, ξ = 0,2), m = 50: der 100er-Pegel zwischen dem 0,73- und 1,49-Fachen des wahren, der 1000er-Pegel zwischen 0,56 und 2,21, "
              "in 6,9 % der Läufe beim 1000er mehr als doppelt so hoch (2 000 Läufe)", "snippets": ("0,73- und 1,49-Fachen", "zwischen 0,56 und 2,21", "6,9 %")},
    "z3_delta": {"demo": "Wiederkehrwert", "repo": "wiederkehrwert-demo", "text": "Delta-Intervall 76,0 bis 91,9 % Überdeckung (13 Zellen, nominell 95 %)", "snippets": ("76,0 bis 91,9 %",)},
    "z3_profil": {"demo": "Wiederkehrwert", "repo": "wiederkehrwert-demo", "text": "Profil-Likelihood 92,4 bis 95,9 % (13 Zellen)", "snippets": ("92,4 % und 95,9 %",)},
    "z3_boot": {"demo": "Wiederkehrwert", "repo": "wiederkehrwert-demo", "text": "Perzentil-Bootstrap (B = 199) 82,2 bis 91,2 % in 6 Zellen, unter dem Profil in 6 von 6", "snippets": ("82,2 % bis 91,2 %", "unter dem Profil in 6 von 6")},
    "z3_haupt": {"demo": "Wiederkehrwert", "repo": "wiederkehrwert-demo", "text": "Zelle ξ = 0,2, m = 50, T = 100: Delta 89,1 %, Profil 95,1 %, Bootstrap 91,2 %", "snippets": ("89,1 %", "95,1 %", "91,2 %")},
    "z2_lmom": {"demo": "GEV an Blockmaxima", "repo": "gev-blockmaxima-demo", "text": "L-Momente mit unmöglichem Endpunkt (ξ = −0,6 und −0,9, m = 20, 50, 100): 8 bis 28 % mit Klammer ξ̂ ≥ −0,999, 14 bis 35 % ohne; ML klebt bei ξ = −0,6, m = 20 in 20 % der Läufe an der Grenze",
                "snippets": ("8 % bis 28 % mit Klammer", "ξ = −0,6 bei m = 10 / 20 / 50: 49 % / 20 % / 1 %")},
    "z4_block": {"demo": "Überschreitungen und die GPD", "repo": "gpd-ueberschreitungen-demo", "text": "Verhältnis Schwelle/Block der ln-Streuung bei gleicher Zahl benutzter Werte (k = n/b): b = 20 1,09 bis 1,38 (Schwelle streut mehr in 16 von 16 Zellen), "
               "b = 100 0,81 bis 1,23 (12 von 16); bei k = 2n/b 0,94 bis 1,07 (b = 20, 8 von 16) bzw. 0,70 bis 1,01 (b = 100, 3 von 16); Zähleinheit Läufe, R = 600", "snippets": ("1,09 bis 1,38 (mehr in 16 von 16 Zellen)", "0,81 bis 1,23 (12 von 16)", "0,94 bis 1,07 (8 von 16)", "0,70 bis 1,01 (3)")},
    "z4_klein": {"demo": "Überschreitungen und die GPD", "repo": "gpd-ueberschreitungen-demo", "text": "ML-Streuung sd·√k bei ξ = 0: 1,69 / 1,43 / 1,20 bei k = 10 / 20 / 50 (Formel 1); die ML klebt bei k = 10 in 35,5 % der Läufe am Rand ξ = −1; PWM streut dort weniger", "snippets": ("1,69 / 1,43 / 1,20", "35,5 %")},
    "z5_regeln": {"demo": "Schwellenwahl", "repo": "schwellenwahl-demo", "text": "Regel S wählt im Spleiß-Fall (Anteil 10 %, c = 2) in 50,0 % der Läufe eine Schwelle im GPD-Bereich, Regel M in 0,2 %; M und G wählen beim Vehikel B in 88 bzw. 82 % der Läufe die niedrigste Schwelle; "
                  "keine Regel schlägt ein festes k = 71 bis 250 (RMSE 0,65 bis 0,85 gegen 0,54 bis 0,59); sieben Regeln auf derselben Reihe: Median des Pegelverhältnisses 2,76, über Faktor 2 in 72,3 % der 300 Läufe",
                  "snippets": ("71,0 / 50,0 / 24,5 %", "M (c = 2) und G wählen in 88 % bzw. 82 %", "0,65 bis 0,85", "Median 2,76", "über Faktor 2 in 72,3 %")},
    "z6_hill": {"demo": "Pegel statt Index", "repo": "hill-pegel-demo", "text": "Hill-Weissman bei verschobener Pareto-Verteilung, k = 100: in 19.888 von 20.000 Läufen unter dem wahren Pegel (99,44 %), Median 0,50; bei k = 250 Median 0,35; bei Lognormal und Exponential wächst der Fehler mit k "
               "(Pegelfaktor 1,50 → 3,17 bzw. 1,46 → 3,30 von k = 50 auf 250)", "snippets": ("19.888 von 20.000", "0,61 → 0,50 → 0,35", "Lognormal 1,50 → 1,90 → 3,17")},
    "z6_vb": {"demo": "Pegel statt Index", "repo": "hill-pegel-demo", "text": "Vehikel B, Tagesmaximum einmal in 1 000 Tagen, 50 Tage: Median Pegel-Dach/wahr der GEV frei 1,146 (300 Läufe) bzw. 1,265 (3 000 Läufe), der GPD (k = 100) 0,998 bzw. 1,116, "
              "Gumbel 0,471 (in allen 300 Läufen zu niedrig), Hill-Weissman k = 250 2,404 (0 % unterschätzen); die Mediane streuen mit R stark und sind nur als Band zu lesen", "snippets": ("GEV frei 1,146", "POT mit k = 100 0,998", "Gumbel 0,471", "Hill-Weissman k = 250 2,404")},
    "z1_normal": {"demo": "Das Größte von n", "repo": "maximum-von-n-demo", "text": "Normal, n = 365: Kolmogorov-Abstand der exakten Max-Verteilung zum Gumbel-Gesetz 0,0213 mit Normierung B und 0,0524 mit der Standardnormierung A; "
                  "100-Jahres-Pegel (0,01 je Jahr): exakt 4,033, Gumbel 4,272 (B) bzw. 4,147 (A), also 0,24 σ zu hoch (B)", "snippets": ("0,0213", "0,0524", "4,033", "4,272")},
    "z7_abh": {"demo": "Spitzen-Cluster", "repo": "spitzen-cluster-demo", "text": "max-AR(1), 50 Blöcke zu 50, k = 50, R = 1 500 Läufe, φ = 0 / 0,5 / 0,8, Überdeckung des nominellen 95-%-Intervalls für den 100-Block-Wert: Block-GEV auf der Pegelskala 88,9 / 86,1 / 84,3 % (ln-Skala 93,1 / 93,8 / 91,6 %), "
               "Schwelle (θ-bereinigtes Niveau) ohne θ 89,0 / 62,4 / 32,6 %, mit exaktem θ 89,0 / 75,6 / 60,5 % (bei φ = 0,8 sind das 10 Cluster); am Vehikel A (M/M/1-Wartezeit, Last 50 / 80 %, 3 000 Kunden, 5 % über der Schwelle, p = 10⁻³ je Kunde, R = 2 000) deckt das Intervall "
               "„unabhängig“ in 35,4 / 4,9 %, mit θ̂ in 59,1 / 22,9 %, mit exaktem θ in 61,7 / 25,6 % (in allen 36 Zellen dieser Art 10,2 bis 74,2 % mit exaktem θ, die Abdeckung steigt mit der Zahl der Cluster kaum)", "snippets": ("88,9 / 86,1 / 84,3 %", "89,0 / 62,4 / 32,6 %", "89,0 / 75,6 / 60,5 %", "35,4 / 59,1 / 61,7 %", "4,9 / 22,9 / 25,6 %", "10,2 bis 74,2 %")},
    "z8_trend": {"demo": "Trend", "repo": "trend-spitzen-demo", "text": "Jahre bis 80 % Trennschärfe laut Formel 156 / 99 / 62 / 39 bei β = 0,005 / 0,01 / 0,02 / 0,04 σ je Jahr (Simulation bei diesen Jahren 80,5 % / 80,3 % / 79,8 % / 76,6 %; Test auf Lage-Trend im Gumbel-Modell, ξ bekannt; mit freiem ξ nicht gemessen); "
                 "das Trendmodell mit freiem ξ gegen die stationäre GEV: 4 klar besser, 7 klar schlechter, 1 im Rauschen (12 Zellen, R = 400)", "snippets": ("156 / 99 / 62 / 39", "80,5 % / 80,3 % / 79,8 % / 76,6 %", "4 klar besser, 7 klar schlechter, 1 im Rauschen")},
}


def _c(dist, m, q, method, seed_index=0):
    return REF.matrix_cell(dist, m, q, seed_index)[method]


def build_rows() -> list:
    """Alle Zeilen der Tabelle als Wörterbücher: id, Situation, Messung (Text), Verfahren (Hinweis), Herkunft."""
    nr = lambda m, T: pct(float(V.not_reached(m, T)))
    g = REF.grid()
    fixed = V.fixed_gumbel_cells()
    lo_h, hi_h, _ = V.win_band("hill50")
    lo_h5, hi_h5, _ = V.win_band("hill5")
    under16 = V.gumbel_underestimates(0.9)
    seeds = range(len(REF.seed_keys()))
    under_seeds = [len(V.gumbel_underestimates(0.9, i)) for i in seeds]
    logn_under = [100 * REF.matrix_cell("lognormal", 10, 0.01, i)["gumbel"]["under"] for i in seeds]
    nor_high = [100 * (1 - REF.matrix_cell("normal", 50, 0.01, i)["gumbel"]["under"]) for i in seeds]
    edge_all = [100 * a / b for i in seeds for d in S.MATRIX_DISTS for a, b in [REF.matrix_edge(d, 10, i)]]
    kp = g["kopplung"]
    pf = REF.puffer()
    e_exp = _c("exp", 10, 0.01, "gev_ml")["median_abs"]
    l_exp = _c("exp", 10, 0.01, "gev_lmom")["median_abs"]
    e_par = _c("pareto", 10, 0.01, "gev_ml")["median_abs"]
    l_par = _c("pareto", 10, 0.01, "gev_lmom")["median_abs"]
    edge = {d: REF.matrix_edge(d, 10) for d in S.MATRIX_DISTS}
    edge_lo = min(100 * a / b for a, b in edge.values())
    edge_hi = max(100 * a / b for a, b in edge.values())
    vb2000 = pf["cells"]["vehikel_b|2000"]["rows"]
    cv = REF.cvar()["emp_r20000"]
    pareto_g = _c("pareto", 50, 0.01, "gumbel")
    exp_g, exp_e = _c("exp", 50, 0.01, "gumbel"), _c("exp", 50, 0.01, "gev_ml")
    nor_g, nor_e = _c("normal", 50, 0.01, "gumbel"), _c("normal", 50, 0.01, "gev_ml")
    nor_l = _c("normal", 50, 0.01, "gev_lmom")
    h5p = _c("pareto", 50, 0.01, "hill5")
    h5e = _c("exp", 50, 0.01, "hill5")
    h5n = _c("normal", 50, 0.01, "hill5")
    h5l = _c("lognormal", 50, 0.01, "hill5")
    rows = [
        dict(id="r_reihe", situation="Reihe kürzer als die Periode (m = T/2, je Block)",
             messung=f"{nr(50, 100)} % der 50-Jahres-Reihen erreichen den 100er-Pegel nicht (exakt, (1 − 1/T)^m), {nr(50, 1000)} % den 1000er-Pegel; bei m = T = 100 sind es noch {nr(100, 100)} %. "
                     f"Pegel aus 50 Jahren, {ZITATE['z3_band']['text']}.",
             verfahren="keine Schätzung ohne Extrapolations-Hinweis; das Band zeigen, nie nur eine Zahl", herkunft="exakt (diese Demo); „Wiederkehrwert“ (Band)"),
        dict(id="r_kurz", situation="Kurze Reihe (m ≤ 20 Maxima, je Block)",
             messung=f"Median des Betrags von ln(Pegel-Dach/wahr) bei m = 10, 100-Jahres-Pegel (400 Läufe): GEV-ML {de(e_exp)} gegen L-Momente {de(l_exp)} bei Exponential, {de(e_par)} gegen {de(l_par)} bei der Pareto-Verteilung; "
                     f"die RMSE der ML sind ausreißerbestimmt (hier nicht als Zahl geführt); die ML klebt bei m = 10 in {de(edge_lo, 1)} bis {de(edge_hi, 1)} % der Läufe an der Grenze ξ = −1 (erster Zufalls-Seed; über alle drei {de(min(edge_all), 2)} bis {de(max(edge_all), 1)} %). {ZITATE['z2_lmom']['text']}.",
             verfahren="L-Momente (ξ̂ geklammert), oder ξ fest nur bei begründetem ξ (bei falschem ξ = 0 liegt der Fehler bei der Pareto-Verteilung über dem der ML); die ML nicht ohne Randbehandlung", herkunft="Matrix (diese Demo); „GEV an Blockmaxima“"),
        dict(id="r_gumbel_fest", situation="Rand aus Fachwissen sicher ξ = 0 (je Block)",
             messung=f"Exponential, m = 50, 100-Jahres-Pegel (400 Läufe): RMSE von ln {de(exp_g['rmse'])} (Gumbel, ξ = 0 fest) gegen {de(exp_e['rmse'])} (GEV-ML); Gumbel gewinnt dort und in allen {len(fixed)} festen Zellen "
                     f"(alle Exponential-Zellen und die Normal-Zellen mit m = 10 und 20) in allen drei Zufalls-Seeds. Bei wahrem ξ = 0,5 (Pareto-Verteilung) liegt dasselbe Verfahren im Median bei {de(pareto_g['median_ratio'])} des wahren Pegels.",
             verfahren="Gumbel nur mit Begründung; sonst GEV frei", herkunft="Matrix (diese Demo)"),
        dict(id="r_schwer", situation="Rand unbekannt, ξ > 0 plausibel",
             messung=f"Gumbel unterschätzt bei der Pareto-Verteilung α = 2 (m = 50, 100-Jahres-Pegel, 400 Läufe) im Median bei {de(pareto_g['median_ratio'])} des wahren Pegels in {pct(pareto_g['under'])} % der Läufe; "
                     f"in {len(under16)} von 32 Zellen (Pareto-Verteilung und Lognormal, erster Zufalls-Seed; über alle drei Zufalls-Seeds {min(under_seeds)} bis {max(under_seeds)}, die Lognormal-Zelle m = 10, 100-Jahres-Pegel liegt bei {de(min(logn_under), 1)} bis {de(max(logn_under), 1)} %) in mindestens 90 % der Läufe. Vehikel B („Pegel statt Index“, nur als Band): {ZITATE['z6_vb']['text']}.",
             verfahren="nicht Gumbel", herkunft="Matrix (diese Demo); „Pegel statt Index“"),
        dict(id="r_normal", situation="Normal-artig (leichter Rand, ξ_u < 0)",
             messung=f"Gumbel überschätzt leicht (Median {de(nor_g['median_ratio'])} bei m = 50, in {pct(1 - nor_g['under'])} % der Läufe zu hoch, erster Zufalls-Seed; über alle drei {de(min(nor_high), 1)} bis {de(max(nor_high), 1)} %); der Fehler ist bei GEV-ML, L-Momenten und Gumbel ähnlich "
                     f"(RMSE {de(nor_e['rmse'])} / {de(nor_l['rmse'])} / {de(nor_g['rmse'])}, Gumbel etwas höher), der Unterschied liegt vor allem in der Verzerrung. {ZITATE['z1_normal']['text']}.",
             verfahren="GEV frei verzerrt weniger; ξ = 0 fest kostet hier kaum Fehler, aber eine Überschätzung", herkunft="Matrix (diese Demo); „Das Größte von n“"),
        dict(id="r_block_schwelle", situation="Block gegen Schwelle (je Beobachtung)",
             messung=f"„Überschreitungen und die GPD“ führt die Messung: {ZITATE['z4_block']['text']}. Die Matrix-Zelle hier (m = 50, b = 100, k = 50 = n/b, Ziel nah an den Daten, 2 000 Läufe) gibt das RMSE-Verhältnis Schwelle/Block "
                     f"{de(kp['exp']['ratio_rmse'])} / {de(kp['pareto']['ratio_rmse'])} / {de(kp['normal']['ratio_rmse'])} / {de(kp['lognormal']['ratio_rmse'])} (Exponential / Pareto-Verteilung / Normal / Lognormal); "
                     f"bei k = 250 (fünffache Zahl benutzter Werte) ist der RMSE der Schwelle kleiner ({de(_c('pareto', 50, 0.01, 'gpd_ml5')['rmse'])} gegen {de(_c('pareto', 50, 0.01, 'gev_ml')['rmse'])} bei der Pareto-Verteilung), "
                     f"mit k = 50 liegt die Schwelle dort bei {de(_c('pareto', 50, 0.01, 'gpd_ml50')['rmse'])} und damit über den Blöcken; das sagt nichts über „die Schwelle nutzt Daten besser“.",
             verfahren="Schwelle, wenn man viele Überschreitungen hat und die Schwelle begründen kann; sonst Block; die Wertung „die Schwelle nutzt die Daten besser“ gilt nur mit der Kopplung", herkunft="„Überschreitungen und die GPD“; Matrix (diese Demo)"),
        dict(id="r_k_klein", situation="Wenige Überschreitungen (k < 100)",
             messung=f"{ZITATE['z4_klein']['text']}.", verfahren="PWM oder Block; k nennen", herkunft="„Überschreitungen und die GPD“"),
        dict(id="r_schwellenwahl", situation="Schwelle wählen",
             messung=f"{ZITATE['z5_regeln']['text']}.", verfahren="Schwelle begründen, Stabilität prüfen; keine Automatik als Beleg (die Regeln sind eigene Festlegungen von „Schwellenwahl“)", herkunft="„Schwellenwahl“"),
        dict(id="r_hill", situation="Hill-Weissman",
             messung=f"Matrix (m = 50, 100-Jahres-Pegel, 400 Läufe, k = 5 % der Beobachtungen): RMSE {de(h5p['rmse'])} bei der Pareto-Verteilung (kleinster der Zelle), aber {de(h5e['rmse'])} / {de(h5n['rmse'])} / {de(h5l['rmse'])} bei Exponential / Normal / Lognormal; "
                     f"Median Pegel-Dach/wahr {de(h5e['median_ratio'])} bei Exponential, in {pct(1 - h5e['under'])} % der Läufe zu hoch. Über alle 32 Zellen (drei Zufalls-Seeds) gewinnt Hill {band(lo_h5, hi_h5)} Zellen mit k = 5 % und {band(lo_h, hi_h)} mit k = 50, jeweils nur in Zellen der Pareto-Verteilung. {ZITATE['z6_hill']['text']}.",
             verfahren="nur nach einer Prüfung auf ein Potenzgesetz; den Pegel statt des Index prüfen", herkunft="Matrix (diese Demo); „Pegel statt Index“"),
        dict(id="r_abhaengig", situation="Abhängigkeit (θ < 1)",
             messung=f"{ZITATE['z7_abh']['text']}.", verfahren="Block (Länge ≫ 1/θ) lag in „Spitzen-Cluster“ bei φ = 0,5 und 0,8 über dem Schwellenweg (nur b = 50 gemessen); auch der Block erreicht die 95 % nicht, und die Schwelle mit exaktem θ bleibt darunter; Cluster zählen (Warnung bei θ·k < 50, eigene Festlegung, kein Beleg, dass das Intervall darüber hält)", herkunft="„Spitzen-Cluster“"),
        dict(id="r_trend", situation="Trend (je Jahr)",
             messung=f"{ZITATE['z8_trend']['text']}.", verfahren="den Pegel von heute schätzen, nicht den der Reihe; ein Trendmodell erst ab etwa 50 Jahren und stärkerem Trend", herkunft="„Trend“"),
        dict(id="r_intervall", situation="Intervall (je Block, 95 % nominell)",
             messung=f"{ZITATE['z3_delta']['text']}; {ZITATE['z3_profil']['text']}; {ZITATE['z3_boot']['text']}. {ZITATE['z3_haupt']['text']}.",
             verfahren="Profil-Likelihood, nicht Delta; die Überdeckung ist nur über viele Wiederholungen messbar", herkunft="„Wiederkehrwert“"),
        dict(id="r_puffer", situation="Quantil jenseits der Daten (n·p < 1; je Beobachtung)",
             messung=f"P(q > 2p) = (1 − 2p)^n: {de(B.prob_q_above(5000, 1e-4), 2)} bei n·p = 0,5, {de(B.prob_q_above(2000, 1e-4), 2)} bei 0,2, {de(B.prob_q_above(500, 1e-4), 2)} bei 0,05 (exakt, Beta, verteilungsfrei). "
                     f"Vehikel B, n = 2 000, p = 10⁻⁴: GPD {de(vb2000['gpd|0.0001']['gt2'], 2)}, Hill-Weissman {de(vb2000['hill|0.0001']['gt2'], 2)}, GEV {de(vb2000['gev|0.0001']['gt2'], 2)} gegen {de(vb2000['emp|0.0001']['gt2'], 2)} beim empirischen Quantil "
                     f"(1 000 Läufe); bei n·p = 0,2 wird der Gesamtpuffer der GPD nicht kleiner ({de(vb2000['gpd|0.0001']['tot'], 2)} gegen {de(vb2000['emp|0.0001']['tot'], 2)}); die Normal-Formel überschreitet das Ziel bei p = 10⁻⁴ in {pct(1 - vb2000['normal|0.0001']['hit'], 1)} % der Läufe "
                     f"(Median q/p {de(vb2000['normal|0.0001']['medq'], 0)}); sechs Monatsspitzen: P(q > 2p) = (5/6)^6 = {de(float(B.blocks_max_prob(6, 1 / 12)), 3)} bei p = 1/12.",
             verfahren="n·p nennen; bei n·p < 1 einen Aufschlag erwarten, denn kein Rand-Verfahren hält das Ziel in allen vier Fällen (n·p = 0,2: GPD und GEV in 36 bis 60 % der Läufe, Hill-Weissman in drei Fällen mit einem 1,1- bis 1,7-fachen Puffer, bei der Pareto-Verteilung in 50 %); die Normal-Formel nicht bei schwerem Rand (Pareto-Verteilung α = 2 und Vehikel B: sie überschreitet das Ziel häufiger als das empirische Quantil)", herkunft="Puffer (diese Demo)"),
        dict(id="r_cvar", situation="CVaR aus Szenarien",
             messung=f"n = 600, α = 0,90 (Voreinstellung der `cvar-demo`): Median {de(cv['pareto|600']['0.9']['median'], 3)} (Pareto-Verteilung) bis {de(cv['normal|600']['0.9']['median'], 3)} (Normal) des echten; "
                     f"α = 0,99: {de(cv['pareto|600']['0.99']['median'], 2)} bis {de(cv['normal|600']['0.99']['median'], 2)}; α = 0,999: {de(cv['pareto|600']['0.999']['median'], 2)} bis {de(cv['normal|600']['0.999']['median'], 2)} (20 000 Läufe, fünf ausgedachte Verteilungen).",
             verfahren="Verweis `cvar-demo`; die Tail-Form nur mit einer Rand-Schätzung; das α der Demo nennen", herkunft="CVaR-Raster (diese Demo)"),
        dict(id="r_mehrdim", situation="Mehrdimensional", messung="nicht gerechnet.", verfahren="Verweis `randabhaengigkeit-demo`", herkunft="-"),
    ]
    return rows


_ROWS: list | None = None


def rows() -> list:
    global _ROWS
    if _ROWS is None:
        _ROWS = build_rows()
    return _ROWS


def row_by_id(rid: str) -> dict:
    return next(r for r in rows() if r["id"] == rid)
