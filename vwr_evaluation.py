"""Auswertung: die reine Funktion `hinweise`, die exakten Größen, die Analysen der vier Experimente und die Verdikte als reine Funktionen.

Diese Demo fasst zusammen: Zahlen, die eine andere Demo der Linie führt („Das Größte von n“ Normierung, „GEV an Blockmaxima“ L-Momente, „Wiederkehrwert“ Intervalle und Streuband, „Überschreitungen und die GPD“ Block gegen Schwelle,
„Schwellenwahl“, „Pegel statt Index“ Hill, „Spitzen-Cluster“ Abhängigkeit, „Trend“), stehen in `vwr_table.ZITATE` mit Demo und Wortlaut und werden nicht neu hergeleitet. Selbst gemessen sind nur die Matrix (`vwr_grid.json`), die CVaR-Einordnung (`vwr_cvar.json`) und der Puffer
(`vwr_puffer.json`); die Reihenlänge gegen die Wiederkehrperiode ist exakt. Alle Verteilungen sind ausgedacht; über den Nutzen in Logistik, Lager oder Hafen sagt nichts hier etwas."""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

import vwr_buffer as B
import vwr_constants as C
import vwr_cvar as CV
import vwr_estimators as E
import vwr_reference as REF
import vwr_scenario as S
import vwr_vecrng as VR

LENGTHS = (10, 20, 50, 100)
TARGETS = (0.01, 0.001)
TARGET_YEARS = {0.01: 100, 0.001: 1000}


# --- Zahlen schreiben (Dezimalkomma, Tausenderpunkt) --------------------------------------------------------------------------------------------
def de(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x: float, nd: int = 1) -> str:
    """Anteil 0 bis 1 als Prozentzahl mit Komma, ohne Prozentzeichen."""
    return de(100.0 * x, nd)


def thousands(n) -> str:
    return f"{int(round(n)):,}".replace(",", ".")


def band(lo, hi) -> str:
    """„6“ oder „12 bis 14“."""
    return str(lo) if lo == hi else f"{lo} bis {hi}"


def p_text(p: float) -> str:
    """10⁻² usw. für die Zehnerpotenzen der Puffer-Ziele."""
    e = int(round(math.log10(p)))
    sup = {-1: "⁻¹", -2: "⁻²", -3: "⁻³", -4: "⁻⁴", -5: "⁻⁵"}.get(e, f"^{e}")
    return f"10{sup}" if abs(p - 10.0 ** e) < 1e-12 * p else de(p, 4)


# --- Einstellungen -----------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    m1: int = C.DEFAULT_M1
    T1: int = C.DEFAULT_T
    k1: int = C.DEFAULT_K1
    theta1: float = C.DEFAULT_THETA
    beta1: float = C.DEFAULT_BETA
    rand1: str = C.DEFAULT_RAND
    dist2: str = C.DEFAULT_DIST2
    m2: int = C.DEFAULT_M2
    target2: int = C.DEFAULT_TARGET2
    R2: int = C.DEFAULT_R2
    seed: int = C.DEFAULT_SEED
    case3: str = C.DEFAULT_CASE3
    n3: int = C.DEFAULT_N3
    p3: float = C.DEFAULT_P3
    dist4: str = C.DEFAULT_DIST4
    n4: int = C.DEFAULT_N4
    alpha4: float = C.DEFAULT_ALPHA4


# --- Exakte Größen: Reihenlänge gegen Wiederkehrperiode, Trennschärfe des Trendtests ---------------------------------------------------------------
def not_reached(m_years: int, period: int) -> Fraction:
    """Anteil der Reihen aus m iid-Jahresmaxima ohne eine Beobachtung auf der Höhe des T-Jahres-Pegels: (1 − 1/T)^m (exakt)."""
    return B.p_series_not_reaching(m_years, period)


def _phi(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def trend_power(beta: float, m_years: int) -> float:
    """Trennschärfe des Likelihood-Quotienten-Tests auf einen Lage-Trend β (σ-Einheiten je Jahr) bei m Jahren, ξ bekannt (asymptotisch, λ = β² m (m²−1)/12; „Trend“)."""
    lam = beta * beta * m_years * (m_years * m_years - 1) / 12.0
    r = math.sqrt(lam)
    return _phi(r - 1.959964) + _phi(-r - 1.959964)


def years_for_80(beta: float) -> int:
    """Kleinste Zahl Jahre m, bei der der Test auf Lage-Trend β mindestens 80 % Trennschärfe hat (Test im Gumbel-Modell mit bekanntem ξ; mit freiem ξ nicht gemessen, also eine Untergrenze)."""
    if beta <= 0:
        raise ValueError("β muss positiv sein")
    m = 3
    while trend_power(beta, m) < 0.8:
        m += 1
    return m


# --- Die reine Funktion `hinweise` ---------------------------------------------------------------------------------------------------------------
HINT_CODES = ("extrapolation", "kurze_reihe", "k_klein", "wenig_cluster", "trend_nicht_erkennbar")
HINT_ROW = {"extrapolation": "r_reihe", "kurze_reihe": "r_kurz", "k_klein": "r_k_klein", "wenig_cluster": "r_abhaengig", "trend_nicht_erkennbar": "r_trend"}
LIMIT_EXTRAPOLATION = 0.5       # eigene Festlegung: mindestens die Hälfte der Reihen enthält keinen Wert dieser Höhe
LIMIT_SHORT = 20                # eigene Festlegung: m ≤ 20 Maxima
LIMIT_K = 100                   # eigene Festlegung: k < 100 Überschreitungen
LIMIT_CLUSTER = 50              # eigene Festlegung: θ·k < 50 Cluster


def hinweise(m_years: int, period: int, k=None, theta: float = 1.0, beta=None) -> list:
    """Codes der zutreffenden Warnungen in fester Reihenfolge. Die Schwellen 50 %, 20, 100 und 50 sind eigene Festlegungen aus den Zahlen der Tabelle, keine Literaturregeln:
    extrapolation (mindestens die Hälfte der Reihen erreicht den Pegel nicht, exakt), kurze_reihe (m ≤ 20), k_klein (k < 100), wenig_cluster (θ·k < 50),
    trend_nicht_erkennbar (m unter den Jahren für 80 % Trennschärfe; nur für den Test im Gumbel-Modell mit bekanntem ξ)."""
    w = []
    if float(not_reached(m_years, int(period))) >= LIMIT_EXTRAPOLATION:
        w.append("extrapolation")
    if m_years <= LIMIT_SHORT:
        w.append("kurze_reihe")
    if k is not None and k < LIMIT_K:
        w.append("k_klein")
    if k is not None and theta * k < LIMIT_CLUSTER:
        w.append("wenig_cluster")
    if beta is not None and beta > 0 and m_years < years_for_80(beta):
        w.append("trend_nicht_erkennbar")
    return w


def hint_texts(s: Settings) -> list:
    """(Code, Text) der zutreffenden Warnungen mit der Zahl dieser Einstellung; die Herkunft steht in der Tabelle (Zeile `HINT_ROW[code]`)."""
    codes = hinweise(s.m1, s.T1, s.k1, s.theta1, s.beta1 if s.beta1 > 0 else None)
    out = []
    for c in codes:
        if c == "extrapolation":
            t = f"Extrapolation: in {pct(float(not_reached(s.m1, s.T1)))} % der {s.m1}-Jahres-Reihen liegt keine Beobachtung auf der Höhe des {s.T1}-Jahres-Pegels (exakt); der Pegel ist keine Ablesung."
        elif c == "kurze_reihe":
            t = f"Kurze Reihe: {s.m1} Maxima (höchstens {LIMIT_SHORT}); die ML-Anpassung der GEV ist dort instabil, L-Momente sind stabiler, ein fest gewähltes ξ nur bei begründetem ξ (Tabelle)."
        elif c == "k_klein":
            t = f"Wenige Überschreitungen: k = {s.k1} (unter {LIMIT_K}); die Streuung (1 + ξ)/√k gilt erst ab k von etwa 100 bis 200."
        elif c == "wenig_cluster":
            t = f"Wenig Cluster: θ·k = {de(s.theta1 * s.k1, 0)} Cluster unter den {s.k1} Überschreitungen (unter {LIMIT_CLUSTER}, eigene Festlegung); die Überdeckung der Intervalle bleibt unter 95 %, und in „Spitzen-Cluster“ stieg sie mit mehr Clustern kaum (Tabelle)."
        else:
            t = f"Trend nicht erkennbar: bei {de(s.beta1, 3)} σ je Jahr braucht der Test laut Formel {years_for_80(s.beta1)} Jahre für 80 % Trennschärfe („Trend“: die Simulation erreicht bei diesen Längen 76,6 bis 80,5 %), die Reihe hat {s.m1}."
        out.append((c, t))
    return out


# --- Zitierte Zahlen aus „Wiederkehrwert“ (Streuband, GEV(100; 20; 0,2), 90-%-Band von x̂/x_T, R = 2 000) ---------------------------------------------------
BAND_STUECK3 = {   # (m, T) -> (q05, q95, Median)
    (20, 10): (0.83, 1.22, 0.98), (20, 100): (0.63, 2.12, 0.96), (20, 1000): (0.43, 4.92, 0.95),
    (50, 10): (0.89, 1.13, 0.99), (50, 100): (0.73, 1.49, 0.99), (50, 1000): (0.56, 2.21, 0.98),
    (100, 10): (0.92, 1.09, 1.00), (100, 100): (0.80, 1.29, 1.00), (100, 1000): (0.67, 1.62, 1.01),
    (200, 10): (0.94, 1.06, 1.00), (200, 100): (0.85, 1.19, 0.99), (200, 1000): (0.74, 1.41, 0.99),
    (500, 10): (0.97, 1.04, 1.00), (500, 100): (0.91, 1.10, 1.00), (500, 1000): (0.84, 1.21, 1.00),
}


@dataclass(frozen=True)
class Lage:
    s: Settings
    not_reached: float
    hints: tuple
    band: tuple | None


def analyse_lage(s: Settings) -> Lage:
    band = BAND_STUECK3.get((s.m1, s.T1))
    return Lage(s, float(not_reached(s.m1, s.T1)), tuple(hint_texts(s)), band)


def relevant_rows(s: Settings, hints) -> list:
    """Zeilen der Entscheidungstabelle, die zur Lage passen: je Warnung ihre Zeile, dazu die Zeile zum Rand (Regler „Rand“) und immer Intervall und Puffer."""
    rows = [HINT_ROW[c] for c, _ in hints]
    rows += {"bekannt": ["r_gumbel_fest"], "schwer": ["r_schwer", "r_hill"], "unbekannt": ["r_schwer"]}[s.rand1]
    rows += ["r_intervall", "r_puffer"]
    out = []
    for r in rows:
        if r not in out:
            out.append(r)
    return out


def verdict_lage(a: Lage):
    s = a.s
    text = (f"{s.m1} Jahre, Ziel {s.T1}-Jahres-Pegel: in {pct(a.not_reached)} % der Reihen liegt keine Beobachtung auf dieser Höhe (exakt, (1 − 1/{s.T1})^{s.m1}), in {pct(1 - a.not_reached)} % mindestens eine. ")
    if a.band is not None:
        lo, hi, med = a.band
        text += (f"Aus {s.m1} Jahresmaxima geschätzt liegt der Pegel zu 90 % zwischen dem {de(lo)}- und {de(hi)}-Fachen des wahren (Median {de(med)}; GEV(μ = 100, σ = 20, ξ = 0,2), "
                 "2 000 Läufe, „Wiederkehrwert“). ")
    else:
        text += "Das Streuband dieser Kombination hat „Wiederkehrwert“ nicht gemessen; hier steht nur der exakte Anteil. "
    if a.hints:
        text += "Zutreffende Hinweise: " + "; ".join(t.split(":")[0] for _, t in a.hints) + "."
        return "warning", text
    return "success", text + "Kein Hinweis trifft zu (die Schwellen der Hinweise sind eigene Festlegungen)."


# --- Experiment 2: alle Verfahren auf dieselbe Reihe ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Live:
    dist: str
    m: int
    target: int
    R: int
    truth: float
    errors: dict            # Verfahren -> Feld der Fehler ln(Pegel-Dach/wahr), nan bei ungültigen Läufen
    first: dict             # Verfahren -> Pegel der ersten Reihe


def analyse_live(s: Settings) -> Live:
    """R Reihen aus der gewählten Verteilung (SplitMix64, Zufalls-Seed der Seitenleiste), alle Verfahren auf dieselben Reihen; Fehler gegen den exakten wahren Pegel."""
    q = 0.01 if s.target2 == 100 else 0.001
    ny = S.NY
    stream = VR.Stream(s.seed + C.SEED_OFFSET_2)
    X = S.draw(s.dist2, stream, (s.R2, ny * s.m2))
    truth = S.true_level(s.dist2, q)
    methods = E.matrix_methods_for(ny * s.m2)
    errors = {mt: np.full(s.R2, np.nan) for mt in methods}
    first = {}
    for r in range(s.R2):
        lv = E.levels_matrix(X[r], ny, (q,), methods)
        for mt in methods:
            errors[mt][r] = E.log_error(lv[mt][0], truth)
        if r == 0:
            first = {mt: lv[mt][0] for mt in methods}
    return Live(s.dist2, s.m2, s.target2, s.R2, truth, errors, first)


def live_summary(live: Live, method: str) -> dict:
    e = live.errors[method]
    ok = e[np.isfinite(e)]
    if ok.size == 0:
        return {"n": 0, "median_ratio": math.nan, "rmse": math.nan, "under": math.nan, "q10": math.nan, "q90": math.nan}
    return {"n": int(ok.size), "median_ratio": float(math.exp(np.median(ok))), "rmse": float(math.sqrt(np.mean(ok * ok))), "under": float(np.mean(ok < 0)),
            "q10": float(math.exp(np.quantile(ok, 0.1))), "q90": float(math.exp(np.quantile(ok, 0.9)))}


def frozen_cell(s: Settings) -> dict | None:
    """Die eingefrorene Matrix-Zelle (Zufalls-Seed-Basis 0) zur Auswahl; None bei Vehikel B (nicht in der Matrix, Z5 h)."""
    if s.dist2 == "vehikel_b":
        return None
    q = 0.01 if s.target2 == 100 else 0.001
    return REF.matrix_cell(s.dist2, s.m2, q, 0)


def cell_winner(cell: dict, with_hill: bool = True, runs: int = 400):
    """Verfahren mit dem kleinsten RMSE der Zelle (nur Verfahren mit mindestens 90 % gültigen Läufen); (RMSE, Verfahren)."""
    best = None
    for mt, st in cell.items():
        if not with_hill and mt.startswith("hill"):
            continue
        if st["n"] < 0.9 * runs:
            continue
        if best is None or st["rmse"] < best[0]:
            best = (st["rmse"], mt)
    return best


def win_counts(seed_index: int, with_hill: bool = True) -> dict:
    """Zählung der Siege über die 32 Zellen (4 Verteilungen × 4 Längen × 2 Ziele) für eine Zufalls-Seed-Basis."""
    counts: dict = {}
    for d in S.MATRIX_DISTS:
        for m in LENGTHS:
            for q in TARGETS:
                w = cell_winner(REF.matrix_cell(d, m, q, seed_index), with_hill)
                counts[w[1]] = counts.get(w[1], 0) + 1
    return counts


def win_band(method: str, with_hill: bool = True) -> tuple:
    """(kleinste, größte Zahl der Siege über die drei Zufalls-Seed-Basen, Zahlen je Basis)."""
    per = [win_counts(i, with_hill).get(method, 0) for i in range(len(REF.seed_keys()))]
    return min(per), max(per), tuple(per)


def fixed_gumbel_cells() -> list:
    """Zellen (Verteilung, m, Ziel), die Gumbel in allen drei Zufalls-Seed-Basen gewinnt."""
    out = []
    for d in S.MATRIX_DISTS:
        for m in LENGTHS:
            for q in TARGETS:
                if all(cell_winner(REF.matrix_cell(d, m, q, i))[1] == "gumbel" for i in range(len(REF.seed_keys()))):
                    out.append((d, m, q))
    return out


def gumbel_underestimates(threshold: float = 0.9, seed_index: int = 0) -> list:
    """Zellen, in denen Gumbel in mindestens `threshold` der Läufe unterschätzt."""
    out = []
    for d in S.MATRIX_DISTS:
        for m in LENGTHS:
            for q in TARGETS:
                if REF.matrix_cell(d, m, q, seed_index)["gumbel"]["under"] >= threshold:
                    out.append((d, m, q))
    return out


def best_rmse_range(m: int = 50, target: float = 0.01, seed_index: int = 0) -> dict:
    """Kleinster RMSE (über alle Verfahren der Zelle) je Verteilung bei m Jahren und gegebenem Ziel."""
    return {d: cell_winner(REF.matrix_cell(d, m, target, seed_index)) for d in S.MATRIX_DISTS}


def verdict_matrix(s: Settings, live: Live):
    """Verdikt zu Experiment 2: die Zelle der Auswahl (eingefroren, R = 400) und die Zählung über alle 32 Zellen."""
    lo, hi, per = win_band("gumbel")
    lo2, hi2, _ = win_band("gpd_ml5", with_hill=False)
    q = 0.01 if s.target2 == 100 else 0.001
    head = ""
    if s.dist2 != "vehikel_b":
        cell = REF.matrix_cell(s.dist2, s.m2, q, 0)
        g, gm = cell["gumbel"], cell["gev_ml"]
        best = cell_winner(cell)
        head = (f"{S.DIST_SHORT[s.dist2]}, {s.m2} Jahre, {s.target2}-Jahres-Pegel (eingefroren, 400 Läufe): Gumbel (ξ = 0 fest) liegt im Median bei {de(g['median_ratio'])} des wahren Pegels "
                f"und unterschätzt in {pct(g['under'])} % der Läufe (RMSE von ln {de(g['rmse'])}, GEV-ML {de(gm['rmse'])}); kleinster RMSE: {E.METHOD_SHORT[best[1]]} mit {de(best[0])}. ")
    else:
        head = "Vehikel B ist nicht in der eingefrorenen Matrix (nur die Live-Läufe oben zeigen es). "
    text = (head + f"Über alle 32 Zellen gewinnt Gumbel {band(lo, hi)} (drei Zufalls-Seeds; Zählung ± 3), die GPD-ML mit k = 5 % ohne Hill {band(lo2, hi2)}: kein Verfahren gewinnt durchgehend, und ein Sieg "
            "heißt nur kleinster RMSE in dieser Zelle bei diesen festen k.")
    return ("info", text)


# --- Experiment 3: wie viel bleibt unsicher? Puffer jenseits der Daten ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Puffer:
    case: str
    n: int
    p: float
    np_product: float
    j: int
    p_gt2: float            # exakt, verteilungsfrei
    p_within: float         # P(q ≤ p), exakt
    e_q_over_p: float
    rows: dict              # Verfahren -> eingefrorene Kennzahlen (R = 1 000 Läufe)
    btrue: float


def analyse_puffer(s: Settings) -> Puffer:
    pf = REF.puffer()
    cell = pf["cells"][f"{s.case3}|{s.n3}"]
    pi = pf["ps"].index(s.p3)
    rows = {mt: cell["rows"][f"{mt}|{s.p3:g}"] for mt in E.PUFFER_METHODS}
    return Puffer(s.case3, s.n3, s.p3, s.n3 * s.p3, E.emp_index(s.n3, s.p3), B.prob_q_above(s.n3, s.p3), B.prob_q_within(s.n3, s.p3), B.expected_q(s.n3, s.p3) / s.p3, rows, cell["btrue"][pi])


def verdict_puffer(a: Puffer):
    """Verdikt zu Experiment 3. Das empirische Quantil ist exakt berechenbar (Beta), die Rand-Verfahren sind eingefroren gemessen (1 000 Läufe)."""
    emp, gpd = a.rows["emp"], a.rows["gpd"]
    case = S.DIST_LABELS[a.case]
    text = (f"Puffer zum Ziel p = {p_text(a.p)} je Beobachtung bei n = {thousands(a.n)} Beobachtungen (n·p = {de(a.np_product, 2)}): das empirische Quantil übersteigt das Ziel mit Wahrscheinlichkeit "
            f"{de(a.p_gt2, 3)} um mehr als das Doppelte (exakt, Beta-Verteilung, für jede stetige Verteilung gleich); seine wahre Überschreitungswahrscheinlichkeit ist im Mittel das {de(a.e_q_over_p, 2)}-Fache des Ziels. ")
    if a.np_product >= 5:
        text += (f"Ab n·p ≈ 5 genügt das empirische Quantil im Maß P(q > 2p): die GPD an die größten 5 % hat P(q > 2p) = {de(gpd['gt2'], 3)}, das empirische {de(emp['gt2'], 3)} ({case}, 1 000 Läufe). ")
        return "success", text
    text += f"Die GPD an die größten 5 % hat in diesem Fall ({case}, 1 000 Läufe) P(q > 2p) = {de(gpd['gt2'], 3)} gegen {de(emp['gt2'], 3)} beim empirischen Quantil, "
    if gpd["tot"] >= emp["tot"]:
        text += f"der Gesamtpuffer (Median Puffer/wahr × f90) wird aber nicht kleiner: {de(gpd['tot'], 2)} gegen {de(emp['tot'], 2)}; ein kleinerer Aufschlag ist dann ein höherer Grundwert."
    else:
        text += f"und der Gesamtpuffer (Median Puffer/wahr × f90) ist kleiner: {de(gpd['tot'], 2)} gegen {de(emp['tot'], 2)}."
    return ("warning" if a.np_product < 1 else "info"), text


def short_series_rows(xi: float, m: int) -> dict:
    """Eingefrorene Zeilen der kurzen Blockreihe (Teil 2): {Verfahren|p: Kennzahlen}."""
    return REF.puffer()["short"][f"{xi}|{m}"]


# --- Experiment 4: was sieht ein CVaR vom Tail? ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CvarAnalysis:
    dist: str
    n: int
    alpha: float
    exact: float
    ratios: np.ndarray       # Live: Verhältnis empirischer/exakter CVaR je Lauf
    frozen: dict | None      # eingefroren (n = 600: R = 20 000, sonst 2 000)
    frozen_runs: int
    none_above: float        # α^n, exakt
    gpd: dict | None


def analyse_cvar(s: Settings) -> CvarAnalysis:
    R = 100 if s.n4 >= 6000 else C.R4_LIVE
    stream = VR.Stream(s.seed + C.SEED_OFFSET_4)
    X = S.draw(s.dist4, stream, (R, s.n4))
    exact = CV.es_exact(s.dist4, s.alpha4)
    ratios = CV.cvar_empirical_rows(X, s.alpha4) / exact
    d = REF.cvar()
    if s.n4 == 600:
        fr, runs = d["emp_r20000"][f"{s.dist4}|600"][str(s.alpha4)], 20000
    else:
        fr, runs = d["emp_r2000"][f"{s.dist4}|{s.n4}"][str(s.alpha4)], 2000
    g = d["gpd_r2000"].get(f"{s.dist4}|{s.n4}", {}).get(str(s.alpha4))
    return CvarAnalysis(s.dist4, s.n4, s.alpha4, exact, ratios, fr, runs, CV.prob_no_scenario_above_quantile(s.alpha4, s.n4), g)


def verdict_cvar(a: CvarAnalysis):
    fr = a.frozen
    text = (f"{CV.CVAR_LABELS[a.dist]}, n = {thousands(a.n)} Szenarien, α = {de(a.alpha, 3)}: der CVaR aus den Szenarien liegt im Median bei {de(fr['median'], 3)} des exakten ({thousands(a.frozen_runs)} Läufe, eingefroren); "
            f"mit Wahrscheinlichkeit {pct(a.none_above)} % liegt kein einziges Szenario über dem {de(100 * a.alpha, 1)}-%-Quantil (exakt α^n). ")
    if (1 - a.alpha) * a.n < 1:
        text += f"Hier ist (1 − α)·n = {de((1 - a.alpha) * a.n, 2)} unter 1: der empirische CVaR ist das Maximum der Szenarien. "
    text += "Die Verteilungen sind ausgedacht, nicht die der `cvar-demo`."
    kind = "warning" if fr["median"] < 0.9 else "info"
    return kind, text
