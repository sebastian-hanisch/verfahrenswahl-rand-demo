"""Welches Verfahren wann? Wegweiser für den Pegel am Rand - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Demo der Linie „Extremwerte erklärt“ (Kante `kontrast` zu allen Demos davor): stellt die Verfahren der Demos davor nebeneinander und zeigt, wo jedes aufhört. Vier Experimente: meine Lage und die Hinweise
der Entscheidungstabelle, alle Verfahren auf dieselbe Reihe (Matrix aus 32 Zellen), wie viel bleibt unsicher (Streuband zitiert, Puffer jenseits der Daten), was sieht ein CVaR vom Tail. Aufbau nach DEMO-PLAYBOOK §2.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import vwr_buffer as B
import vwr_constants as C
import vwr_cvar as CV
import vwr_estimators as E
import vwr_evaluation as V
import vwr_reference as REF
import vwr_scenario as S
import vwr_table as TB
import vwr_texts as T
from vwr_evaluation import de, p_text, pct, thousands
from vwr_layout import dunkles_design, kopf_css
from vwr_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from vwr_visualization import build_cvar, build_levels, build_matrix, build_puffer, build_reihe

st.set_page_config(page_title="Welches Verfahren wann? – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _live(dist2, m2, target2, R2, seed):
    return V.analyse_live(V.Settings(dist2=dist2, m2=m2, target2=target2, R2=R2, seed=seed))


@st.cache_data(show_spinner=False)
def _cvar(dist4, n4, alpha4, seed):
    return V.analyse_cvar(V.Settings(dist4=dist4, n4=n4, alpha4=alpha4, seed=seed))


def _d(x, nd):
    """Delta-Text mit ASCII-Vorzeichen: Streamlit liest nur '+' und '-' als Richtung."""
    return f"{x:+.{nd}f}".replace(".", ",")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in vwr_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 Welches Verfahren wann? Wegweiser für den Pegel am Rand")
st.markdown(T.INTRO)

with st.expander("So funktioniert der Wegweiser", expanded=True):
    st.markdown(T.HOW_IT_WORKS)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Gilt für die Läufe in Experiment 2 und die Szenarien in Experiment 4 (jedes zieht aus einem eigenen Strom).")
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=randomize_seed)
    st.markdown("**Experiment 1: meine Lage**")
    m1 = st.slider("Reihenlänge m (Jahre)", *bounds("m1"), key="m1", help="Jahre zu je 100 Beobachtungen; die Jahresmaxima sind die Blöcke.")
    T1 = st.select_slider("Wiederkehrperiode T (Jahre)", options=C.T_OPTIONS, key="T1", help="Ziel ist der T-Jahres-Pegel: je Jahr mit Wahrscheinlichkeit 1/T überschritten.")
    k1 = st.slider("Überschreitungen k (Schwellenverfahren)", *bounds("k1"), step=C.K1_STEP, key="k1", help="Zahl der Werte über der Schwelle; die Blockverfahren haben m Maxima.")
    theta1 = st.select_slider("Extremalindex θ", options=C.THETA_OPTIONS, key="theta1", format_func=lambda v: "1 (unabhängig)" if v == 1.0 else de(v, 1), help="Kleiner als 1 heißt: Spitzen kommen in Gruppen; θ·k Gruppen unter den k Überschreitungen.")
    beta1 = st.select_slider("Trend (σ je Jahr)", options=C.BETA_OPTIONS, key="beta1", format_func=lambda v: "kein Trend" if v == 0 else de(v, 3), help="Trend der Lage in Einheiten der Streuung je Jahr.")
    rand1 = st.radio("Rand", C.RAND_OPTIONS, key="rand1", format_func=lambda k: C.RAND_LABELS[k], help="Was über den Rand bekannt ist; bestimmt, welche Zeilen der Tabelle die Seite zusätzlich zeigt.")
    st.markdown("**Experiment 2: alle Verfahren auf dieselbe Reihe**")
    dist2 = st.selectbox("Verteilung", C.DIST2_OPTIONS, key="dist2", format_func=lambda k: S.DIST_LABELS[k], help="Alle Verteilungen sind ausgedacht; Vehikel B ist die Verweilzeit-Mischung der Linie (nicht in der eingefrorenen Matrix).")
    m2 = st.select_slider("Jahre m (Experiment 2)", options=C.M2_OPTIONS, key="m2")
    target2 = st.select_slider("Ziel (Jahre)", options=C.TARGET2_OPTIONS, key="target2", format_func=lambda v: f"{v}-Jahres-Pegel")
    R2 = st.slider("Läufe R (live)", *bounds("R2"), step=C.R2_STEP, key="R2", help="Zähleinheit: unabhängige Reihen. Die Matrix mit 400 Läufen je Zelle ist eingefroren.")
    st.markdown("**Experiment 3: Puffer jenseits der Daten**")
    case3 = st.selectbox("Fall", C.CASE3_OPTIONS, key="case3", format_func=lambda k: S.DIST_LABELS[k])
    n3 = st.select_slider("Beobachtungen n", options=C.N3_OPTIONS, key="n3", format_func=thousands)
    p3 = st.select_slider("Ziel p je Beobachtung", options=C.P3_OPTIONS, key="p3", format_func=p_text, help="n·p ist die erwartete Zahl der Beobachtungen oberhalb des Ziels.")
    st.markdown("**Experiment 4: CVaR**")
    dist4 = st.selectbox("Verteilung (Experiment 4)", C.DIST4_OPTIONS, key="dist4", format_func=lambda k: CV.CVAR_LABELS[k])
    n4 = st.select_slider("Szenarien n", options=C.N4_OPTIONS, key="n4", format_func=thousands, help="n = 600 ist die Zahl der Referenzszenarien der `cvar-demo`.")
    alpha4 = st.select_slider("Niveau α", options=C.ALPHA4_OPTIONS, key="alpha4", format_func=lambda v: de(v, 3))

sync_query_params({"m1": int(m1), "T1": int(T1), "k1": int(k1), "theta1": float(theta1), "beta1": float(beta1), "rand1": rand1, "dist2": dist2, "m2": int(m2), "target2": int(target2), "R2": int(R2),
                   "seed_input": int(seed), "case3": case3, "n3": int(n3), "p3": float(p3), "dist4": dist4, "n4": int(n4), "alpha4": float(alpha4)})

settings = V.Settings(int(m1), int(T1), int(k1), float(theta1), float(beta1), rand1, dist2, int(m2), int(target2), int(R2), int(seed), case3, int(n3), float(p3), dist4, int(n4), float(alpha4))

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

a1 = V.analyse_lage(settings)
st.markdown("## 🎯 Experiment 1: Meine Lage, was die Tabelle dazu sagt")
c1, c2, c3 = st.columns(3)
c1.metric("Reihen ohne Beobachtung auf Pegelhöhe (exakt)", f"{pct(a1.not_reached)} %", delta=f"{settings.m1} Jahre, T = {settings.T1}", delta_color="off")
c2.metric("Reihen mit mindestens einer", f"{pct(1 - a1.not_reached)} %", delta="1 − (1 − 1/T)^m", delta_color="off")
c3.metric("Zutreffende Hinweise", str(len(a1.hints)), delta=" · ".join(c for c, _ in a1.hints) if a1.hints else "keiner", delta_color="off")
st.plotly_chart(build_reihe(a1), width="stretch", key="reihe_chart")
if a1.hints:
    for code, text in a1.hints:
        st.warning(text)
else:
    st.success("Kein Hinweis trifft zu. Die Schwellen der Hinweise sind eigene Festlegungen aus den Zahlen der Tabelle.")
st.caption(T.CAPTION_EXP1)
rel = V.relevant_rows(settings, a1.hints)
st.markdown("**Zutreffende Zeilen der Entscheidungstabelle**")
st.dataframe(pd.DataFrame([{"Situation": TB.row_by_id(r)["situation"], "Was die Messung sagt": TB.row_by_id(r)["messung"], "Verfahren: Hinweis": TB.row_by_id(r)["verfahren"], "Herkunft": TB.row_by_id(r)["herkunft"]} for r in rel]),
             hide_index=True, width="stretch")
with st.expander("📋 Die ganze Entscheidungstabelle (alle Zeilen)"):
    st.dataframe(pd.DataFrame([{"Situation": r["situation"], "Was die Messung sagt": r["messung"], "Verfahren: Hinweis": r["verfahren"], "Herkunft": r["herkunft"]} for r in TB.rows()]),
                 hide_index=True, width="stretch")

st.markdown("---")

# --- Die übrigen Rechnungen (vor den Kernfragen) ------------------------------------------------------------------------------------------------
est2 = settings.R2 * 0.02
if est2 >= 0.8:
    with st.spinner(f"Ziehe {settings.R2} Reihen und passe alle Verfahren an (rund {de(est2, 1)} s, in der Cloud länger) …"):
        live = _live(settings.dist2, settings.m2, settings.target2, settings.R2, settings.seed)
else:
    live = _live(settings.dist2, settings.m2, settings.target2, settings.R2, settings.seed)
a3 = V.analyse_puffer(settings)
a4 = _cvar(settings.dist4, settings.n4, settings.alpha4, settings.seed)

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**1. Was sagt die Tabelle zu meiner Lage?**")
kind, text = V.verdict_lage(a1)
getattr(st, kind)(text)
st.markdown("**2. Welches Verfahren liegt wo daneben?**")
kind, text = V.verdict_matrix(settings, live)
getattr(st, kind)(text)
st.markdown("**3. Wie viel bleibt unsicher, auch beim Puffer?**")
kind, text = V.verdict_puffer(a3)
getattr(st, kind)(text)
st.markdown("**4. Was sieht ein CVaR vom Tail?**")
kind, text = V.verdict_cvar(a4)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Alle Verfahren auf dieselbe Reihe")
st.caption(T.CAPTION_EXP2)
st.plotly_chart(build_levels(live), width="stretch", key="levels_chart")
rows2 = []
for mt in live.errors:
    sm = V.live_summary(live, mt)
    rows2.append({"Verfahren": E.METHOD_LABELS[mt], "gültige Läufe": sm["n"], "Median Pegel-Dach/wahr": de(sm["median_ratio"], 2), "Anteil unter wahr": f"{pct(sm['under'])} %" if sm["n"] else "-",
                  "RMSE von ln": de(sm["rmse"], 2), "10 % bis 90 %": f"{de(sm['q10'], 2)} bis {de(sm['q90'], 2)}"})
st.dataframe(pd.DataFrame(rows2), hide_index=True, width="stretch")
st.caption(f"Wahrer Pegel: {de(live.truth, 3)} (exakt). {live.R} Läufe, Zufalls-Seed {settings.seed}. Bei m = 10 sind k = 50 und k = 5 % der Beobachtungen dasselbe Verfahren.")
st.markdown("**Die Matrix: 4 Verteilungen × 4 Reihenlängen × 2 Ziele (eingefroren, 400 Läufe je Zelle)**")
sel = (settings.dist2, settings.m2, 0.01 if settings.target2 == 100 else 0.001) if settings.dist2 != "vehikel_b" else None
st.plotly_chart(build_matrix(0, sel), width="stretch", key="matrix_chart")
wins_rows = []
for mt in E.MATRIX_METHODS:
    lo, hi, per = V.win_band(mt)
    lo2, hi2, per2 = V.win_band(mt, with_hill=False)
    wins_rows.append({"Verfahren": E.METHOD_LABELS[mt], "Siege (drei Zufalls-Seeds)": " / ".join(str(v) for v in per), "Siege ohne Hill": ("-" if mt.startswith("hill") else " / ".join(str(v) for v in per2))})
st.dataframe(pd.DataFrame(wins_rows), hide_index=True, width="stretch")
st.caption("Siege = Zellen mit dem kleinsten RMSE von ln(Pegel-Dach/wahr) in dieser Zelle bei diesen festen k, Zähleinheit Zellen (32); die Zählung gilt als ± 3 Zellen. Dass die Schwelle bei k = 5 % einen kleineren RMSE hat als die Blöcke, "
           "kommt (bei der Pareto-Verteilung) von der fünffachen Zahl benutzter Werte: mit k = 50 liegt die Schwelle dort über den Blöcken (Block gegen Schwelle führt „Überschreitungen und die GPD“).")

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Wie viel bleibt unsicher?")
st.caption(T.CAPTION_EXP3)
st.markdown("**Streuband des Pegels, zitiert aus „Wiederkehrwert“** (GEV(μ = 100, σ = 20, ξ = 0,2), 90-%-Band von x̂/x_T, 2 000 Läufe je Zelle)")
band_rows = [{"m (Jahre)": mm, **{f"T = {tt}": f"{de(lo, 2)} bis {de(hi, 2)} (Median {de(med, 2)})" for tt in (10, 100, 1000) for lo, hi, med in [V.BAND_STUECK3[(mm, tt)]]}} for mm in (20, 50, 100, 200, 500)]
st.dataframe(pd.DataFrame(band_rows), hide_index=True, width="stretch")
st.markdown("**Puffer jenseits der Daten**")
m1c, m2c, m3c = st.columns(3)
m1c.metric("n·p (Beobachtungen oberhalb des Ziels)", de(a3.np_product, 2), delta=f"n = {thousands(a3.n)}, p = {p_text(a3.p)}", delta_color="off")
m2c.metric("P(q > 2p), empirisches Quantil (exakt)", de(a3.p_gt2, 3), delta="Beta-Verteilung, verteilungsfrei", delta_color="off")
m3c.metric("Erwartete Überschreitung E[q]/p", de(a3.e_q_over_p, 2), delta=f"j = {a3.j} von n = {thousands(a3.n)}", delta_color="off")
st.plotly_chart(build_puffer(a3), width="stretch", key="puffer_chart")
rows3 = []
for mt in E.PUFFER_METHODS:
    r = a3.rows[mt]
    rows3.append({"Verfahren": E.PUFFER_LABELS[mt], "P(q ≤ p)": de(r["hit"], 3), "P(q > 2p)": de(r["gt2"], 3), "Median q/p": de(r["medq"], 2), "Median Puffer/wahr": de(r["medr"], 2), "Aufschlag f90": de(r["f90"], 2),
                  "Gesamtpuffer (Median × f90)": de(r["tot"], 2), "Fehlschläge": r["fail"]})
st.dataframe(pd.DataFrame(rows3), hide_index=True, width="stretch")
st.caption(f"{S.DIST_LABELS[a3.case]}, 1 000 Läufe je Zelle (eingefroren), wahrer Puffer {de(a3.btrue, 3)}. f90 ist der Faktor auf den eigenen Schätzer, mit dem der Puffer in 90 % der Läufe hält; der Gesamtpuffer "
           "(Median Puffer/wahr × f90) ist der Größenvergleich: ein kleinerer Aufschlag kann ein höherer Grundwert sein.")
if a3.np_product < 1 and E.emp_index(a3.n, a3.p) == a3.n:
    f90x = B.f90_of_maximum(a3.case, a3.n, a3.p)
    st.info(f"Bei n·p < 1 ist das empirische Quantil das Maximum: sein Aufschlag f90 ist exakt aus F^n berechenbar, {de(f90x, 3)} (Aufschlag {de(100 * (f90x - 1), 1)} %); die 1 000 Läufe liegen im Band um diesen Wert.")
with st.expander("Kurze Reihen von Blockmaxima: sechs Monatsspitzen reichen nicht"):
    xi_short = st.radio("ξ der GEV(0, 1, ξ)", (0.0, 0.2), horizontal=True, key="xi_short")
    short_rows = []
    for mm in (6, 12, 24, 50):
        rows_s = V.short_series_rows(xi_short, mm)
        for pp in REF.puffer()["short_ps"]:
            for nm, lab in (("max", "Maximum der Blöcke"), ("gev_ml", "GEV-ML"), ("gumbel", "Gumbel-ML (ξ = 0 fest)"), ("gev_lmom", "GEV-L-Momente")):
                r = rows_s[f"{nm}|{pp:.6f}"]
                short_rows.append({"Blöcke m": mm, "Ziel p je Block": "1/12" if pp > 0.05 else "1/120", "Verfahren": lab, "P(q > 2p)": de(r["gt2"], 3), "Median q/p": de(r["medq"], 2), "Aufschlag 90 % (σ)": de(r["aufschlag90"], 2)})
    st.dataframe(pd.DataFrame(short_rows), hide_index=True, width="stretch")
    st.caption("Das Maximum der m Blöcke hat q ~ Beta(1, m), also P(q > 2p) = (1 − 2p)^m: bei m = 6 und p = 1/12 genau (5/6)^6 = 0,335; die Läufe (1 000) liegen im Band darum. Der Aufschlag ist additiv in Einheiten der Streuung σ.")

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Was sieht ein CVaR vom Tail?")
st.caption(T.CAPTION_EXP4)
q1, q2, q3 = st.columns(3)
q1.metric("Exakter CVaR", de(a4.exact, 3), delta="aus der Verteilung", delta_color="off")
q2.metric("Median empirisch/exakt (eingefroren)", de(a4.frozen["median"], 3), delta=f"{thousands(a4.frozen_runs)} Läufe", delta_color="off")
q3.metric("Kein Szenario über dem Quantil (exakt α^n)", f"{pct(a4.none_above)} %", delta=f"(1 − α)·n = {de((1 - a4.alpha) * a4.n, 2)}", delta_color="off")
st.plotly_chart(build_cvar(a4), width="stretch", key="cvar_chart")
frozen_rows = []
cv = REF.cvar()
for a in (0.9, 0.95, 0.99, 0.999):
    r = (cv["emp_r20000"] if a4.n == 600 else cv["emp_r2000"])[f"{a4.dist}|{a4.n}"][str(a)]
    g = cv["gpd_r2000"].get(f"{a4.dist}|{a4.n}", {}).get(str(a))
    frozen_rows.append({"α": de(a, 3), "Median empirisch/exakt": de(r["median"], 3), "Anteil der Läufe unter 1": f"{pct(r['below_one'])} %", "mit GPD an die oberen 5 % (Median)": de(g["median"], 3) if g else "nicht gerechnet",
                        "GPD ungültig (ξ̂ ≥ 1)": str(g["invalid"]) if g else "-"})
st.dataframe(pd.DataFrame(frozen_rows), hide_index=True, width="stretch")
st.caption(f"{thousands(a4.frozen_runs)} Läufe je Zelle (eingefroren); das Mittel des Verhältnisses wird nicht geführt (bei der Pareto-Verteilung α = 2 hat die Varianz keinen endlichen Wert). Die Live-Verteilung oben nutzt "
           f"{len(a4.ratios)} Läufe mit dem Zufalls-Seed der Seitenleiste.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.ASSUMPTIONS)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(T.SYMBOLS)
    for block in T.FORMULA_BLOCKS:
        st.markdown(block)
    st.markdown(T.FORMULAS_OUTRO)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Extremwerte: Blöcke, Schwelle, Pegel](https://sebastianhanisch.net/konzepte-extremwert.html)."
)
