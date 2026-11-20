"""Plotly-Figuren (alle Achsen fest, `fixedrange`; Beschriftungen als Annotationen)."""

import math

import numpy as np
import plotly.graph_objects as go

import vwr_constants as C
import vwr_estimators as E
import vwr_evaluation as V
import vwr_scenario as S

_LAYOUT = dict(margin=dict(l=60, r=20, t=40, b=60), height=360, legend=dict(orientation="h", y=-0.25))


def _fixed(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_reihe(a: V.Lage) -> go.Figure:
    """Anteil der Reihen ohne Beobachtung auf der Höhe des T-Jahres-Pegels gegen die Reihenlänge m (exakt); die gewählte Länge ist markiert."""
    s = a.s
    ms = np.arange(1, max(201, s.m1 + 20))
    y = [100.0 * float(V.not_reached(int(m), s.T1)) for m in ms]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ms, y=y, mode="lines", line=dict(color=C.COLOR_LINE, width=3), name=f"T = {s.T1} Jahre", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[s.m1], y=[100.0 * a.not_reached], mode="markers", marker=dict(size=14, color=C.COLOR_BAD, symbol="diamond"), name="gewählte Reihe", hoverinfo="skip"))
    fig.add_hline(y=50, line=dict(color="gray", dash="dot"))
    fig.add_annotation(x=0.99, xref="paper", y=50, yref="y", text="50 %: Schwelle des Hinweises", showarrow=False, yshift=10, xanchor="right", font=dict(size=11, color="gray"))
    fig.add_annotation(x=s.m1, y=100.0 * a.not_reached, text=f"{V.de(100.0 * a.not_reached, 1)} %", showarrow=True, ax=40, ay=-30, font=dict(size=12))
    fig.update_layout(title="Anteil der Reihen, die den Pegel nicht erreichen (exakt)", xaxis_title="Reihenlänge m (Jahre)", yaxis_title="Anteil in %", yaxis=dict(range=[0, 102]), **_LAYOUT)
    return _fixed(fig)


def build_levels(live: V.Live) -> go.Figure:
    """Pegel-Dach/wahr je Verfahren über die Läufe: Median und 10- bis 90-%-Spanne (log-Achse), die erste Reihe als Raute; 1 ist der wahre Pegel."""
    methods = list(live.errors)
    fig = go.Figure()
    for mt in methods:
        sm = V.live_summary(live, mt)
        if sm["n"] == 0:
            continue
        fig.add_trace(go.Scatter(x=[sm["q10"], sm["q90"]], y=[E.METHOD_SHORT[mt]] * 2, mode="lines", line=dict(color=C.COLOR_OTHER, width=8), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[sm["median_ratio"]], y=[E.METHOD_SHORT[mt]], mode="markers", marker=dict(size=11, color=C.COLOR_GOOD), showlegend=False, hoverinfo="skip"))
        f = live.first.get(mt)
        if f is not None and math.isfinite(f) and f > 0:
            fig.add_trace(go.Scatter(x=[f / live.truth], y=[E.METHOD_SHORT[mt]], mode="markers", marker=dict(size=10, color=C.COLOR_BAD, symbol="diamond"), showlegend=False, hoverinfo="skip"))
    fig.add_vline(x=1.0, line=dict(color=C.COLOR_TRUTH, dash="dash"))
    fig.add_annotation(x=0, xref="paper", y=1.0, yref="paper", text="Punkt: Median, Balken: 10 bis 90 %, Raute: erste Reihe, gestrichelt: wahrer Pegel", showarrow=False, yshift=14, xanchor="left", font=dict(size=11))
    fig.update_layout(title=f"Pegel-Dach / wahr, {live.R} Läufe", xaxis=dict(type="log", title="Pegel-Dach / wahrer Pegel (log-Achse)", range=[-1.2, 1.2]), yaxis=dict(autorange="reversed"), **{**_LAYOUT, "height": 380})
    return _fixed(fig)


_WIN_COLORS = {"gumbel": "#B5543C", "gpd_ml5": "#2F6B65", "gpd_pwm50": "#6BA39B", "gpd_ml50": "#9DC4BE", "hill50": "#D9A441", "hill5": "#E8C676", "gev_ml": "#4c78a8", "gev_lmom": "#8EAFD0"}


def build_matrix(seed_index: int = 0, selected=None) -> go.Figure:
    """Die 32 Zellen der Matrix: je Zelle das Verfahren mit dem kleinsten RMSE (eingefroren, 400 Läufe); die gewählte Zelle ist umrandet."""
    import vwr_reference as REF
    cols = [(m, q) for q in V.TARGETS for m in V.LENGTHS]
    xlab = [f"{V.TARGET_YEARS[q]} J., m = {m}" for m, q in cols]
    fig = go.Figure()
    shown = set()
    for yi, d in enumerate(S.MATRIX_DISTS):
        for xi_, (m, q) in enumerate(cols):
            w = V.cell_winner(REF.matrix_cell(d, m, q, seed_index))[1]
            fig.add_trace(go.Scatter(x=[xi_], y=[yi], mode="markers", marker=dict(size=34, symbol="square", color=_WIN_COLORS[w]), name=E.METHOD_SHORT[w], legendgroup=w,
                                     showlegend=w not in shown, hoverinfo="skip"))
            shown.add(w)
            fig.add_annotation(x=xi_, y=yi, text=E.METHOD_SHORT[w].split(" ")[0][:6], showarrow=False, font=dict(size=9, color="white"))
    if selected is not None:
        d, m, q = selected
        fig.add_shape(type="rect", x0=cols.index((m, q)) - 0.5, x1=cols.index((m, q)) + 0.5, y0=S.MATRIX_DISTS.index(d) - 0.5, y1=S.MATRIX_DISTS.index(d) + 0.5, line=dict(color=C.COLOR_TRUTH, width=3))
    fig.update_layout(title="Sieger je Zelle (kleinster RMSE von ln; 32 Zellen, 400 Läufe)", xaxis=dict(tickmode="array", tickvals=list(range(8)), ticktext=xlab, tickangle=-30, range=[-0.6, 7.6]),
                      yaxis=dict(tickmode="array", tickvals=list(range(4)), ticktext=[S.DIST_SHORT[d] for d in S.MATRIX_DISTS], autorange="reversed", range=[3.6, -0.6]), **{**_LAYOUT, "height": 380, "legend": dict(orientation="h", y=-0.4)})
    return _fixed(fig)


def build_puffer(a: V.Puffer) -> go.Figure:
    """P(q > 2p) je Verfahren (eingefroren, 1 000 Läufe) und der exakte Wert des empirischen Quantils (Beta, Raute)."""
    names = [E.PUFFER_LABELS[mt] for mt in E.PUFFER_METHODS]
    vals = [a.rows[mt]["gt2"] for mt in E.PUFFER_METHODS]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=[C.COLOR_BAD if mt in ("normal", "lognormal") else C.COLOR_LINE for mt in E.PUFFER_METHODS], hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[names[0]], y=[a.p_gt2], mode="markers", marker=dict(size=14, color=C.COLOR_TRUTH, symbol="diamond"), name="exakt (Beta)", hoverinfo="skip"))
    fig.update_layout(title="Anteil der Läufe, in denen der Puffer das Ziel um mehr als das Doppelte überschreitet", yaxis=dict(range=[0, 1.05], title="P(q > 2p)"), xaxis=dict(tickangle=-25), **{**_LAYOUT, "height": 400})
    return _fixed(fig)


def build_cvar(a: V.CvarAnalysis) -> go.Figure:
    """Verteilung des Verhältnisses empirischer/exakter CVaR über die Live-Läufe; 1 ist der exakte Wert, die Raute der eingefrorene Median."""
    r = a.ratios[np.isfinite(a.ratios)]
    fig = go.Figure(go.Histogram(x=r, nbinsx=30, marker_color=C.COLOR_LINE, hoverinfo="skip"))
    fig.add_vline(x=1.0, line=dict(color=C.COLOR_TRUTH, dash="dash"))
    fig.add_vline(x=a.frozen["median"], line=dict(color=C.COLOR_BAD, width=3))
    fig.add_annotation(x=a.frozen["median"], y=1.0, yref="paper", text=f"Median (eingefroren) {V.de(a.frozen['median'], 3)}", showarrow=False, xanchor="right", yshift=-10, font=dict(size=11, color=C.COLOR_BAD))
    fig.update_layout(title=f"CVaR aus {a.n} Szenarien / exakter CVaR, {len(r)} Läufe", xaxis_title="Verhältnis (1 = exakt)", yaxis_title="Läufe", **_LAYOUT)
    return _fixed(fig)
