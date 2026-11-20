"""Texte der App (Einleitung, Erklärungen, Grenzen, Formeln). Zahlen, die aus Messungen stammen, stehen hier nicht: die App holt sie aus den eingefrorenen Dateien oder rechnet sie exakt."""

INTRO = """
**Welches Verfahren wann? Wegweiser für den Pegel am Rand.** Jede Demo der Linie hat eine Frage mit einem Verfahren beantwortet: Blöcke, Schwelle, Hill, Intervall, Cluster, Trend. Diese Demo stellt sie
nebeneinander und zeigt, **wo jedes aufhört**: Es gibt kein Verfahren, das durchgehend gewinnt (gezählt in 32 Zellen aus 4 Verteilungen × 4 Reihenlängen × 2 Zielen), und was bleibt, ist die
**Unsicherheit des Pegels**, bis hin zu Zielen, die jenseits der Daten liegen. Die Demo rechnet fast nichts Neues: die Zahlen der anderen Demos sind mit der Demo zitiert, die sie führt; selbst gemessen sind nur die
Matrix, die Einordnung des CVaR und der Puffer jenseits der Daten.

Alle Verteilungen sind ausgedacht; über den Nutzen in Logistik, Lager oder Hafen sagt diese Seite nichts. Weiter unten: vier **🔬 Experimente**, **🚧 Wo die Annahmen enden** und der Expander
**📐 Mathematische Formulierung**.
"""

HOW_IT_WORKS = """
1. **Meine Lage:** Reihenlänge, Wiederkehrperiode, Überschreitungen k, Abhängigkeit (θ) und Trend einstellen; die Seite meldet die **zutreffenden Zeilen der Entscheidungstabelle** und, exakt, wie viele
   Reihen den Pegel gar nicht erreichen.
2. **Alle Verfahren auf dieselbe Reihe:** acht Verfahren (GEV-ML, GEV-L-Momente, Gumbel mit ξ = 0, GPD-ML, GPD-PWM, Hill-Weissman) schätzen denselben Pegel; die Demo kennt den wahren Pegel und misst den
   Fehler als ln(Pegel-Dach/wahr). Die Matrix aus 32 Zellen ist eingefroren.
3. **Wie viel bleibt unsicher?** Das Streuband des Pegels (zitiert aus „Wiederkehrwert“) und der **Puffer jenseits der Daten**: wenn das Ziel seltener ist als eine Beobachtung je Reihe (n·p < 1), hält in den gemessenen Fällen kein Rand-Verfahren das Ziel in allen vier Verteilungen (GPD und GEV in 36 bis 60 % der Läufe, Hill-Weissman nur mit Überhöhung).
4. **Was sieht ein CVaR vom Tail?** Ein CVaR aus n Szenarien liegt bei n = 60 und 600 im Median unter dem echten, und der Abstand wächst mit dem Niveau und dem Gewicht des Rands; bei n = 6 000 und α ≤ 0,95 beträgt er höchstens 2 %.
"""

CAPTION_EXP1 = ("Die Tabelle sagt, wann man ein Verfahren **nicht** nehmen sollte, nicht, welches das richtige ist. Die Schwellen der Hinweise (50 % der Reihen, 20 Maxima, k = 100, 50 Cluster) sind eigene Festlegungen "
                "aus den Zahlen der Tabelle, keine Literaturregeln. Ein Jahr sind hier 100 Beobachtungen („Wiederkehrwert“ zählt 365).")
CAPTION_EXP2 = ("Eine Reihe aus m Jahren zu je 100 Beobachtungen, alle Verfahren sehen dieselbe Reihe; der wahre Pegel ist exakt aus F^100 (bzw. beim Vehikel B aus der Gewichtstabelle). Fehler = ln(Pegel-Dach/wahr): 0 heißt richtig, "
                "−0,69 die Hälfte. Die Läufe hier sind live (Zufalls-Seed der Seitenleiste); die Matrix mit 400 Läufen je Zelle und die Zählung der Siege sind eingefroren.")
CAPTION_EXP3 = ("Streuband und Intervall-Überdeckung sind aus „Wiederkehrwert“ zitiert (kein eigenes Streuband-Experiment). Der Puffer ist das Quantil zur Überschreitungswahrscheinlichkeit p je Beobachtung; "
                "n·p ist die erwartete Zahl der Beobachtungen oberhalb des Ziels. q ist die wahre Überschreitungswahrscheinlichkeit des geschätzten Puffers, exakt aus F.")
CAPTION_EXP4 = ("Der CVaR (Erwartungsfehlbetrag) aus n gleich wahrscheinlichen Szenarien in der Form der LPs der `cvar-demo` (Rockafellar und Uryasev), verglichen mit dem exakten CVaR der ausgedachten Verteilung. "
                "Die Verteilung der `cvar-demo` ist hier nicht nachgebaut. CVaR bleibt dort Optimierungsziel; hier nur die Einordnung.")

ASSUMPTIONS = """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ausgedachte Verteilungen** (vier in der Matrix, fünf beim CVaR, vier beim Puffer) | Die Tabelle sagt, wann ein Verfahren in diesen Modellen versagt, nicht, welches in einem Betrieb das richtige ist. Eine reale Reihe von Lasten, Wartezeiten oder Umschlag, an der die Verfahren geprüft wären, gibt es in den gelesenen Quellen nicht. | - |
| **Feste k** (50 und 5 % der Beobachtungen) und **eine Blocklänge** (100 Beobachtungen) | Andere k und andere Blocklängen können die Reihenfolge ändern; die Schwellenwahl selbst steht nicht in der Matrix. | „Schwellenwahl“, „Überschreitungen und die GPD“ (Block gegen Schwelle) |
| **Unabhängige Beobachtungen, kein Trend** | Bei Abhängigkeit und Trend gelten die Zeilen von „Spitzen-Cluster“ und „Trend“ (bei Abhängigkeit hielt in „Spitzen-Cluster“ kein 95-%-Intervall seine Überdeckung, auch mit exaktem θ nicht, und mehr Cluster halfen kaum); Abhängigkeit zusammen mit Trend ist nicht gemessen. | „Spitzen-Cluster“, „Trend“ |
| **Zählung der Siege** | Ein Sieg heißt nur kleinster RMSE in einer Zelle; Unterschiede nahe am RMSE sind Rauschen (Standardfehler eines RMSE bei 400 Läufen grob 4 % des Wertes), die Zählung gilt als ± 3 Zellen. | - |
| **Keine Kostenfunktion** beim Puffer | Gemessen sind Überschreitungswahrscheinlichkeit und Größe des Puffers, nicht Kosten; welche Zeile „besser“ ist, bleibt offen. | - |
| **Keine Abhängigkeit und kein Trend beim Puffer**, keine Summe über die Wiederbeschaffungszeit | Die Aussagen über den Puffer gelten für einzelne Beobachtungen je Zeiteinheit; die Summe über eine Wiederbeschaffungszeit ist nicht gemessen. | „Spitzen-Cluster“, „Trend“ |
| **Der CVaR der `cvar-demo`** | Gemessen sind fünf ausgedachte Verteilungen, nicht die 600 Szenarien der `cvar-demo`; dass „der echte Tail höher liegt“, stützt diese Messung nur für diese fünf Verteilungen im Median. | `cvar-demo` |
| **Mehrdimensionale Spitzen** | nicht gerechnet. | `randabhaengigkeit-demo` |
| **Quellen** | Das Deich-Beispiel (Lehrbuch) und die Abstract-Belege (Zhu u. a. 2017, Lee u. a. 2022) sind nur als „Lehrbuch/Abstract“ geführt; Belzile/Davison und Slater u. a. nur nach Zusammenfassung, ungeprüft. | - |
"""

SYMBOLS = """
**Lesehilfe.** Gegeben sind eine Reihe von Beobachtungen (oder ihre Jahresmaxima) und ein Ziel; gesucht ist der **Pegel** (Wiederkehrwert), und gefragt wird, wie weit ein Verfahren ihn verfehlt.

| Symbol | Bedeutung |
|---|---|
| $m$, $n_y$ | Zahl der Jahre (Blöcke) und Beobachtungen je Jahr (hier 100); $N = m\\,n_y$ |
| $T$, $q = 1/T$ | Wiederkehrperiode in Jahren und Überschreitungswahrscheinlichkeit je Jahr |
| $p$ | Überschreitungswahrscheinlichkeit je Beobachtung; $p = 1 - (1-q)^{1/n_y}$ |
| $\\xi$ | Extremwertindex: $\\xi > 0$ Fréchet-Typ (schwerer Rand), $\\xi = 0$ Gumbel, $\\xi < 0$ Weibull-Typ; `scipy.stats.genextreme` hat $c = -\\xi$, `genpareto` $c = \\xi$ |
| $\\theta$ | Extremalindex (Gruppierung der Spitzen), **nicht** mit $\\xi$ verwandt |
| $k$ | Zahl der Überschreitungen über der Schwelle $u$ |
| $\\hat x$, $x$ | geschätzter und wahrer Pegel; Fehler $\\ln(\\hat x / x)$ |
| $q_{\\text{wahr}}$ | wahre Überschreitungswahrscheinlichkeit des geschätzten Puffers |
| $\\alpha$, $n$ | Niveau des CVaR und Zahl der Szenarien |
"""

FORMULA_BLOCKS = [
    r"""**Reihe gegen Wiederkehrperiode (exakt).** Bei unabhängigen Jahresmaxima erreicht keines von $m$ Jahren den $T$-Jahres-Pegel mit Wahrscheinlichkeit""",
    r"""$$\left(1 - \frac{1}{T}\right)^{m}, \qquad \text{bei } m = T \to 1/e \approx 0{,}368 .$$""",
    r"""**Wahrer Pegel je Block aus der Verteilung je Beobachtung.** Ein Block aus $n_y$ Beobachtungen hat die Verteilungsfunktion $F^{n_y}$; der wahre Pegel zu $q$ je Block ist die Lösung von""",
    r"""$$F^{n_y}(x) = 1 - q \quad\Longleftrightarrow\quad 1 - F(x) = p = 1 - (1-q)^{1/n_y} .$$""",
    r"""**Verfahren.** GEV-Pegel zu $q$ je Block (für $\xi \neq 0$; $\xi = 0$ als Grenzwert $\mu - \sigma \ln(-\ln(1-q))$):""",
    r"""$$\hat x = \mu + \frac{\sigma}{\xi}\Big(\big(-\ln(1-q)\big)^{-\xi} - 1\Big) .$$""",
    r"""GPD-Pegel an die $k$ größten von $N$ Werten (Schwelle $u$ = der $(k+1)$-größte Wert, $\zeta = k/N$), für $\xi \neq 0$:""",
    r"""$$\hat x = u + \frac{\sigma}{\xi}\Big(\big(\zeta / p\big)^{\xi} - 1\Big) .$$""",
    r"""Hill-Weissman-Pegel mit $H_k = \frac{1}{k}\sum_{i=1}^{k} \ln\big(X_{(i)} / X_{(k+1)}\big)$ (verlangt positive Werte):""",
    r"""$$\hat x = X_{(k+1)} \Big(\frac{k}{N p}\Big)^{H_k} .$$""",
    r"""**Fehlermaß.** Je Zelle der Matrix der RMSE von $\ln(\hat x / x)$ über die Läufe; ein Verfahren „gewinnt“ die Zelle, wenn dieser Wert am kleinsten ist.""",
    r"""**Puffer jenseits der Daten (exakt).** Das empirische Quantil $X_{(j)}$ mit $j = \lceil (1-p)(n+1) \rceil$ (höchstens $n$) hat für jede stetige Verteilung die wahre Überschreitungswahrscheinlichkeit""",
    r"""$$q_{\text{wahr}} = 1 - F(X_{(j)}) \sim \mathrm{Beta}(n-j+1,\, j), \qquad P(q_{\text{wahr}} \le x) = P\big(\mathrm{Bin}(n, x) \ge n-j+1\big) .$$""",
    r"""Für das Maximum ($j = n$, also $n p < 1$) folgt $E[q_{\text{wahr}}] = 1/(n+1)$ und""",
    r"""$$P(q_{\text{wahr}} > 2p) = (1 - 2p)^{n} .$$""",
    r"""**CVaR aus Szenarien** (Rockafellar und Uryasev): mit $\zeta$ als dem $\lceil \alpha n \rceil$-kleinsten Szenario""",
    r"""$$\widehat{\mathrm{CVaR}}_\alpha = \zeta + \frac{1}{(1-\alpha)\,n} \sum_{i=1}^{n} \max(x_i - \zeta,\, 0) .$$""",
    r"""Ist $(1-\alpha)\,n < 1$, ist das das Maximum der Szenarien; mit Wahrscheinlichkeit $\alpha^{n}$ liegt kein Szenario über dem $\alpha$-Quantil. Exakt zum Vergleich: Normal $\varphi(z_\alpha)/(1-\alpha)$,
Exponential $1 - \ln(1-\alpha)$, Pareto-Verteilung $\alpha = 2$: $2\,(1-\alpha)^{-1/2}$, Lognormal $\sigma = 1$ und Student-$t$ mit $\nu = 3$ in geschlossener Form (`vwr_cvar.es_exact`).""",
]
FORMULAS_OUTRO = ("Dies ist ein **vollständiges** Modell für das, was die Seite rechnet: Verteilungen, Pegel, Fehlermaß, Puffer und CVaR sind definiert; die Anpassungen (ML, L-Momente, PWM) sind in den Demos „GEV an Blockmaxima“, „Überschreitungen und die GPD“ und „Pegel statt Index“ ausgeführt "
                  "und hier nur aufgerufen. Literatur nur als Lehrbuch (Coles 2001; Embrechts, Klüppelberg, Mikosch 1997); keine Neuheit.")
