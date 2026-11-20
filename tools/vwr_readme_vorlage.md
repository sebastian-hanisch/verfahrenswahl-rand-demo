# Welches Verfahren wann? Wegweiser für den Pegel am Rand – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-verfahrenswahl-rand-demo.streamlit.app/)** -->

Demo der Linie **„Extremwerte erklärt“** in der „Konzepte“-Reihe der Website „Sebastian Hanisch – Operations Research und Machine Learning“ (Bereich Statistik, Feld „Seltenes und Extremes“). Es ist die **Zusammenfluss-Demo**: Die Demos davor haben je eine Frage mit einem Verfahren beantwortet (Maximum von n, GEV an Blockmaxima, Wiederkehrwert und Intervalle, Überschreitungen und GPD, Schwellenwahl, Hill, Abhängigkeit, Trend); diese Demo stellt sie nebeneinander und sagt, **wo jedes aufhört**. Die Kante heißt `kontrast`: Es behebt kein Verfahren, es hebt die **stille Annahme „es gibt ein richtiges Verfahren“** auf. Muster im Bestand ist `verfahrenswahl-aenderung-demo` (ein Wegweiser für Kontrollkarten, anderer Inhalt).

Vehikel sind **ausgedachte** Verteilungen mit exakt bekanntem Rand (Exponential, Pareto-Verteilung α = 2, Normal, Lognormal σ = 1, für den CVaR dazu Student-t mit ν = 3, für den Puffer Normal(100, 20)) und das ausgedachte **Vehikel B** (eine Verweilzeit-Mischung mit 1 224 Komponenten, gemeinsame Tabelle der Linie). Die Demo kennt die Wahrheit und misst, wie weit ein Verfahren den exakten Pegel verfehlt; im Betrieb gibt es sie nicht. In den gelesenen Quellen gibt es keine reale Reihe von Lasten, Wartezeiten oder Umschlag, an der die Verfahren geprüft wären: **Über den Nutzen einer besseren oder schlechteren Pegel-Schätzung in Logistik, Lager oder Hafen sagt diese Demo nichts.** Das Deich-Beispiel der Lehrbücher und die Abstract-Belege (Zhu u. a. 2017, Lee u. a. 2022) stehen nur als „Lehrbuch/Abstract“ unter 🚧.

Zähleinheiten: **Läufe** (Wiederholungen eines ganzen Experiments), **Zellen** (Verteilung × Reihenlänge × Ziel), **Jahre** (Blöcke zu je 100 Beobachtungen; „Wiederkehrwert“ zählt 365 Tageswerte je Jahr, „der 100-Jahres-Pegel“ ist darum in beiden Demos nicht dieselbe Zahl), **Beobachtungen** n, **Überschreitungen** k, **Szenarien** n. **Pegel** heißt der geschätzte Wert zu einer Wiederkehrperiode; **die Wiederkehrperiode ist ein Mittel, keine Frist** (innerhalb von T Jahren wird der T-Jahres-Pegel nur mit 1 − (1 − 1/T)^T → 63,2 % mindestens einmal überschritten, bei T = 100 mit 63,4 %). Der Fehler eines Pegels steht als **ln(Pegel-Dach/wahr)** (0 heißt richtig, −0,69 die Hälfte); ‚wahr‘ ist exakt aus F bzw. F^n, nie aus GEV-Parametern (beim Vehikel B rechnet die Puffer-Tabelle q aus einer Interpolation der Überlebensfunktion; Test: relativer Fehler unter 1e-3). Der Extremwertindex ξ ist **nicht** der Extremalindex θ. Konvention: ξ > 0 Fréchet-Typ (schwerer Rand), ξ = 0 Gumbel, ξ < 0 Weibull-Typ; `scipy.stats.genextreme` hat c = −ξ, `scipy.stats.genpareto` c = ξ (nur in den Tests, die App rechnet mit numpy und der Standardbibliothek).

## Warum dieses Problem

Wer einen Pegel jenseits der Daten braucht, hat in den Demos davor sechs Verfahren kennengelernt, und jedes gewinnt dort, wo seine Voraussetzung stimmt. Hier wird **gezählt**: In einem Raster aus 4 Verteilungen × 4 Reihenlängen (10 bis 100 Jahre zu je 100 Beobachtungen) × 2 Zielen (100- und 1000-Jahres-Pegel) gewinnt Gumbel mit festem ξ = 0 in {{GUMBEL_BAND}} von 32 Zellen den kleinsten RMSE, unterschätzt aber in 15 bis 16 Zellen (Pareto-Verteilung und Lognormal, je nach Zufalls-Seed; die Lognormal-Zelle m = 10, 100-Jahres-Pegel liegt bei 89,5 bis 93,8 %) in etwa 90 % der Läufe oder mehr; die Schwelle (GPD-ML an 5 % der Beobachtungen) gewinnt 15 Zellen ohne Hill, benutzt dabei aber fünfmal so viele Werte wie die Blöcke; Hill gewinnt unter den vier gemessenen Verteilungen nur bei der Pareto-Verteilung (dem reinen Potenzgesetz, das er voraussetzt) und liegt bei den anderen im Median beim 1,66- bis 1,69-Fachen des wahren 100-Jahres-Pegels. **Was bleibt, ist die Unsicherheit:** aus 50 Jahren Daten liegt der 100-Jahres-Pegel bei GEV(μ = 100, σ = 20, ξ = 0,2) zu 90 % zwischen dem 0,73- und 1,49-Fachen des wahren („Wiederkehrwert“), in 60,5 % der 50-Jahres-Reihen liegt nicht einmal eine Beobachtung auf dieser Höhe (exakt), und wenn das Ziel seltener ist als eine Beobachtung je Reihe (n·p < 1), hält in den gemessenen Fällen kein Rand-Verfahren das Ziel in allen vier Verteilungen: GPD und GEV in 36 bis 60 % der Läufe, Hill-Weissman nur um den Preis eines 1,1- bis 1,7-fachen Puffers und bei der Pareto-Verteilung in 50 % (n·p = 0,2).

**Kernfragen:** (1) Was sagt die Tabelle zu meiner Lage? (2) Welches Verfahren liegt wo daneben? (3) Wie viel bleibt unsicher, auch beim Puffer? (4) Was sieht ein CVaR vom Tail? Die Tabelle sagt, wann man ein Verfahren **nicht** nehmen sollte, nicht, welches das richtige ist; sie gilt für unabhängige Beobachtungen aus den genannten Verteilungen (bei Abhängigkeit „Spitzen-Cluster“, bei Trend „Trend“), bei fester eigener Wahl der k (50 und 5 % der Beobachtungen).

## Befunde und Korrekturen (gegenüber Plan und Vormessung)

Die Vormessungen liefen mit `numpy.default_rng` und scipy; hier laufen Matrix, CVaR-Raster und Puffer mit der SplitMix64-Vorlage (numpy-Vektorform, Ganzzahlen und Gleichverteilung bitgleich zur Vorlage) und **anderen Zufalls-Seeds** (Basen 910 000 / 920 000 / 930 000 für die Matrix, 940 000 Kopplung, 950 000 CVaR, 960 000 Puffer) neu; die Anpassungen sind die der Demos „GEV an Blockmaxima“, „Überschreitungen und die GPD“ und „Pegel statt Index“, ohne scipy. Was sich verschoben hat (alt → neu, Zähleinheit Läufe R bzw. Zellen):

* **Zählung der Siege** (32 Zellen, kleinster RMSE, drei Zufalls-Seeds): Gumbel 15 / 12 / 14 → **12 / 13 / 14**; GPD-ML mit k = 5 % ohne Hill 14 bis 15 → 15 / 15 / 15; GPD-PWM ohne Hill 2 bis 5 → 5 / 4 / 2; Hill 8 → 8 (2 Zellen mit k = 50, 6 mit k = 5 %, alle in den acht Zellen der Pareto-Verteilung). Die **12 festen Zellen** (alle acht Exponential-Zellen und die vier Normal-Zellen mit m = 10 und 20) gewinnt Gumbel in allen drei Zufalls-Seeds; die drei Lognormal-Zellen (100-Jahres-Pegel m = 10 und 20, 1000-Jahres-Pegel m = 10) sind Beinahe-Gleichstände und wechseln den Sieger (GPD-PWM, Gumbel, L-Momente). Die Zählung gilt als ± 3 Zellen.
* **Matrix-Zeile m = 50, 100-Jahres-Pegel** (RMSE von ln, R = 400): Pareto-Verteilung GEV-ML 0,45 → 0,48, GPD-ML (k = 50) 0,52 → 0,53, L-Momente 0,44 → 0,43, GPD-PWM 0,45 → 0,44, Hill k = 50 0,32 → 0,33, Hill k = 5 % 0,18 → 0,20; Exponential: Gumbel unterschätzt in 56,0 → 54,5 % der Läufe; Lognormal: Gumbel Median 0,73 → 0,74. Alle Verschiebungen im Rahmen des Standardfehlers eines RMSE bei R = 400 (grob 4 % des Wertes) bis auf die Pareto-Verteilung (GEV-ML 0,45 → 0,48: rund 1,5 Standardfehler).
* **Gumbel bei der Pareto-Verteilung** (Median Pegel-Dach/wahr, 100-Jahres-Pegel): 0,41 bis 0,46 → 0,39 bis 0,47 (m = 10 bis 100). **Normal:** „Gumbel überschätzt in 92 bis 100 % der Läufe bei m ≥ 50“ → 88,8 % (m = 50) und 97,8 % (m = 100); bei m = 20 sind es 75,5 %. **Hill bei ξ = 0** (Median, m = 50, k = 5 %): 1,65 und 2,3 bis 2,5 → 1,66 (100-Jahres-Pegel) und 2,38 bis 2,56 (1000-Jahres-Pegel).
* **Kurze Reihe m = 10, Median des Betrags |ln(Pegel-Dach/wahr)|** (Exponential / Pareto-Verteilung / Normal / Lognormal): GEV-ML 0,23 / 0,91 / 0,17 / 0,56 → 0,25 / 0,83 / 0,15 / 0,53; L-Momente 0,17 / 0,54 / 0,13 / 0,38 → 0,18 / 0,59 / 0,11 / 0,34; Gumbel 0,10 / 0,91 / 0,07 / 0,37 → 0,10 / 0,96 / 0,07 / 0,37; GPD-ML 0,18 / 0,54 / 0,12 / 0,38 → 0,18 / 0,60 / 0,12 / 0,35; GPD-PWM 0,17 / 0,44 / 0,13 / 0,33 → 0,17 / 0,52 / 0,13 / 0,29; Hill 0,49 / 0,26 / 0,49 / 0,55 → 0,51 / 0,31 / 0,51 / 0,52. Die Rangfolge „ML schlechter als L-Momente“ bleibt in allen vier Verteilungen. Die ML klebt bei m = 10 in 1,8 bis 12,5 % der Läufe an der Grenze ξ = −1 (erster Zufalls-Seed; über alle drei Zufalls-Seeds 0,75 bis 12,5 %, je Verteilung anders; Plan: 2,8 bis 11 %).
* **Block gegen Schwelle in der Matrix-Zelle** (m = 50, b = 100, k = 50 = n/b; „Überschreitungen und die GPD“ führt die Messung): Verhältnis RMSE Schwelle/Block 1,00 / 1,16 / 1,00 / 1,04 (Plan, R = 400) und 1,02 / 1,09 / 1,04 / 1,03 (Nachzug, R = 2 000) → bei R = 400 hier 0,99 / 1,12 / 1,05 / 1,08, bei **R = 2 000 1,04 / 1,09 / 1,03 / 1,06**; ln-Streuung 0,99 / 1,08 / 1,02 / 1,02 → 1,01 / 1,08 / 1,01 / 1,06. Die Zelle der Pareto-Verteilung bleibt bei 1,09; bei Lognormal liegt der neue Wert 1,06 gegen 1,03 (Standardfehler 0,01 bis 0,02). ξ̂ bei Exponential (R = 2 000): GEV-ML Mittel −0,007, Standardabweichung 0,118 → −0,007 und 0,121; GPD-ML −0,067 und 0,168 → −0,059 und 0,171; GPD-PWM −0,035 und 0,166 → −0,026 und 0,169.
* **Korrektur U4: Der Aufschlag „122 %“ ist exakt 120,5 %.** Das empirische Quantil ist bei n·p < 1 das Maximum der Beobachtungen; sein 90-%-Aufschlag f90 folgt exakt aus F^n (das 10-%-Quantil des Maximums löst F(x)^n = 0,1): Vehikel B, n = 2 000, p = 10⁻⁴: **f90 = 2,205, Aufschlag 120,5 %**. Die Simulation mit 1 000 Läufen gibt 2,16 (Aufschlag 116 %), also im Rauschen um den exakten Wert; die Vormessung nannte 2,22.
* **Puffer, Vehikel B** (1 000 Läufe; Plan → neu): n·p = 0,2: P(q > 2p) empirisch 0,692 → 0,672 (exakt 0,670), GPD 0,253 → 0,220, GEV 0,349 → 0,318, Hill-Weissman 0,007 → 0,004; Gesamtpuffer empirisch 1,51 → 1,48, GPD 1,63 → 1,68, GEV 1,83 → 1,80, Hill 1,42 → 1,41 (der Befund „der kleinere Aufschlag der GPD ist ein höherer Grundwert“ bleibt bei n·p = 0,2); n·p = 0,5: empirisch 0,373 → 0,353 (exakt 0,368), GPD 0,098 → 0,094, GEV 0,195 → 0,181. Normal-Formel, Median q/p bei p = 10⁻⁴: 105 bis 124 → 105 bis 119 (n = 5 000 / 2 000 / 500: 104,9 / 107,6 / 118,8).
* **Sechs Monatsspitzen** (m = 6, p = 1/12): P(q > 2p) für das Maximum simuliert 0,340 / 0,348 (ξ = 0 / 0,2) → 0,305 / 0,352, exakt 0,335 (die 0,305 liegen 2,0 Standardfehler unter dem exakten Wert; bei dieser Zahl von Vergleichen im Rahmen des Zufalls, eine Deutung, nicht gesondert geprüft); Aufschlag 1,5 σ und 2,1 bis 2,2 σ → 1,44 σ und 2,07 σ. Gumbel-ML bei m = 24, p = 1/120, ξ = 0,2: 0,77 gegen 0,67 des Maximums → 0,78 gegen 0,65.
* **CVaR** (n = 600, 20 000 Läufe, Median; Plan → neu, α = 0,999): Normal 0,905 → 0,903, Exponential 0,854 → 0,855, Lognormal 0,694 → 0,699, Pareto-Verteilung 0,466 → 0,467, Student-t 0,635 → 0,632; α = 0,90: Pareto-Verteilung 0,962 → 0,961. Mit GPD an die oberen 5 % (R = 2 000, α = 0,999; Normal … Student-t): 0,937 / 0,909 / 0,873 / 0,758 / 0,799 → 0,934 / 0,917 / 0,888 / 0,725 / 0,800; ungültig (ξ̂ ≥ 1) bei der Pareto-Verteilung 58 → 54 von 2 000 Läufen.
* **Zitierte Zahlen, die eine Demo im Bau verschoben hat** (Eigentümer-Prinzip: hier mit Verweis übernommen, nicht neu hergeleitet): „Wiederkehrwert“, Zelle ξ = 0,2, m = 50, T = 100: Delta 86,9 → 89,1 %, Profil 95,2 → 95,1 %, Bootstrap 90,8 → 91,2 %; Streuband 0,74 bis 1,50 → 0,73 bis 1,49 (100er) und 0,57 bis 2,2 → 0,56 bis 2,21 (1000er); Delta-Spanne 70,4 bis 91,4 → 76,0 bis 91,9 %, Profil 93,3 bis 95,2 (12 Zellen) → 92,4 bis 95,9 % (13 Zellen), Bootstrap 84,6 bis 91,5 → 82,2 bis 91,2 %. „Überschreitungen und die GPD“: Streuverhältnis Schwelle/Block bei b = 20, k = n/b: 1,08 bis 1,30 → 1,09 bis 1,38; ML am Rand bei k = 10: 32,5 → 35,5 %. „Schwellenwahl“: sieben Regeln auf derselben Reihe, Median 2,66 → 2,76, über Faktor 2 in 80,0 → 72,3 %; Regel S 51,7 → 50,0 %. „Pegel statt Index“: Vehikel B, GEV frei 1,360 → 1,146 (300 Läufe; 1,265 bei 3 000), GPD k = 100 1,155 → 0,998 (1,116 bei 3 000): **die Mediane sind Einzelläufe, die mit R stark streuen, und stehen hier nur als Band mit R**; „in allen 1 000 Läufen“ bei verschobener Pareto-Verteilung (U2) → 19 888 von 20 000 (99,44 %). „Spitzen-Cluster“ (Überdeckung des 95-%-Intervalls in Läufen; „Spitzen-Cluster“ hat am 2026-11-20 einen Vorzeichenfehler in der Kovarianz von σ̂ und ξ̂ im Delta-Intervall der GPD behoben und führt die Korrektur): Schwelle ohne θ / mit exaktem θ (φ = 0 / 0,5 / 0,8) 96 / 80 / 50 und 96 / 91 / 81 % → 95,3 / 78,3 / 49,1 und 95,3 / 89,5 / 82,0 % (erste Fassung von „Spitzen-Cluster“) → **89,0 / 62,4 / 32,6 und 89,0 / 75,6 / 60,5 %**; am Vehikel A (Last 50 / 80 %, unabhängig / θ̂ / exaktes θ) 73,9 / 94,8 / 97,4 und 17,9 / 70,2 / 86,9 % → **35,4 / 59,1 / 61,7 und 4,9 / 22,9 / 25,6 %**; Block (Pegelskala) 88,1 / 88,2 / 84,5 → 88,9 / 86,1 / 84,3 % (hängt nicht am Delta-Intervall, unverändert). Die frühere Zeile dieser Demo „Block oder Schwelle mit θ-Korrektur“ ist damit nicht mehr gedeckt: mit exaktem θ deckt die Schwelle bei φ = 0,8 nur 60,5 %. „Trend“: Trendmodell gegen stationär 4 klar besser / 5 klar schlechter / 3 im Rauschen → 4 / 7 / 1.
* **Plan-Wortlaut, der nicht stimmt:** „ein zusätzlicher größter Wert hebt die GPD-Schwelle nie“ ist falsch herum: die Schwelle ist der (k+1)-größte Wert und kann durch einen neuen größten Wert nur steigen oder gleich bleiben (so im Test). Die Plan-Obergrenzen der Live-Läufe (R ≤ 100 bei der Pareto-Verteilung, R ≤ 10 sonst) entfallen: mit der numpy-Vektorform des Zufalls sind alle Verteilungen live bis R = 200 möglich.

## Modell

Eine Reihe besteht aus m Jahren zu je n_y = 100 Beobachtungen (N = 100 m), iid aus F. Das Ziel ist der Pegel zur Überschreitungswahrscheinlichkeit q je Jahr (q = 1/100 und 1/1000); je Beobachtung ist das p = 1 − (1 − q)^(1/n_y), der wahre Pegel löst F^100(x) = 1 − q (Orakel im Test, gegen `scipy.stats.<dist>.isf`). Formeln, Symbole und Konventionen stehen im 📐-Expander der App (ein vollständiges Modell für das, was die Seite rechnet; die Anpassungen selbst sind in den Demos „GEV an Blockmaxima“, „Überschreitungen und die GPD“ und „Pegel statt Index“ ausgeführt). Presets stehen in der Seitenleiste, die Kernfragen vor den Experimenten.

## Methodik

* **Matrix (Experiment 2):** acht Verfahren auf dieselben Reihen: GEV-ML (ξ frei, eigener Nelder-Mead), GEV-L-Momente (ξ̂ geklammert auf [−0,999; 0,99]) und Gumbel-ML (ξ = 0 fest) an die m Jahresmaxima; GPD-ML und GPD-PWM an die k = 50 größten Werte; GPD-ML und Hill-Weissman mit k = 5 % der Beobachtungen (50, 100, 250, 500 bei m = 10, 20, 50, 100; bei m = 10 sind k = 50 und k = 5 % dasselbe Verfahren); Hill-Weissman mit k = 50. **Die k sind feste eigene Festlegungen**, keine Schwellenwahl-Regel („Schwellenwahl“). R = 400 Läufe je Zelle, 32 Zellen, drei Zufalls-Seed-Basen; ein Verfahren „gewinnt“ eine Zelle mit dem kleinsten RMSE von ln(Pegel-Dach/wahr) unter den Verfahren mit mindestens 90 % gültigen Läufen. Die Anpassungen sind wörtliche Abschriften aus den Nachbar-Demos (Demos „GEV an Blockmaxima“, „Überschreitungen und die GPD“, „Pegel statt Index“), geprüft von `tests/test_copies.py`.
* **Streuband und Intervalle (Experiment 3)** sind aus „Wiederkehrwert“ zitiert; die zitierten Zahlen stehen mit Demo und Wortlaut in `vwr_table.ZITATE`, und `tests/test_claims.py` sucht ihren Wortlaut in den READMEs der Nachbar-Demos (übersprungen, wenn der Ordner fehlt).
* **Puffer jenseits der Daten:** der Puffer b ist das Quantil zur Überschreitungswahrscheinlichkeit p je Beobachtung; sechs Verfahren (empirisches Quantil, Normal-Formel, Lognormal-Formel, GPD und Hill-Weissman an die größten 5 %, GEV an Blockmaxima zu 100); gemessen wird die wahre Überschreitungswahrscheinlichkeit q des geschätzten Puffers, exakt aus F (beim Vehikel B aus einer Interpolation, Fehler unter 1e-3). Das empirische Quantil ist **exakt berechenbar**: q ~ Beta(n−j+1, j) für jede stetige Verteilung. R = 1 000 Läufe je Zelle (4 Fälle × 3 n × 3 p), dazu kurze Blockreihen (m = 6 bis 50, ξ = 0 und 0,2).
* **CVaR-Einordnung:** empirischer CVaR aus n gleich wahrscheinlichen Szenarien in der Form von Rockafellar und Uryasev (die Form der LPs der `cvar-demo`) gegen den exakten (geschlossene Formeln), Median des Verhältnisses; R = 20 000 Läufe bei n = 600, R = 2 000 sonst. Die Szenarien der `cvar-demo` sind hier nicht nachgebaut.
* **Hinweise (`vwr_evaluation.hinweise`):** fünf Codes mit eigenen Schwellen (Anteil der Reihen ohne Pegelhöhe ≥ 50 %, m ≤ 20 Maxima, k < 100, θ·k < 50 Cluster, Jahre unter denen für 80 % Trennschärfe des Trendtests). **Die Schwellen sind eigene Festlegungen aus den Zahlen der Tabelle, keine Literaturregeln;** die Schwelle 50 % der Extrapolation ist an keiner Messung begründet (bei m = T = 100 erreichen 36,6 % der Reihen den Pegel nicht, und es gibt keine Warnung); ebenso ist die Schwelle θ·k < 50 Cluster nicht als Grenze belegt: in „Spitzen-Cluster“ deckte das Intervall mit exaktem θ in 36 Zellen höchstens 74,2 %, und mit mehr Clustern stieg die Abdeckung kaum.
* **Zufall:** nur SplitMix64 (`vwr_rng.py` ist die wörtliche Kopie der Vorlage des Portfolios, `vwr_vecrng.py` rechnet dieselbe Folge als numpy-Vektor: Ganzzahlen und Gleichverteilung bitgleich, Normalzahlen bis 1e-12). Zufallsanteile stehen als Band, Gleitkomma-Letztstellen nur mit Toleranz.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen sind eigene Nachrechnung in diesen Modellen; Voraussetzung ist überall: iid-Beobachtungen aus den genannten Verteilungen, die genannten festen k, die genannte Zahl Läufe.

### A. Verfahrens-Matrix (m = 50 Jahre, 100-Jahres-Pegel, R = 400 Läufe, Zufalls-Seed-Basis 910 000)

RMSE von ln(Pegel-Dach/wahr); alle Verfahren sehen dieselben Reihen:

{{MATRIX}}

**Zählung der Siege** über die 32 Zellen (kleinster RMSE; drei Zufalls-Seed-Basen; die Zählung gilt als ± 3 Zellen, ein RMSE hat bei R = 400 einen Standardfehler von grob 4 % des Wertes):

{{WINS}}

* **Gumbel gewinnt {{GUMBEL_BAND}} von 32 Zellen** (12 feste Zellen: alle Exponential-Zellen und die Normal-Zellen mit m = 10 und 20, dazu 0 bis 2 Beinahe-Gleichstände bei Lognormal), **unterschätzt aber in 16 von 32 Zellen in mindestens 90 % der Läufe** (erster Zufalls-Seed; je Zufalls-Seed 15 bis 16 Zellen der Pareto-Verteilung und der Lognormal-Verteilung, die Lognormal-Zelle m = 10, 100-Jahres-Pegel liegt bei 89,5 bis 93,8 %; in keiner Normal- oder Exponential-Zelle, dort höchstens 64,5 %). Das ist **keine Aussage zugunsten von Gumbel**: Er gewinnt dort, wo der Rand der Blockmaxima tatsächlich Gumbel-artig ist (Exponential mit ξ = 0, Normal bei kurzen Reihen); wo ξ = 0 nur im Grenzwert gilt (Lognormal), unterschätzt er. Bei der Pareto-Verteilung liegt er im Median beim 0,39- bis 0,47-Fachen (100-Jahres-Pegel, m = 10 bis 100) bzw. 0,16- bis 0,20-Fachen (1000-Jahres-Pegel) des wahren Pegels.
* **Hill gewinnt unter den vier gemessenen Verteilungen nur bei der Pareto-Verteilung** (dem reinen Potenzgesetz, das er voraussetzt; alle 8 Siege in den 8 Zellen der Pareto-Verteilung, in allen drei Zufalls-Seeds) und liegt bei Exponential, Normal und Lognormal im Median beim 1,66-, 1,66- und 1,69-Fachen des wahren 100-Jahres-Pegels (k = 5 %, m = 50), beim 1000-Jahres-Pegel beim 2,43-, 2,38- und 2,56-Fachen.
* **Schwelle gegen Block:** Dass GPD-ML mit k = 5 % einen kleineren RMSE hat als die Blöcke (Pareto-Verteilung 0,35 gegen 0,48), kommt bei der Pareto-Verteilung von der **fünffachen Zahl benutzter Werte** (k = 250 gegen 50 Maxima; mit k = 50 liegt die GPD-ML dort bei 0,53 und damit über den Blöcken) und sagt nichts über „die Schwelle nutzt Daten besser“; bei gleicher Zahl (k = n/b) streut sie bei Exponential und Normal kaum anders (1,01), bei Pareto-Verteilung und Lognormal um 8 und 6 % mehr (Verhältnis der ln-Streuung Schwelle/Block 1,01 / 1,08 / 1,01 / 1,06 für Exponential / Pareto-Verteilung / Normal / Lognormal, Zelle m = 50, 100-Jahres-Pegel; der Zufallsanteil der Quotienten ist nicht gemessen; Block gegen Schwelle führt „Überschreitungen und die GPD“, mit beiden Kopplungen):

{{KOPPLUNG}}

{{XI}}

**Kurze Reihe, m = 10 Jahre (N = 1 000), 100-Jahres-Pegel, Median des Betrags |ln(Pegel-Dach/wahr)|** (unempfindlich gegen Ausreißer):

{{M10}}

Die RMSE der GEV-ML bei m = 10 (1,46 / 3,49 / 1,49 / 1,88) sind ausreißerbestimmt und hängen an der Randbehandlung ξ → −1 des Schätzers: **sie werden nicht als Zahl des Verfahrens geführt**; belastbar ist die Rangfolge „ML schlechter als L-Momente“ im Median.

### B. Reihenlänge gegen Wiederkehrperiode (exakt)

Der Anteil der Reihen aus m iid-Jahresmaxima, die den T-Jahres-Pegel nicht erreichen, ist (1 − 1/T)^m: **60,5 %** (m = 50, T = 100), **95,1 %** (T = 1000), **36,6 %** bei m = T = 100; das 50-Jahres-Maximum erreicht den wahren 100er-Wert also in 39,5 %, den 1000er-Wert in 4,9 %. Streuband des Pegels aus 50 Jahren **bei GEV(μ = 100, σ = 20, ξ = 0,2)** (nicht nur „ξ = 0,2“: das Band hängt an Ort und Skala), **zitiert aus „Wiederkehrwert“** (R = 2 000 Läufe je Zelle):

{{BAND3}}

### C. Puffer jenseits der Daten (R = 1 000 Läufe je Zelle; Bezugsmenge: Überschreitungswahrscheinlichkeit p je Beobachtung)

**Die Überschreitungswahrscheinlichkeit des empirischen Quantils verfehlt das Ziel nicht wegen der Form des Rands, sondern wegen der Zahl der Beobachtungen oberhalb des Ziels; wie groß der nötige Aufschlag auf den Wert ist, hängt dagegen stark am Rand** (f90 des Maximums bei n = 2 000, p = 10⁻⁴: 1,083 bei Normal, 1,361 bei Exponential, 2,205 bei Vehikel B, 3,392 bei der Pareto-Verteilung). Die wahre Überschreitungswahrscheinlichkeit des empirischen Quantils ist für jede stetige Verteilung Beta(n−j+1, j), j = ⌈(1−p)(n+1)⌉ (höchstens n); exakt:

{{PUFFER_EXAKT}}

Bei n·p < 1 ist der Puffer das Maximum: P(q > 2p) = (1 − 2p)^n; n = 2 000, p = 10⁻⁴ gibt 0,670 und im Mittel die 5,00-fache Soll-Überschreitung (E[q] = 1/(n + 1)). Die Simulation (1 000 Läufe, vier Fälle) liegt in allen zwölf Zellen im Band von 4,5 Standardfehlern um diese exakten Werte. Verfahren am Ziel n = 2 000, p = 10⁻⁴ (n·p = 0,2):

{{PUFFER}}

* **Ab n·p ≥ 5 bringen GPD und Hill in P(q > 2p) nichts gegenüber dem empirischen Quantil** (16 Zellen aus vier Fällen mit n·p = 5, 20 und 50: höchstens 0,033 empirisch, 0,032 GPD, 0,034 Hill-Weissman; ihr Gesamtpuffer liegt dort nicht über dem des empirischen Quantils); die GEV an nur 5 Blöcken schneidet dort schlechter ab (bis 0,162 bei der Pareto-Verteilung, n = 500, p = 10⁻²), bei 20 und 50 Blöcken liegt sie nahe am empirischen Quantil (höchstens 0,038 gegen 0,033). Darunter streuen die Fehlgriffe von GPD, Hill und GEV bei n·p ≤ 0,2 weniger (Vehikel B 0,220 / 0,004 / 0,318 gegen 0,672); **GPD und GEV halten das Ziel dennoch nur in 36 bis 60 % der Läufe** (P(q ≤ p) bei n·p = 0,2), Hill-Weissman hält es bei Vehikel B, Exponential und Normal (95,6 / 100 / 99,9 %) durch einen 1,1- bis 1,7-fachen Puffer (Median Puffer/wahr 1,56 / 1,65 / 1,11), bei der Pareto-Verteilung nur in 50,1 %. Der **Gesamtpuffer der GPD wird bei n·p = 0,2 in keinem der vier Fälle kleiner** (Vehikel B: GPD 1,68 gegen 1,48 empirisch): der kleinere Aufschlag der GPD ist dort ein höherer Grundwert; bei n·p = 0,5 (n = 5 000, p = 10⁻⁴) ist er in allen vier Fällen kleiner (Vehikel B 1,38 gegen 1,48, Pareto-Verteilung 1,53 gegen 1,87), und der von Hill-Weissman liegt bei n·p = 0,2 bei Vehikel B und Pareto-Verteilung darunter (1,41 gegen 1,48, 1,50 gegen 1,87). Bei Normal-Daten ist die GPD bei n·p = 0,2 schlechter als die Normal-Formel (0,451 gegen 0,003).
* **Verteilungsannahmen:** Die Normal-Formel übersteigt bei Vehikel B das Ziel p = 10⁻⁴ (n = 2 000) in allen 1 000 Läufen um mehr als das Doppelte (P(q > 2p) = 1,000), im Median ist q das 107,6-Fache von p; die Lognormal-Formel puffert bei Exponential-Daten 4,26-fach (p = 10⁻³) und 7,15-fach (p = 10⁻⁴) zu groß (Median Puffer/wahr, n = 2 000); die Normal-Formel ist bei Normal-Daten fast optimal (f90 1,00 bis 1,02).
* **Aufschlag f90 und Gesamtpuffer sind verschiedene Maße:** f90 ist ein Faktor auf den eigenen Schätzer; bei Vehikel B und n·p = 0,2 braucht das empirische Quantil f90 = 2,205 exakt (Aufschlag **120,5 %**, U4; simuliert 2,16), die GPD 1,53 (53 %). Bei n·p < 1 ist f90 des Maximums exakt aus F^n berechenbar.
* **Kurze Blockreihen:** Das Maximum von m Blöcken hat q ~ Beta(1, m); bei m = 6, p = 1/12 ist E[q] = 1/7 und P(q > 2p) = (5/6)^6 = 0,335; sechs Monatsspitzen reichen für ein 90-%-sicheres Ziel nicht (Aufschlag etwa 1,44 σ bei ξ = 0, 2,07 σ bei ξ = 0,2). GEV-ML bringt bei m = 6 nichts (0,346 / 0,337); bei falschem ξ = 0 festgehalten kippt Gumbel (m = 24, p = 1/120, ξ = 0,2: 0,78 gegen 0,65 des Maximums).

### D. CVaR aus Szenarien (Median des Verhältnisses empirisch/exakt)

{{CVAR}}

**Ein CVaR aus Szenarien liegt bei n = 60 und 600 im Median unter dem echten, und der Abstand wächst mit dem Niveau und dem Gewicht des Rands; bei n = 6 000 und α ≤ 0,95 beträgt er höchstens 2 % (2 000 Läufe; Pareto-Verteilung 1,5 %).** Im Bereich der Referenz der `cvar-demo` (n = 600, α = 0,90) ist der Abstand klein (0,961 bis 0,997); die großen Abstände liegen bei α ≥ 0,99. Bei α = 0,999 und n = 600 ist (1−α)·n = 0,6 < 1: der empirische CVaR **ist das Maximum der 600 Szenarien**, und mit Wahrscheinlichkeit 0,999^600 = 54,86 % liegt kein Szenario über dem 99,9-%-Quantil (exakt). Das **Mittel** des Verhältnisses wird nicht geführt (bei der Pareto-Verteilung α = 2 ist die Varianz unendlich). Eine GPD an die oberen 5 % hilft dort, nicht überall:

{{CVAR_GPD}}

Die README der `cvar-demo` sagt, der echte Tail liege vermutlich höher; **das stützt diese Messung für diese fünf Verteilungen und n = 600 im Median, nicht für die Verteilung der `cvar-demo`**. CVaR bleibt dort Optimierungsziel.

## Die Entscheidungstabelle

Jede Zeile nennt Zähleinheit und Bezugsmenge; Zeilen anderer Demos zitieren die Demo, die die Zahl führt:

{{ENTSCHEIDUNG}}

**Zitierte Zahlen** (Demo, Wortlaut; `tests/test_claims.py` sucht ihn in den READMEs der Nachbar-Demos):

{{ZITATE}}

## Ehrliche Grenzen

* **Ausgedachte Verteilungen, feste k, eine Blocklänge.** Vier Verteilungen in der Matrix, k = 50 und 5 % der Beobachtungen, Blöcke zu 100 Beobachtungen. Andere Blocklängen, andere k, andere Reihenlängen und **Vehikel B in der Matrix** sind nicht gemessen (Vehikel B steht als Zeile von „Pegel statt Index“ und als Puffer-Fall; live in Experiment 2 kann man es ziehen).
* **Die Zählung der Siege ist keine Rangordnung der Verfahren**: ein Sieg ist der kleinste RMSE in einer Zelle, Beinahe-Gleichstände wechseln den Sieger (Zählung ± 3).
* **Keine Kostenfunktion beim Puffer**, keine Abhängigkeit, kein Trend, keine Summe über die Wiederbeschaffungszeit; welche Zeile „besser“ ist, bleibt offen. Der Nutzen in Logistik, Lager und Hafen ist nicht belegt (Ampel: Orakel grün, Messbefund gelb, Anwendungsnutzen rot).
* **Die Schwellen der Hinweise** (50 %, 20, 100, 50) sind eigene Festlegungen aus den Zahlen der Tabelle.
* **Plattform:** Zufallsanteile als Band, Gleitkomma nur mit Toleranz; Linux (Docker, `demo_linux_check`) und Streamlit Cloud sind nicht geprüft; die Live-Rechenzeiten sind lokal gemessen und in der Cloud länger.

## Tests

Die Suite hat **267 Tests**; Laufzeit seriell 153 s und 163 s (zwei Läufe, Windows, ein Thread, Python 3.14; die Zeit hängt an Rechner und Last). Zusammensetzung: Handbeispiele 17, Orakel 86 (scipy nur dort), Eigenschaften 13, Randfälle 9, eingefrorene Daten samt `--check` der Werkzeuge 14, Claims 40, Kopien 8, Zufall 4, Vehikel B 7, Verdikte 14, Presets 7, Texte 7, Oberfläche (AppTest) 9, Layout 29, Footer 2, Laufzeit-Importe 1. `tests/test_app.py` beginnt mit der Portfolio-Vorlage (sie zählt nur `st.slider`), darunter stehen die Tests für die Auswahl-Regler.

Zahl → Test (die Zahlen der README stehen in den Daten bzw. werden exakt gerechnet; ausgenommen sind Laufzeiten, die alten Plan- und Vormessungszahlen unter „Befunde und Korrekturen“ und die zitierten Zahlen anderer Demos, deren Wortlaut in den READMEs der Nachbar-Demos gesucht wird; die Tabellen oben werden von `tools/vwr_readme_tables.py` aus den Daten erzeugt, und `test_claims.py` prüft, dass die README sie unverändert enthält):

| Befund | Test | Art |
|---|---|---|
| Reihenlänge 60,5 / 95,1 / 36,6 / 63,4 / 39,5 % | `test_claims.py::test_reihenlaenge_exakt`, `test_handbeispiele.py::test_h1_*` | exakt (Brüche) |
| Matrix-Tabelle, Zählung der Siege, 12 feste Zellen, 16 Zellen, Hill nur bei der Pareto-Verteilung | `test_claims.py::test_matrix_*`, `test_reference.py::test_zaehlung_*` | eingefroren; `--check` rechnet als Band neu |
| Kurze Reihe m = 10, ML am Rand | `test_claims.py::test_kurze_reihe_*` | eingefroren |
| Block gegen Schwelle in der Matrix-Zelle, ξ̂-Streuung | `test_claims.py::test_kopplung_*` | eingefroren (R = 2 000) |
| Beta-Größen des Puffers, 0,370 / 0,670 / 0,905, 4,998, (5/6)^6, 0,904 | `test_claims.py::test_puffer_exakt_*`, `test_oracle_verfahrenswahl_rand.py::test_beta_*`, `test_handbeispiele.py::test_h6_*` | exakt, gegen `scipy.stats.beta` |
| Aufschlag 120,5 % (U4), 2,205 | `test_claims.py::test_aufschlag_*`, `test_oracle_verfahrenswahl_rand.py::test_aufschlag_*` | exakt; Simulation im Band |
| Puffer-Tabelle (Vehikel B und drei weitere Fälle), kurze Blockreihen | `test_claims.py::test_puffer_*`, `test_reference.py::test_puffer_*` | eingefroren; Orakel im Band |
| CVaR-Tabellen, 54,86 %, (1−α)·n < 1 | `test_claims.py::test_cvar_*`, `test_oracle_verfahrenswahl_rand.py::test_exakter_cvar_*`, `test_handbeispiele.py::test_h4_*` | eingefroren; exakt gegen Quadratur |
| Entscheidungstabelle, zitierte Zahlen | `test_claims.py::test_entscheidungstabelle_*`, `test_claims.py::test_zitate_*` | gegen die Daten bzw. die READMEs der Nachbar-Demos |
| Zählungen über die drei Zufalls-Seeds (15 bis 16 Zellen, 89,5 bis 93,8 %, 0,75 bis 12,5 %, 88,8 bis 94,0 %) | `test_claims.py::test_gumbel_unterschaetzung_*`, `test_claims.py::test_ml_an_der_grenze_*` | eingefroren |
| Puffer unter n·p < 1 (36 bis 60 %, Hill-Weissman, Gesamtpuffer je n·p, GEV je Blockzahl), f90 des Maximums | `test_claims.py::test_puffer_unter_einer_beobachtung_*`, `test_puffer_gesamtpuffer_*`, `test_normal_formel_*` | eingefroren; f90 exakt |
| CVaR-Abstand bei n = 60, 600 und 6 000 | `test_claims.py::test_cvar_abstand_*` | eingefroren |
| „Spitzen-Cluster“ nach der Korrektur dort | `test_claims.py::test_stueck7_*` | Wortlaut im Zitat und in der README; die Zahlen selbst stehen im README von „Spitzen-Cluster“ |
| Hinweise, Jahre für 80 % (99 / 62 / 39), Schwelle bei m = 68 / 69 | `test_handbeispiele.py::test_h5_*`, `test_properties.py::test_hinweise_*` | exakt |
| Verdikte je Preset | `test_verdicts.py` | Zeichenketten gegen die Daten |
| Befunde und Korrekturen (alt → neu) | Messung dieser README; die neuen Werte stehen in den Tabellen oben | nur neu gemessen, die alten Werte sind Plan-Zahlen |

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-App: vier Experimente, Entscheidungstabelle, 🚧, 📐 |
| `vwr_estimators.py` | die acht Verfahren der Matrix und die Verfahren des Puffers (numpy, Standardbibliothek) |
| `vwr_gev.py`, `vwr_blockfit.py`, `vwr_gpd.py`, `vwr_hill.py` | wörtliche Abschriften aus „GEV an Blockmaxima“ (`gev-blockmaxima-demo`), „Überschreitungen und die GPD“ (`gpd-ueberschreitungen-demo`) und „Pegel statt Index“ (`hill-pegel-demo`), geprüft von `tests/test_copies.py` |
| `vwr_scenario.py` | Verteilungen: Ziehen, exakte Überlebensfunktion, exakter Pegel |
| `vwr_buffer.py`, `vwr_cvar.py` | exakte Größen des Puffers (Beta, Binomialsumme) und der CVaR (empirisch und exakt) |
| `vwr_evaluation.py`, `vwr_table.py` | `hinweise`, Analysen, Verdikte als reine Funktionen; Entscheidungstabelle und zitierte Zahlen |
| `vwr_reference.py` | lädt die eingefrorenen Messungen |
| `vwr_grid.json`, `vwr_cvar.json`, `vwr_puffer.json` | eingefrorene Messungen (Quelle und Stand je Teil) |
| `vwr_vehicle.py`, `vwr_vehikel_b_tabelle.json` | Vehikel B: byte-gleiche Kopie der gemeinsamen Gewichtstabelle (SHA-256 `669f821c78e0bf38c47fa846232f27cd2ff126f4fceadca5369a5c9af5551c9f`) |
| `vwr_rng.py`, `vwr_vecrng.py` | SplitMix64: Kopie der Vorlage und Vektorform |
| `vwr_constants.py`, `vwr_presets.py`, `vwr_layout.py`, `vwr_texts.py`, `vwr_visualization.py` | Regler, Presets mit Permalink, fester Kopf, Texte, Abbildungen |
| `tools/vwr_make_grid.py`, `vwr_make_cvar.py`, `vwr_make_puffer.py`, `vwr_make_vehicle.py` | erzeugen und prüfen (`--check`) die Daten und die Tabelle des Vehikels B |
| `tools/vwr_readme_tables.py`, `tools/vwr_readme_vorlage.md` | erzeugen die Tabellen der README aus den Daten |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Randfälle, eingefrorene Daten, Claims, Kopien, Zufall, Verdikte, Presets, Texte, Oberfläche, Layout, Footer, Laufzeit-Importe |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit: streamlit, numpy, plotly, pandas; dev zusätzlich pytest und scipy (nur Tests) |

## 🚧 Bewusst nicht enthalten / nicht gemessen

* **Nutzen in Betrieb:** Keine Aussage über den Wert einer besseren Pegel-Schätzung in Logistik, Lager oder Hafen; das Deich-Beispiel (Embrechts u. a. 8.1.5) und die Abstract-Belege (Zhu u. a. 2017, Berkeley-Bericht 2022, Lee u. a. 2022) nur als „Lehrbuch/Abstract“.
* **Intervalle für den GPD-Pegel, Schwellenunsicherheit im Pegel-Intervall, Profil-Likelihood in der Matrix** sind in der Linie nicht gemessen; Abhängigkeit zusammen mit Trend, Schwelle unter Trend, Kovariaten außer Zeit und die Modellwahl „erst Trendtest, dann Modell“ ebenfalls nicht.
* **Hill, Schwellenwahl und Block gegen Schwelle bei anderen Blocklängen unter Abhängigkeit** (nur b = 50 in „Spitzen-Cluster“); die Voraussetzung „Block ≫ 1/θ“ ist eine Folgerung aus der Theorie (Coles, Satz 5.1). Ob die zu geringe Überdeckung der Schwelle bei Abhängigkeit am Delta-Intervall oder an der GPD-Anpassung bei Clustern liegt, trennt auch „Spitzen-Cluster“ nicht.
* **Modellwahl-Regeln und Schwellenautomaten in der Matrix**, andere k, andere Reihenlängen, Vehikel A und B in der Matrix, Mehrdimensionales (Verweis `randabhaengigkeit-demo`).
* **Die θ-Quelle bleibt Stufe A** (Peshkova nur Kurzseite, Hooghiemstra und Meester nicht gelesen); die Formel θ = (1−ρ)² ist in „Spitzen-Cluster“ durch zwei Rechenwege belegt.
* **CVaR:** nur fünf ausgedachte Verteilungen, nicht die Szenarien der `cvar-demo`; der Zusammenhang mit der LP-Optimierung (der CVaR des Plans, nicht der Nachfrage) ist nicht gemessen.
* **Quellen nicht geprüft:** Belzile/Davison (die Zahlen 48 bis 348 mm nur nach Zusammenfassung, ungeprüft) und Slater u. a. 2021.
* **Nicht geprüft:** Linux/Docker (`demo_linux_check`), Streamlit Cloud, dunkles Design, echter Mausklick auf Plotly, unabhängiger Aussagenprüfer, `modul_frisch.py`, Erklärseite „verständlich erklärt“ (nicht gebaut).

## Quellen und Stufen

* **W (Wortlaut gelesen, nur Gleichungen und Befunde in eigenen Worten):** Coles 2001, *An Introduction to Statistical Modeling of Extreme Values*; Embrechts, Klüppelberg, Mikosch 1997; Leadbetter, Lindgren, Rootzén 1983; Rootzén und Katz 2013; Clauset u. a. 2009. Huang, Nychka, Zhang 2018 (arXiv 1802.09387, Tabelle 1): W, lokal gelesen.
* **Teils gelesen:** Scarrott und MacDonald 2012.
* **Nach Zusammenfassung des Abrufwerkzeugs, ungeprüft:** Belzile und Davison 2026, Slater u. a. 2021. **Nur Abstract:** Zhu u. a. 2017, Lee u. a. 2022.
* **Nicht im Original gelesen** (nie als gelesen ausgegeben): Pickands 1975, Balkema und de Haan 1974, Hill 1975, Fisher und Tippett 1928, Jenkinson 1955, Davison und Smith 1990, Hosking und Wallis, Hall 1979, Smith 1985, de Haan und Ferreira, Beirlant u. a., Rockafellar und Uryasev (die Form des CVaR steht in der `cvar-demo`). Die Buchkopien haben unklare Weitergaberechte, darum stehen hier nur Gleichungen und Befunde in eigenen Worten.
* **Keine Neuheit:** alles Lehrbuch; neu ist die Vorführbarkeit mit Zähleinheit und exakten Orakeln.

## Lokal ausführen

```
pip install -r requirements.txt
streamlit run app.py --server.port 10091
```

Daten neu erzeugen und prüfen: `python tools/vwr_make_grid.py --write` (beim Bau 131 s), `vwr_make_cvar.py --write` (37,6 s), `vwr_make_puffer.py --write` (82,5 s), jeweils auch `--check`. Die Zeiten stehen als `laufzeit_s` in den Dateien und hängen an Rechner und Last: bei einer Prüfung unter fremder Last dauerten die drei Läufe 575, 224 und 518 s.

Gebaut mit Streamlit, Plotly, pandas und numpy.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
