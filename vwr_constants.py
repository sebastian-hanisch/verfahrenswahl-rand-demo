"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

SEED_MAX, DEFAULT_SEED = 999999, 35

# Experiment 1: meine Lage
M1_MIN, M1_MAX, DEFAULT_M1 = 5, 500, 50              # Reihenlänge in Jahren (Blöcke, je 100 Beobachtungen)
T_OPTIONS, DEFAULT_T = (10, 30, 100, 300, 1000), 100  # Wiederkehrperiode in Jahren
K1_MIN, K1_MAX, DEFAULT_K1, K1_STEP = 10, 500, 250, 10  # Überschreitungen der Schwellenverfahren
THETA_OPTIONS, DEFAULT_THETA = (1.0, 0.5, 0.2, 0.1), 1.0  # Extremalindex θ (1 = unabhängige Beobachtungen)
BETA_OPTIONS, DEFAULT_BETA = (0.0, 0.005, 0.01, 0.02, 0.04), 0.0  # Trend je Jahr in σ-Einheiten (0 = kein Trend)
RAND_OPTIONS, DEFAULT_RAND = ("unbekannt", "bekannt", "schwer"), "unbekannt"
RAND_LABELS = {"unbekannt": "unbekannt, ξ > 0 nicht ausgeschlossen", "bekannt": "aus Fachwissen sicher ξ = 0", "schwer": "schwer (Pareto-Verteilung-artig, ξ > 0)"}

# Experiment 2: alle Verfahren auf dieselbe Reihe
DIST2_OPTIONS, DEFAULT_DIST2 = ("exp", "pareto", "normal", "lognormal", "vehikel_b"), "pareto"
M2_OPTIONS, DEFAULT_M2 = (10, 20, 50, 100), 50
TARGET2_OPTIONS, DEFAULT_TARGET2 = (100, 1000), 100  # 100- bzw. 1000-Jahres-Pegel
R2_MIN, R2_MAX, DEFAULT_R2, R2_STEP = 10, 200, 100, 10   # Läufe live; die Matrix (R = 400) ist eingefroren

# Experiment 3: wie viel bleibt unsicher, Puffer
CASE3_OPTIONS, DEFAULT_CASE3 = ("vehikel_b", "pareto", "exp", "normal_100_20"), "pareto"
N3_OPTIONS, DEFAULT_N3 = (500, 2000, 5000), 2000
P3_OPTIONS, DEFAULT_P3 = (1e-2, 1e-3, 1e-4), 1e-3

# Experiment 4: CVaR
DIST4_OPTIONS, DEFAULT_DIST4 = ("normal", "exp", "lognormal", "pareto", "t3"), "pareto"
N4_OPTIONS, DEFAULT_N4 = (60, 600, 6000), 600
ALPHA4_OPTIONS, DEFAULT_ALPHA4 = (0.90, 0.95, 0.99, 0.999), 0.99
R4_LIVE = 200                                          # Läufe der Live-Verteilung (bei n = 6 000 nur 100)

SEED_OFFSET_2, SEED_OFFSET_4 = 20000, 40000           # eigene Ströme je Experiment aus demselben Zufalls-Seed

COLOR_GOOD = "#2F6B65"
COLOR_BAD = "#B5543C"
COLOR_OTHER = "rgba(150,150,150,0.55)"
COLOR_LINE = "#4c78a8"
COLOR_TRUTH = "#222222"

PRESETS = {
    "50 Jahre, 100er-Pegel": {},        # Standard: alle Regler auf ihren Voreinstellungen
    "Kurze Reihe, 10 Jahre": {"m1": 10, "T1": 100, "k1": 50, "theta1": 1.0, "beta1": 0.0, "rand1": "unbekannt", "dist2": "exp", "m2": 10, "target2": 100},
    "50 Jahre, 1000er-Pegel": {"T1": 1000, "target2": 1000},
    "Abhängig, θ = 0,2": {"k1": 150, "theta1": 0.2},
    "Trend 0,02": {"beta1": 0.02},
    "Schwerer Rand (Pareto-Verteilung)": {"rand1": "schwer", "dist2": "pareto", "target2": 1000},
    "Normal": {"dist2": "normal", "dist4": "normal"},
    "Puffer jenseits der Daten": {"case3": "vehikel_b", "n3": 2000, "p3": 1e-4},
}
PRESET_HELP = {
    "Kurze Reihe, 10 Jahre": "10 Jahre zu je 100 Beobachtungen, Ziel 100-Jahres-Pegel: die Warnung „kurze Reihe“ und die Matrix-Zelle für Exponential-Daten.",
    "50 Jahre, 100er-Pegel": "Die Voreinstellung: 50 Jahre, Ziel 100-Jahres-Pegel, Pareto-Verteilung in der Matrix. Nur die Warnung „Extrapolation“ trifft zu.",
    "50 Jahre, 1000er-Pegel": "Dieselbe Reihe, Ziel 1000-Jahres-Pegel: in 95 % der Reihen liegt keine Beobachtung auf dieser Höhe.",
    "Abhängig, θ = 0,2": "Spitzen kommen in Gruppen (Extremalindex 0,2): bei k = 150 sind das nur 30 Gruppen, die Warnung „wenig Cluster“ trifft zu.",
    "Trend 0,02": "Ein Trend von 0,02 σ je Jahr braucht laut Formel rund 62 Jahre, um mit 80 % Trennschärfe erkannt zu werden (Test auf Lage-Trend, ξ bekannt; die Simulation von „Trend“ liegt dort bei 79,8 %).",
    "Schwerer Rand (Pareto-Verteilung)": "Pareto-Verteilung α = 2, Ziel 1000-Jahres-Pegel in der Matrix: Gumbel liegt weit unter dem wahren Pegel, Hill gewinnt unter den vier gemessenen Verteilungen nur bei dieser (dem reinen Potenzgesetz, das er voraussetzt).",
    "Normal": "Normal-Daten in Matrix und CVaR: ein leichter Rand, Gumbel überschätzt leicht.",
    "Puffer jenseits der Daten": "Vehikel B, 2 000 Beobachtungen, p = 10⁻⁴: n·p = 0,2, das Ziel liegt jenseits der Daten.",
}
