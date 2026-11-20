"""Prüfer der kopierten Gewichtstabelle des Vehikels B (`vwr_vehikel_b_tabelle.json`).

Aufruf (aus dem Demo-Ordner, mit dem Test-venv; OMP_NUM_THREADS=1):
    python tools/vwr_make_vehicle.py --check    # zählt die 1 224 Komponenten aus der Definition neu auf und vergleicht mit der Tabelle (größte Abweichung unter 1e-12)

`aufzaehlung()` ist eine EIGENE Aufzählung des Vehikels B aus der Definition der Linie (kein Auszug aus dem Code der Linie): Art × Kunde × Zoll × Wochenende × Vorlauf (Gauß-Legendre mit
48 Knoten), je Komponente log-Median und Streuung; gleiche (μ, s) werden zusammengefasst. Die Tests (tests/test_vehikel_b.py) benutzen sie als Gegenprobe der kopierten Tabelle."""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

# Definition (Quelle: gemeinsame Definition der Linie „Extremwerte erklärt“, Vehikel B; die Kundeneffekte sind default_rng(12345).normal(0, 0,30, 12) als Zahlen)
ART_P = (0.35, 0.30, 0.25, 0.10)
K_KUNDEN = 12
U_SCALE = (1.0, 0.3, 0.3, 1.0)
S_BASE = (0.55, 0.25, 0.35, 0.90)
P_ZOLL = 0.25
P_WOCHENENDE = 2.0 / 7.0
N_KNOTEN = 48
UK = (-0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927, -0.02260299210315629, -0.2222653956256827,
      -0.410337810534883, 0.19466784065791196, 0.10831743391646849, -0.5858589189036569, 0.7042228963136555, 0.2905490717255771)


def _median(art: int, zoll: int, we: int, vorlauf: float) -> float:
    if art == 0:
        return 3.5 * math.exp(0.7 * zoll + 0.30 * we)       # Import
    if art == 1:
        return 0.6 + 0.85 * vorlauf                          # Export
    if art == 2:
        return 0.3 + 0.8 * vorlauf                           # Umschlag
    return 7.0 * math.exp(0.30 * we)                         # Leer


def aufzaehlung():
    """(Komponenten als Feld Gewicht/μ/s nach dem Zusammenfassen, Zahl der Aufzählungsschritte, Zahl der Schritte mit Gewicht 0)."""
    xg, wg = np.polynomial.legendre.leggauss(N_KNOTEN)
    merged: dict = {}
    steps = zero = 0
    for art in range(4):
        if art in (1, 2):
            lo, hi = (1.0, 10.0) if art == 1 else (0.5, 6.0)
            vor = [(lo + (hi - lo) * (x + 1.0) / 2.0, w / 2.0) for x, w in zip(xg, wg)]
        else:
            vor = [(0.0, 1.0)]
        for kunde in range(K_KUNDEN):
            effekt = UK[kunde] * U_SCALE[art]
            for zoll in (0, 1):
                p_zoll = (1.0 - P_ZOLL if zoll == 0 else P_ZOLL) if art == 0 else (1.0 if zoll == 0 else 0.0)
                for we in (0, 1):
                    p_we = P_WOCHENENDE if we else 1.0 - P_WOCHENENDE
                    for v, wv in vor:
                        steps += 1
                        gew = ART_P[art] / K_KUNDEN * p_zoll * p_we * wv
                        if gew <= 0.0:
                            zero += 1
                            continue
                        mu = math.log(_median(art, zoll, we, v)) + effekt
                        key = (round(mu, 12), S_BASE[art])
                        if key in merged:
                            merged[key][0] += gew
                        else:
                            merged[key] = [gew, mu, S_BASE[art]]
    return np.array(list(merged.values())), steps, zero


def main(argv):
    if "--check" in argv:
        comps, steps, zero = aufzaehlung()
        data = json.loads((ROOT / "vwr_vehikel_b_tabelle.json").read_text(encoding="utf-8"))
        tab = np.array(data["komponenten"])
        assert steps == 4704 and zero == 2328 and len(comps) == 1224 == len(tab), (steps, zero, len(comps), len(tab))
        a = comps[np.lexsort((comps[:, 1], comps[:, 2]))]
        b = tab[np.lexsort((tab[:, 1], tab[:, 2]))]
        diff = float(np.max(np.abs(a - b)))
        assert diff < 1e-12, f"Tabelle weicht ab: {diff:.2e}"
        print(f"Tabelle in Ordnung: {len(tab)} Komponenten, {steps} Aufzählungsschritte ({zero} mit Gewicht 0), größte Abweichung {diff:.1e}")
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
