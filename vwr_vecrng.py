"""SplitMix64 als numpy-Vektor in der Zähler-Form: die k-te Ausgabe der Vorlage `vwr_rng.py` ist eine reine Funktion von (Zufalls-Seed, k).

Die Ausgabe Nummer k (k = 1, 2, ...) der Vorlage hat den Zustand seed + k·γ (mod 2^64) und wird dreimal gemischt; hier wird das für ein ganzes Feld auf einmal gerechnet
(uint64-Arithmetik von numpy läuft über, ohne zu warnen). `u64` und `uniform` sind Wert für Wert gleich der Vorlage (tests/test_rng_vector.py); `normal_pairs`
rechnet die Box-Muller-Formel der Vorlage mit den Funktionen von numpy statt `exact_log`/`exact_cos`: gleich bis auf die letzte Binärstelle, nicht mehr bitgleich
(die Abweichung ist im Test begrenzt). Darum hängen Messreihen mit Normalzahlen nur als Band am Zufalls-Seed, Ganzzahlen und Gleichverteilung bitgleich.
Ohne diese Vektorform kostete `random()` 1,3 Mikrosekunden und `normal()` 180 Mikrosekunden je Zahl in reinem Python (Faktor 139): n·R = 10 Millionen Ziehungen wären 13 s
bzw. 30 Minuten gewesen."""

import numpy as np

MASK = (1 << 64) - 1
_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31, _S11 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11)
TWO53 = float(1 << 53)


def u64(seed: int, count: int, start: int = 0) -> np.ndarray:
    """Die Ausgaben Nummer start+1 ... start+count der Vorlage (uint64)."""
    idx = np.arange(start + 1, start + count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed & MASK) + idx * _GAMMA
        z = (z ^ (z >> _S30)) * _M1
        z = (z ^ (z >> _S27)) * _M2
        return z ^ (z >> _S31)


def uniform(seed: int, count: int, start: int = 0) -> np.ndarray:
    """`random()` der Vorlage: 53 Bit, gleichverteilt in [0, 1)."""
    return (u64(seed, count, start) >> _S11).astype(np.float64) / TWO53


def normal_pairs(u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """Box-Muller wie `SplitMix64.normal()` (u1 nach unten durch 1e-12 begrenzt), mit numpy-Funktionen."""
    return np.sqrt(-2.0 * np.log(np.maximum(u1, 1e-12))) * np.cos(2.0 * np.pi * u2)


class Stream:
    """Ein fortlaufender Strom: jede Ziehung setzt dort fort, wo die letzte endete (wie ein Objekt der Vorlage)."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.pos = 0

    def random(self, count: int) -> np.ndarray:
        out = uniform(self.seed, count, self.pos)
        self.pos += count
        return out

    def normal(self, count: int) -> np.ndarray:
        """`count` Normalzahlen; verbraucht 2·count Ausgaben, Paare (u1, u2) aufeinanderfolgend wie in der Vorlage."""
        u = self.random(2 * count)
        return normal_pairs(u[0::2], u[1::2])
