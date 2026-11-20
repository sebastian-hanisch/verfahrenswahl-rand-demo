"""Die numpy-Vektorform des Zufalls (`vwr_vecrng`) gegen die Vorlage (`vwr_rng`): Ganzzahlen und Gleichverteilung bitgleich, Normalzahlen bis 1e-12 (numpy-log/-cos statt exact_log/exact_cos)."""

import math

import numpy as np

import vwr_rng as R
import vwr_vecrng as VR


def test_u64_und_gleichverteilung_sind_bitgleich_zur_vorlage():
    for seed in (0, 42, 910000, 2**62 + 5):
        g = R.SplitMix64(seed)
        ref_u = [g.next_u64() for _ in range(50)]
        assert VR.u64(seed, 50).tolist() == ref_u
        g = R.SplitMix64(seed)
        ref = [g.random() for _ in range(50)]
        assert [float(v).hex() for v in VR.uniform(seed, 50)] == [v.hex() for v in ref]


def test_strom_setzt_fort_wie_ein_objekt_der_vorlage():
    s = VR.Stream(7)
    a, b = s.random(10), s.random(5)
    g = R.SplitMix64(7)
    ref = [g.random() for _ in range(15)]
    assert np.concatenate([a, b]).tolist() == ref


def test_normalzahlen_im_band_der_vorlage():
    g = R.SplitMix64(123)
    ref = np.array([g.normal() for _ in range(200)])
    got = VR.Stream(123).normal(200)
    assert np.max(np.abs(got - ref)) < 1e-12


def test_normalzahlen_haben_mittel_null_und_streuung_eins_im_band():
    x = VR.Stream(5).normal(200_000)
    assert abs(x.mean()) < 4 / math.sqrt(x.size) and abs(x.std() - 1) < 4 / math.sqrt(2 * x.size)
