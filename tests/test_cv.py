"""Cyclic voltammetry: the diffusion-controlled "duck".

A reversible soluble couple gives the textbook CV with a ~59 mV peak separation
(one electron, 298 K) and a peak current that scales as the square root of the
scan rate (Randles-Sevcik).
"""

import numpy as np
import pytest

from discopt.mkm.electrochem import cyclic_voltammogram


def _peaks(U, i):
    n = len(U) // 2
    Epc, ipc = U[:n][i[:n].argmin()], i[:n].min()      # cathodic (forward sweep)
    Epa, ipa = U[n:][i[n:].argmax()], i[n:].max()       # anodic (reverse sweep)
    return Epc, ipc, Epa, ipa


def test_reversible_peak_separation_is_about_59_mV():
    U, i = cyclic_voltammogram(k0=1.0, scan_rate=0.05, U_start=0.35, U_vertex=-0.35,
                               nx=300, nt_per_sweep=3000)
    Epc, ipc, Epa, ipa = _peaks(U, i)
    # reversible 1-electron couple: dEp = 2.22 RT/F ~ 57-59 mV (a little broadened on a grid)
    assert abs(Epa - Epc) == pytest.approx(0.059, abs=0.012)
    assert ipc < 0 < ipa                                # cathodic negative, anodic positive
    assert Epc < 0 < Epa                                # cathodic below E0, anodic above


def test_randles_sevcik_peak_current_scales_with_sqrt_scan_rate():
    vs = np.array([0.02, 0.05, 0.1, 0.2])
    ip = []
    for v in vs:
        U, i = cyclic_voltammogram(k0=1.0, scan_rate=v)
        ip.append(abs(i[:len(U) // 2].min()))
    ip = np.array(ip)
    ratio = ip / np.sqrt(vs)
    assert ratio.std() / ratio.mean() < 1e-2            # i_p / sqrt(v) is constant


def test_irreversible_peaks_are_further_apart_than_reversible():
    rev = cyclic_voltammogram(k0=1.0, scan_rate=0.05, U_start=0.35, U_vertex=-0.35)
    irr = cyclic_voltammogram(k0=2e-4, scan_rate=0.05, U_start=0.35, U_vertex=-0.35)
    drev = abs(_peaks(*rev)[2] - _peaks(*rev)[0])
    dirr = abs(_peaks(*irr)[2] - _peaks(*irr)[0])
    assert dirr > drev + 0.05                           # sluggish kinetics widen the gap
