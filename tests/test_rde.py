"""Rotating disk electrode: mass-transport coupling and outer-sphere redox.

A solution species' surface activity is an unknown set by balancing the Levich
diffusion flux against the (Butler-Volmer) reaction, so the current crosses over
from kinetic to mass-transport control. The outer-sphere couple Fe2+ <-> Fe3+ has
no adsorbed intermediate at all: the model is just the two solution species and
one faradaic step.
"""

import numpy as np
import pytest

import discopt_mkm as mk

R, T, F = 8.617e-5, 298.0, 1.0


def _fe_rde(omega, A=1e-2, U=0.45, D=1e-5, Cb_Fe2=1.0, Cb_Fe3=0.0):
    """Outer-sphere Fe2+ -> Fe3+ + e- (oxidation, n_electrons = -1) on an RDE."""
    m = mk.Model("fe", T=T, R=R, U=U, F=F)
    Fe2 = m.gas("Fe2", H=0.0, S=0.0, composition={"Fe": 1})
    Fe3 = m.gas("Fe3", H=0.2, S=0.0, composition={"Fe": 1})
    rxn = m.step(Fe2 >> Fe3, A=A, Ea=0.0, n_electrons=-1, beta=0.5)
    rde = mk.RotatingDiskElectrode(bulk={Fe2: Cb_Fe2, Fe3: Cb_Fe3}, omega=omega,
                                   diffusivities={Fe2: D, Fe3: D})
    return m, Fe2, Fe3, rxn, rde


def test_outer_sphere_model_solves_without_adsorbates():
    m, Fe2, Fe3, rxn, rde = _fe_rde(omega=400.0)
    assert m.adsorbates == [] and m.sites == []   # no surface intermediate
    sol = mk.solve_steady_state(m, rde)
    assert sol.status == "optimal"


def test_levich_limiting_current_and_scaling():
    # fast kinetics -> fully transport limited; |j| = F k_m C_bulk and scales as sqrt(omega)
    j = {}
    for w in (100.0, 400.0, 1600.0):
        m, Fe2, Fe3, rxn, rde = _fe_rde(omega=w, A=1e4, U=0.8)
        sol = mk.solve_steady_state(m, rde)
        j[w] = F * abs(sol.rate_of_progress(rxn))
        assert j[w] == pytest.approx(rde.limiting_current(Fe2, 1, F), rel=1e-3)
    # Levich: doubling sqrt(omega) (100 -> 400) doubles the limiting current
    assert j[400.0] / j[100.0] == pytest.approx(2.0, rel=1e-3)
    assert j[1600.0] / j[100.0] == pytest.approx(4.0, rel=1e-3)


def test_koutecky_levich_linearity_recovers_kinetic_current():
    omegas = np.array([100.0, 400.0, 900.0, 1600.0, 2500.0])
    j = []
    for w in omegas:
        m, Fe2, Fe3, rxn, rde = _fe_rde(omega=w, A=1e-2, U=0.45)
        sol = mk.solve_steady_state(m, rde)
        j.append(F * abs(sol.rate_of_progress(rxn)))
    j = np.array(j)
    slope, intercept = np.polyfit(omegas ** -0.5, 1.0 / j, 1)
    resid = (1.0 / j) - (slope * omegas ** -0.5 + intercept)
    r2 = 1.0 - resid.var() / (1.0 / j).var()
    assert r2 > 0.9999                              # 1/j is linear in omega^-1/2
    # intercept = 1 / j_kinetic = 1 / (F * k_f * C_bulk)
    kf = 1e-2 * np.exp(0.5 * F * 0.45 / (R * T))    # A * exp(-beta n F U / RT), n = -1
    assert intercept == pytest.approx(1.0 / (F * kf * 1.0), rel=1e-2)


def test_no_current_at_equilibrium_potential():
    # dG = G(Fe3) - G(Fe2) - F U = 0.2 - U  ->  equilibrium at U = 0.2 V (equal bulk activities)
    m, Fe2, Fe3, rxn, rde = _fe_rde(omega=400.0, A=1e2, U=0.20, Cb_Fe2=1.0, Cb_Fe3=1.0)
    sol = mk.solve_steady_state(m, rde)
    assert abs(F * sol.rate_of_progress(rxn)) < 1e-6
