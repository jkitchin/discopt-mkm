"""PFR (spatial DAE) and non-isothermal energy-balance solves."""

import numpy as np
import pytest

import discopt_mkm as mk
from discopt_mkm.energy import EnergyBalance
from discopt_mkm.examples import adiabatic_cstr, co_oxidation
from discopt_mkm.pfr import solve_pfr


def test_isothermal_pfr_atom_balance():
    m, _ = co_oxidation(T=500.0)
    CO, O2, CO2 = m._by_name["CO"], m._by_name["O2"], m._by_name["CO2"]
    sol = solve_pfr(m, {CO: 1.0, O2: 0.5, CO2: 0.0}, length=4.0, velocity=1.0, cat_density=0.1, nfe=12)
    assert sol.status == "optimal"
    assert 0.0 < sol.conversion(CO) < 1.0
    dCO = 1.0 - sol.outlet(CO)
    dCO2 = sol.outlet(CO2)
    dO2 = 0.5 - sol.outlet(O2)
    assert dCO == pytest.approx(dCO2, abs=1e-6)  # CO -> CO2
    assert dO2 == pytest.approx(0.5 * dCO2, abs=1e-6)  # O2 stoichiometry


def test_nonisothermal_cstr_self_consistent():
    m, reactor, energy = adiabatic_cstr(T_in=500.0)
    A = m._by_name["A"]
    sol = mk.solve_steady_state(m, reactor, energy=energy)
    assert sol.status == "optimal"
    T = sol.temperature()
    conv = 1.0 - sol.gas_concentration(A)
    assert T > 500.0  # exothermic: temperature rises

    # an isothermal solve at the solved temperature gives the same conversion
    m2, r2, _ = adiabatic_cstr(T_in=500.0)
    m2.T = T
    conv2 = 1.0 - mk.solve_steady_state(m2, r2).gas_concentration(m2._by_name["A"])
    assert conv == pytest.approx(conv2, abs=2e-3)


def test_adiabatic_pfr_temperature_rise():
    m, _, _ = adiabatic_cstr(T_in=500.0)
    A, B = m._by_name["A"], m._by_name["B"]
    sol = solve_pfr(
        m, {A: 1.0, B: 0.0}, length=1.0, velocity=1.0, cat_density=0.02, nfe=10,
        energy=EnergyBalance(T_in=500.0),
    )
    assert sol.status == "optimal"
    assert sol.outlet_temperature() > 500.0  # heats up along the reactor
    assert sol.outlet(A) + sol.outlet(B) == pytest.approx(1.0, abs=1e-6)  # A+B conserved
