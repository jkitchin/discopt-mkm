"""Apparent orders/Ea, symbolic lumped rate, and rendering."""

import numpy as np
import pytest
import sympy as sp

import discopt_mkm as mk
from discopt_mkm.analysis import apparent_activation_energy, apparent_orders
from discopt_mkm.examples import co_oxidation


# ---------------------------------------------------------------- apparent
def _tof(P, T=500.0):
    m, _ = co_oxidation(T)
    CO, O2, CO2 = m._by_name["CO"], m._by_name["O2"], m._by_name["CO2"]
    r = mk.DifferentialReactor({CO: P[0], O2: P[1], CO2: 0.0})
    return mk.solve_steady_state(m, r).production_rate(CO2)


def test_apparent_orders_match_finite_difference():
    m, reactor = co_oxidation(500.0)
    CO, O2, CO2 = m._by_name["CO"], m._by_name["O2"], m._by_name["CO2"]
    sol = mk.solve_steady_state(m, reactor)
    orders = apparent_orders(sol, CO2)
    base = [1.0, 0.5]
    d = 0.01
    fd_CO = (np.log(_tof([base[0] * (1 + d), base[1]])) - np.log(_tof([base[0] * (1 - d), base[1]]))) / (2 * d)
    fd_O2 = (np.log(_tof([base[0], base[1] * (1 + d)])) - np.log(_tof([base[0], base[1] * (1 - d)]))) / (2 * d)
    assert orders[CO] == pytest.approx(fd_CO, abs=2e-3)
    assert orders[O2] == pytest.approx(fd_O2, abs=2e-3)


def test_apparent_activation_energy_matches_finite_difference():
    m, reactor = co_oxidation(500.0)
    CO2 = m._by_name["CO2"]
    sol = mk.solve_steady_state(m, reactor)
    ad = apparent_activation_energy(sol, CO2)
    dT = 1.0
    fd = 8.617e-5 * 500**2 * (np.log(_tof([1.0, 0.5], 500 + dT)) - np.log(_tof([1.0, 0.5], 500 - dT))) / (2 * dT)
    assert ad == pytest.approx(fd, abs=2e-3)
    # apparent barrier differs from the elementary surface barrier (0.7 eV)
    assert ad != pytest.approx(0.7, abs=1e-2)


# ---------------------------------------------------------------- symbolic
def test_lumped_rate_is_langmuir_hinshelwood():
    m = mk.Model("lh", T=500, R=8.314)
    s = m.site("s", density=1.0)
    A, B = m.gas("A"), m.gas("B")
    As = m.adsorbate("A*", site=s)
    m.step(A + s >> As, Keq=5.0, equilibrated=True)
    m.step(As >> B + s, kf=3.0, irreversible=True)
    rate, syms = mk.lumped_rate_expression(m, B)
    PA, K0, kf1 = syms["P"][A], syms["Keq"][m.reactions[0]], syms["kf"][m.reactions[1]]
    assert sp.simplify(rate - kf1 * K0 * PA / (1 + K0 * PA)) == 0


def test_lumped_rate_requires_quasi_equilibrium():
    m, _ = co_oxidation()
    with pytest.raises(ValueError, match="quasi-equilibrium"):
        mk.lumped_rate_expression(m, m._by_name["CO2"])


# ---------------------------------------------------------------- rendering
def test_reaction_rendering():
    m, _ = co_oxidation()
    r_rev = m.reactions[0]  # CO + Pt <=> CO*
    assert r_rev.to_latex() == r"$CO + Pt \rightleftharpoons CO{}^{\ast}$"
    assert "&#8652;" in r_rev.to_html()  # reversible harpoon
    assert "CO<sup>&lowast;</sup>" in r_rev.to_html()


def test_model_and_solution_html():
    m, reactor = co_oxidation()
    html = m.to_html()
    assert "co_oxidation" in html and "<table" in html and "reversible" in html
    sol = mk.solve_steady_state(m, reactor)
    shtml = sol.to_html()
    assert "coverages" in shtml and "rates of progress" in shtml
