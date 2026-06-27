"""Branching mechanism: selectivity and degree of selectivity control.

A common surface intermediate ``AO*`` branches to a desired partial-oxidation
product ``P1`` or to an over-oxidation product ``P2``. The branch ratio is set
by the oxygen coverage, which rises with ``P_O2`` — so selectivity falls as
activity rises. The "degree of selectivity control" (the log-sensitivity of
``S = r(P1)/(r(P1)+r(P2))`` to each rate constant) is computed by propagating
the steady-state sensitivities analytically; because ``S`` is homogeneous of
degree zero in the rate constants, these controls must sum to zero.
"""

import pytest

import discopt_mkm as mk
from discopt_mkm.analysis import degree_of_rate_control
from discopt_mkm.analysis import stoichiometry as st
from discopt_mkm.examples import selective_oxidation
from discopt_mkm.numeric import (
    net_rate,
    rate_constants,
    rates_of_progress,
    steady_state_numeric,
)


def _product_rates(m, theta, free, pressures, T):
    conc = {g: float(pressures.get(g, 0.0)) for g in m.gas_species}
    kf, kr = rate_constants(m, T, theta)
    rops = rates_of_progress(m, kf, kr, theta, free, conc)
    P1, P2 = m._by_name["P1"], m._by_name["P2"]
    return net_rate(m, P1, rops), net_rate(m, P2, rops)


def _selectivity(m, P_O2, T=500.0):
    A, O2 = m._by_name["A"], m._by_name["O2"]
    th, fr = steady_state_numeric(m, {A: 1.0, O2: P_O2}, T,
                                  theta0={m._by_name["A*"]: 0.6, m._by_name["O*"]: 0.3,
                                          m._by_name["AO*"]: 0.01})
    rP1, rP2 = _product_rates(m, th, fr, {A: 1.0, O2: P_O2}, T)
    return rP1 / (rP1 + rP2), th


def test_mechanism_is_balanced_and_branches():
    m, _ = selective_oxidation()
    # element- and site-balanced even with abstract elements A, O
    assert st.check_element_balance(m) == []
    assert st.check_site_balance(m) == []
    # the intermediate AO* is consumed by two distinct product-forming steps
    AOs = m._by_name["AO*"]
    consumers = [r for r in m.reactions if AOs in r.reactants]
    assert len(consumers) == 2


def test_selectivity_falls_as_oxygen_pressure_rises():
    m, _ = selective_oxidation()
    S_low, th_low = _selectivity(m, 0.1)
    S_high, th_high = _selectivity(m, 4.0)
    # more oxygen -> more O*, so the over-oxidation branch wins: selectivity drops
    assert S_low > 0.8
    assert S_high < S_low
    assert th_high[m._by_name["O*"]] > th_low[m._by_name["O*"]]


def test_degree_of_selectivity_control_sums_to_zero():
    m, reactor = selective_oxidation(P_O2=0.5)
    A, O2, P1, P2 = (m._by_name[n] for n in ("A", "O2", "P1", "P2"))
    _, th = _selectivity(m, 0.5)

    # near-equilibrium adsorption -> use log coordinates for accurate L3 sensitivities
    sol = mk.solve_steady_state(m, reactor, coordinates="log", theta0=th,
                                log_box=3.0, reg_weight=0.1, active_tol=1e-13)
    assert sol.sensitivities_available

    # standard DRC of the desired product obeys Campbell's sum rule
    drc_P1 = degree_of_rate_control(sol, species=P1)
    assert sum(drc_P1.values()) == pytest.approx(1.0, abs=1e-2)

    # selectivity is homogeneous degree 0 in the rate constants -> controls sum to 0
    S_expr = sol.production_rate_expr(P1) / (
        sol.production_rate_expr(P1) + sol.production_rate_expr(P2)
    )
    sel_control = degree_of_rate_control(sol, rate_expr=S_expr)
    assert sum(sel_control.values()) == pytest.approx(0.0, abs=1e-2)

    by_name = {r.name: v for r, v in sel_control.items()}
    # accelerating the selective branch helps; the over-oxidation branch and
    # feeding more O* both hurt selectivity
    assert by_name["AO* -> P1 (selective)"] > 0.1
    assert by_name["AO* -> P2 (over-oxidation)"] < 0.0
    assert by_name["O2 dissociation"] < 0.0
