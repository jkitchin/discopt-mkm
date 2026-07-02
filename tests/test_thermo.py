"""Thermodynamic-consistency tests: the derived reverse rate satisfies k_f/k_r = K_eq."""

import numpy as np
import pytest

import discopt.mkm as mk
from discopt.mkm.analysis.sensitivity import evaluate_expression
from discopt.mkm.examples import co_oxidation
from discopt.mkm.kinetics import k_forward, k_reverse
from discopt.mkm.thermo import K_eq


@pytest.mark.parametrize("T", [400.0, 500.0, 700.0])
def test_kf_over_kr_equals_keq(T):
    """For every step and temperature, k_f / k_r must equal K_eq (by construction)."""
    m, reactor = co_oxidation(T=T)
    sol = mk.solve_steady_state(m, reactor)
    model = sol.dm_model
    for rxn in m.reactions:
        kf = evaluate_expression(k_forward(rxn, sol.T_param, m.R), sol.result, model)
        kr = evaluate_expression(k_reverse(rxn, sol.T_param, m.R, m.Tref), sol.result, model)
        keq = evaluate_expression(K_eq(rxn, sol.T_param, m.R, m.Tref), sol.result, model)
        assert kf / kr == pytest.approx(keq, rel=1e-9)


def test_temperature_changes_rate():
    """Higher temperature should change the turnover frequency (kinetics are T-coupled)."""
    tofs = []
    for T in (450.0, 550.0):
        m, reactor = co_oxidation(T=T)
        sol = mk.solve_steady_state(m, reactor)
        tofs.append(sol.production_rate(m._by_name["CO2"]))
    assert not np.isclose(tofs[0], tofs[1])
