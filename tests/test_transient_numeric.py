"""Transient coverage integrator (for time-varying inputs / PRBS fitting)."""

import numpy as np
import pytest

import discopt.mkm as mk
from discopt.mkm import numeric
from discopt.mkm.examples import co_oxidation


def test_step_response_converges_to_steady_state():
    m, _ = co_oxidation(500.0)
    CO, O2, CO2 = m._by_name["CO"], m._by_name["O2"], m._by_name["CO2"]
    COs, Os = m._by_name["CO*"], m._by_name["O*"]
    conc = {CO: 1.0, O2: 0.5, CO2: 0.0}

    sol = numeric.integrate_coverages(m, conc, 500.0, np.linspace(0, 5, 60), theta0={})
    theta_ss, _ = numeric.steady_state_numeric(m, conc, 500.0, theta0={a: 0.5 for a in m.adsorbates})
    assert sol.y[0, -1] == pytest.approx(theta_ss[COs], abs=1e-3)
    assert sol.y[1, -1] == pytest.approx(theta_ss[Os], abs=1e-3)


def test_time_varying_input_accepted():
    m, _ = co_oxidation(500.0)
    CO, O2, CO2 = m._by_name["CO"], m._by_name["O2"], m._by_name["CO2"]
    # a smooth time-varying CO pressure
    conc_fn = lambda t: {CO: 1.0 + 0.5 * np.sin(t), O2: 0.5, CO2: 0.0}  # noqa: E731
    sol = numeric.integrate_coverages(m, conc_fn, 500.0, np.linspace(0, 3, 40), theta0={})
    assert sol.success and sol.y.shape == (2, 40)
    assert np.all((sol.y >= -1e-6) & (sol.y <= 1.0 + 1e-6))  # coverages stay physical
