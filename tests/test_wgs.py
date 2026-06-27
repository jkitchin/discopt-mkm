"""Water-gas shift: explicit-rate steps + log-coverage DRC reproduce the paper.

Reproduces the Case II degree of rate control from Yang, Achar & Kitchin,
AIChE J. 68(6):e17653 (2022): the net rate is a ~12-order cancellation, so the
DRC must be propagated analytically (finite differences cannot compute it).
"""

import pytest

import discopt_mkm as mk
from discopt_mkm import numeric
from discopt_mkm.analysis import degree_of_rate_control
from discopt_mkm.examples import water_gas_shift


@pytest.fixture(scope="module")
def wgs_solution():
    m, reactor = water_gas_shift(T=480.0)
    seed = {a: 1e-3 for a in m.adsorbates}
    theta0, _ = numeric.steady_state_numeric(m, reactor.pressures, 480.0, theta0=seed)
    sol = mk.solve_steady_state(m, reactor, coordinates="log", theta0=theta0, log_box=8.0)
    return m, sol


def test_net_rate_matches_paper(wgs_solution):
    m, sol = wgs_solution
    assert sol.status == "optimal"
    rH2 = sol.production_rate(m._by_name["H2"])
    assert rH2 == pytest.approx(1.4467e-6, rel=1e-3)


def test_drc_sums_to_one_with_two_controlling_steps(wgs_solution):
    m, sol = wgs_solution
    assert sol.sensitivities_available
    X = degree_of_rate_control(sol, species=m._by_name["H2"])
    assert sum(X.values()) == pytest.approx(1.0, abs=1e-3)

    by_value = sorted(X.values(), reverse=True)
    assert by_value[0] == pytest.approx(0.884, abs=5e-3)  # the rate-controlling step
    assert by_value[1] == pytest.approx(0.116, abs=5e-3)  # the secondary step


def test_explicit_rate_thermodynamic_consistency(wgs_solution):
    m, sol = wgs_solution
    # explicit-rate steps: k_f / k_r == K_eq by construction (kr = kf / Keq)
    from discopt_mkm.analysis.sensitivity import evaluate_expression
    from discopt_mkm.kinetics import k_forward, k_reverse

    for rxn in m.reactions:
        kf = evaluate_expression(k_forward(rxn, sol.T_param, m.R), sol.result, sol.dm_model)
        kr = evaluate_expression(k_reverse(rxn, sol.T_param, m.R, m.Tref), sol.result, sol.dm_model)
        assert kf / kr == pytest.approx(rxn.Keq, rel=1e-9)
