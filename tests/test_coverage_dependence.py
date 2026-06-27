"""Coverage-dependent energetics: lateral interactions, H-callable, BEP alpha."""

import pytest

import discopt_mkm as mk
from discopt_mkm.analysis import degree_of_rate_control
from discopt_mkm.examples import co_oxidation


def test_drc_still_sums_to_one_with_interactions():
    """Lateral interactions + a BEP barrier shift must not break Campbell's sum rule."""
    m, reactor = co_oxidation(T=500.0)
    COs, Os, CO2 = m._by_name["CO*"], m._by_name["O*"], m._by_name["CO2"]
    m.interaction(COs, COs, 0.3)  # repulsive self-interactions
    m.interaction(Os, Os, 0.2)
    m.interaction(COs, Os, 0.1)  # cross term
    m.reactions[2].alpha = 0.5  # BEP: barrier sees half the coverage-energy shift

    sol = mk.solve_steady_state(m, reactor)
    assert sol.status == "optimal"
    X = degree_of_rate_control(sol, species=CO2)
    assert sum(X.values()) == pytest.approx(1.0, abs=1e-3)


def _adsorption_model(eps):
    """Strong reversible adsorption A + * <=> A*; optional self-repulsion eps."""
    m = mk.Model("ads", T=400.0, R=8.314)
    s = m.site("s", density=1.0)
    A = m.gas("A")
    As = m.adsorbate("A*", site=s, H=-20000.0, S=0.0)  # strong binding -> near saturation
    m.step(A + s >> As, A=1e3, Ea=0.0)  # reversible; reverse from thermo
    if eps:
        m.interaction(As, As, eps)
    return m, mk.DifferentialReactor({A: 1.0}), As


def test_repulsion_lowers_saturation_coverage():
    m0, r0, As0 = _adsorption_model(eps=0.0)
    m1, r1, As1 = _adsorption_model(eps=20000.0)  # strong repulsion (J/mol)
    theta_no_rep = mk.solve_steady_state(m0, r0).coverage(As0)
    theta_rep = mk.solve_steady_state(m1, r1).coverage(As1)

    assert theta_no_rep > 0.95  # strong binding nearly saturates the surface
    assert theta_rep < theta_no_rep - 0.1  # repulsion measurably lowers it


def test_H_callable_matches_declarative_interaction():
    """A callable H(theta) reproduces the same coverage as the equivalent interaction."""
    # declarative interaction
    m_dec, r_dec, As_dec = _adsorption_model(eps=8000.0)
    theta_dec = mk.solve_steady_state(m_dec, r_dec).coverage(As_dec)

    # same physics written as a callable H(theta) = H0 + eps*theta_A*
    m = mk.Model("ads_cb", T=400.0, R=8.314)
    s = m.site("s", density=1.0)
    A = m.gas("A")
    As = m.adsorbate("A*", site=s, S=0.0)
    As.H = lambda th, _As=As: -20000.0 + 8000.0 * th[_As]
    m.step(A + s >> As, A=1e3, Ea=0.0)
    theta_cb = mk.solve_steady_state(m, mk.DifferentialReactor({A: 1.0})).coverage(As)

    assert theta_cb == pytest.approx(theta_dec, rel=1e-6)
