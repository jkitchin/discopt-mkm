"""Irreversible steps: kr == 0, no K_eq expression, no product thermo needed."""

import pytest

import discopt_mkm as mk
from discopt_mkm import numeric
from discopt_mkm.analysis import degree_of_rate_control


def _chain(explicit=True, irreversible=True, keq=None):
    """A + * -> A* -> B + * on one site (irreversible chain by default)."""
    m = mk.Model("chain", T=500.0, R=8.314)
    s = m.site("s", density=1.0)
    A = m.gas("A")
    B = m.gas("B")
    As = m.adsorbate("A*", site=s)
    if explicit:
        r1 = m.step(A + s >> As, kf=100.0, Keq=keq, irreversible=irreversible)
        r2 = m.step(As >> B + s, kf=5.0, Keq=keq, irreversible=irreversible)
    else:
        r1 = m.step(A + s >> As, A=1e2, Ea=0.0, irreversible=irreversible)
        r2 = m.step(As >> B + s, A=1e5, Ea=5000.0, irreversible=irreversible)
    reactor = mk.DifferentialReactor({A: 1.0, B: 0.0})
    return m, reactor, (A, B, As, s, r1, r2)


def test_flag_and_equation_arrow():
    m, _, (_, _, _, _, r1, r2) = _chain()
    assert r1.irreversible and r2.irreversible
    assert "->" in r1.equation() and "<=>" not in r1.equation()


def test_no_keq_param_and_zero_reverse():
    m, _, (_, _, _, _, r1, r2) = _chain()
    kf, kr = numeric.rate_constants(m, 500.0)
    assert kr[r1] == 0.0 and kr[r2] == 0.0
    # after wiring, irreversible explicit steps have no Keq_param
    import discopt.modeling as dm

    scratch = dm.Model("s")
    m.wire_parameters(scratch)
    assert r1.Keq_param is None and r2.Keq_param is None


def test_rate_of_progress_is_forward_only():
    m, reactor, (A, B, As, s, r1, r2) = _chain()
    sol = mk.solve_steady_state(m, reactor)
    assert sol.status == "optimal"
    # r2 = kf2 * theta_A* exactly (no reverse term)
    assert sol.rate_of_progress(r2) == pytest.approx(5.0 * sol.coverage(As), rel=1e-6)


def test_arrhenius_irreversible_needs_no_product_thermo():
    # species carry no thermodynamics; a reversible Arrhenius step would need
    # product free energies for K_eq, but an irreversible one must not.
    m, reactor, _ = _chain(explicit=False, irreversible=True)
    sol = mk.solve_steady_state(m, reactor)
    assert sol.status == "optimal"


def test_drc_sums_to_one_with_irreversible_steps():
    m, reactor, (A, B, As, s, r1, r2) = _chain()
    sol = mk.solve_steady_state(m, reactor)
    X = degree_of_rate_control(sol, species=B)
    assert sum(X.values()) == pytest.approx(1.0, abs=1e-3)


def test_irreversible_matches_huge_keq():
    m_irr, reactor_irr, (A1, B1, As1, s1, _, _) = _chain(irreversible=True)
    m_rev, reactor_rev, (A2, B2, As2, s2, _, _) = _chain(irreversible=False, keq=1e30)
    si = mk.solve_steady_state(m_irr, reactor_irr)
    sr = mk.solve_steady_state(m_rev, reactor_rev)
    assert si.coverage(As1) == pytest.approx(sr.coverage(As2), rel=1e-6)
