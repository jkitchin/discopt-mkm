"""Quasi-equilibrium approximation: equilibrated steps + extent/equilibrium reformulation."""

import pytest

import discopt_mkm as mk
from discopt_mkm import numeric
from discopt_mkm.analysis import degree_of_rate_control
from discopt_mkm.examples import water_gas_shift, water_gas_shift_qea


def test_langmuir_hinshelwood_closed_form():
    """A + * <=> A* (equilibrated), A* -> B + * (RDS) gives the analytic LHHW rate."""
    K1, k2, pA = 5.0, 3.0, 2.0
    m = mk.Model("lh", T=500.0, R=8.314)
    s = m.site("s", density=1.0)
    A, B = m.gas("A"), m.gas("B")
    As = m.adsorbate("A*", site=s)
    r1 = m.step(A + s >> As, Keq=K1, equilibrated=True)
    r2 = m.step(As >> B + s, kf=k2, irreversible=True)
    sol = mk.solve_steady_state(m, mk.DifferentialReactor({A: pA, B: 0.0}))

    assert sol.coverage(As) == pytest.approx(K1 * pA / (1 + K1 * pA), rel=1e-6)
    assert sol.production_rate(B) == pytest.approx(k2 * K1 * pA / (1 + K1 * pA), rel=1e-6)
    # the equilibrated step's extent equals the through-flux
    assert sol.rate_of_progress(r1) == pytest.approx(sol.production_rate(B), rel=1e-6)
    # equilibrated step has zero kinetic DRC; the RDS carries all of it
    X = degree_of_rate_control(sol, species=B)
    assert X[r1] == pytest.approx(0.0, abs=1e-9)
    assert X[r2] == pytest.approx(1.0, abs=1e-3)


def test_arrow_for_equilibrated_step():
    m = mk.Model("eq", T=500.0)
    s = m.site("s", density=1.0)
    A = m.gas("A")
    As = m.adsorbate("A*", site=s)
    r = m.step(A + s >> As, Keq=5.0, equilibrated=True)
    assert r.equilibrated and "<=>" in r.equation()


def test_qea_reproduces_full_ssa_rate():
    """WGS QEA (linear, no warm start) matches the full-SSA log solve rate."""
    mf, rf = water_gas_shift(T=480.0)
    th0, _ = numeric.steady_state_numeric(mf, rf.pressures, 480.0, theta0={a: 1e-3 for a in mf.adsorbates})
    full = mk.solve_steady_state(mf, rf, coordinates="log", theta0=th0, log_box=8.0)
    rH2_full = full.production_rate(mf._by_name["H2"])

    m, reactor = water_gas_shift_qea(T=480.0)
    sol = mk.solve_steady_state(m, reactor)  # linear coordinates, no warm start
    assert sol.status == "optimal"
    rH2_qea = sol.production_rate(m._by_name["H2"])
    assert rH2_qea == pytest.approx(rH2_full, rel=2e-3)


def test_qea_drc_matches_paper_with_small_active_tol():
    """With active_tol below the tiny coverages, QEA DRC reproduces the paper."""
    m, reactor = water_gas_shift_qea(T=480.0)
    sol = mk.solve_steady_state(m, reactor, active_tol=1e-13)
    X = degree_of_rate_control(sol, species=m._by_name["H2"])
    kinetic = {r.name: x for r, x in X.items() if not r.equilibrated}
    assert sum(X.values()) == pytest.approx(1.0, abs=1e-2)
    vals = sorted(kinetic.values(), reverse=True)
    assert vals[0] == pytest.approx(0.884, abs=5e-3)
    assert vals[1] == pytest.approx(0.116, abs=5e-3)
    # equilibrated steps contribute exactly zero
    assert all(X[r] == 0.0 for r in m.reactions if r.equilibrated)


def test_equilibrated_steps_rejected_in_transient():
    m, reactor = water_gas_shift_qea(T=480.0)
    with pytest.raises(ValueError, match="steady-state"):
        mk.solve_transient(m, reactor, t_span=(0.0, 1.0), nfe=5)
