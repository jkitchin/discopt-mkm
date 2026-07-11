"""Transient (time-series) parameter estimation: fit_kinetics_transient."""

import numpy as np
import pytest

import discopt.mkm as mk
from discopt.mkm import numeric
from discopt.mkm.examples import co_oxidation, water_gas_shift_qea
from discopt.mkm.transient_fit import _element_boundaries, _step_input


# ---------------------------------------------------------------------------
# synthetic data helper: integrate the true model piecewise and sample the
# CO2 net production rate (or a coverage) at irregular measurement times
# ---------------------------------------------------------------------------

def _synthetic_run(m, co_input, T, t_meas, response="rate"):
    CO, O2, CO2 = m._by_name["CO"], m._by_name["O2"], m._by_name["CO2"]
    COs, Os = m._by_name["CO*"], m._by_name["O*"]
    site = m.sites[0]
    edges, vals = co_input
    tf = float(np.max(t_meas))
    knots = np.concatenate([edges[(edges > 0) & (edges < tf)], [0.0, tf]])
    knots = np.unique(knots)

    tg, yg, theta = [], [], {COs: 0.0, Os: 0.0}
    for a, b in zip(knots[:-1], knots[1:]):
        i = min(np.searchsorted(edges, 0.5 * (a + b), "right") - 1, len(vals) - 1)
        conc = {CO: float(vals[i]), O2: 0.5, CO2: 0.0}
        sol = numeric.integrate_coverages(m, conc, T, np.linspace(a, b, 200), theta0=theta)
        for k in range(len(sol.t)):
            th = {COs: sol.y[0, k], Os: sol.y[1, k]}
            fr = {site: 1 - th[COs] - th[Os]}
            if response == "rate":
                yg.append(numeric.turnover_frequency(m, CO2, th, fr, conc, T))
            else:
                yg.append(th[COs])
            tg.append(sol.t[k])
        theta = {COs: sol.y[0, -1], Os: sol.y[1, -1]}
    tg, yg = np.array(tg), np.array(yg)
    return np.interp(t_meas, tg, yg)


# ---------------------------------------------------------------------------
# fast structural tests (no NLP solve)
# ---------------------------------------------------------------------------

def test_element_boundaries_align_switches():
    eb = _element_boundaries(0.0, 2.0, [0.7, 1.3], nfe=10)
    # switch times are exact boundaries
    for sw in (0.7, 1.3):
        assert np.any(np.isclose(eb, sw))
    # refinement: no element wider than span/nfe
    assert np.max(np.diff(eb)) <= 2.0 / 10 + 1e-12
    assert np.all(np.diff(eb) > 0)
    assert eb[0] == 0.0 and eb[-1] == 2.0
    # boundary-layer split right after t0: first element much narrower than h_max
    assert eb[1] - eb[0] < 0.1 * (2.0 / 10)
    # near-coincident switches are deduplicated (no sliver elements)
    eb2 = _element_boundaries(0.0, 2.0, [0.7, 0.7 + 1e-15], nfe=10)
    assert np.all(np.diff(eb2) > 1e-12)


def test_step_input_shapes_and_lookup():
    class G:
        name = "CO"

    g = G()
    # len(times) == len(values): last value held to the end
    sw, f = _step_input((np.array([0.0, 1.0]), np.array([2.0, 3.0])), g, 0.0, 2.0)
    assert list(sw) == [1.0]
    assert f(0.5) == 2.0 and f(1.0) == 3.0 and f(1.7) == 3.0  # right-continuous
    assert np.allclose(f(np.array([0.2, 1.2])), [2.0, 3.0])   # vectorized
    # PRBS convention: one trailing edge
    sw, f = _step_input((np.array([0.0, 1.0, 2.0]), np.array([2.0, 3.0])), g, 0.0, 2.0)
    assert list(sw) == [1.0] and f(1.5) == 3.0
    # constants
    sw, f = _step_input(0.5, g, 0.0, 2.0)
    assert len(sw) == 0 and f(1.0) == 0.5

    with pytest.raises(ValueError, match="strictly increasing"):
        _step_input((np.array([0.0, 0.0]), np.array([1.0, 2.0])), g, 0.0, 2.0)
    with pytest.raises(ValueError, match="times.*values|values.*times"):
        _step_input((np.array([0.0]), np.array([1.0, 2.0])), g, 0.0, 2.0)


def test_run_validation_errors():
    m, _ = co_oxidation(T=500.0)
    CO, O2, CO2 = m._by_name["CO"], m._by_name["O2"], m._by_name["CO2"]
    surf = m.reactions[2]
    fit = [mk.FitParam(surf, "A", lb=1e6, ub=1e10)]
    P = {CO: 1.0, O2: 0.5, CO2: 0.0}

    def run(**kw):
        base = dict(response=CO2, t=[0.5, 1.0], y=[1.0, 1.0], T=500.0, pressures=P)
        base.update(kw)
        return mk.TransientRun(**base)

    with pytest.raises(ValueError, match="equal-length"):
        mk.fit_kinetics_transient(m, [run(y=[1.0])], fit)
    with pytest.raises(ValueError, match="within t_span"):
        mk.fit_kinetics_transient(m, [run(t_span=(0.0, 0.6))], fit)
    with pytest.raises(ValueError, match="sigma must be positive"):
        mk.fit_kinetics_transient(m, [run(sigma=0.0)], fit)
    with pytest.raises(ValueError, match="labels must be unique"):
        mk.fit_kinetics_transient(m, [run(label="a"), run(label="a")], fit)
    with pytest.raises(ValueError, match="at least one run"):
        mk.fit_kinetics_transient(m, [], fit)
    # a response species no reaction produces or consumes is rejected
    m2, _ = co_oxidation(T=500.0)
    spectator = m2.gas("Ar")
    fit2 = [mk.FitParam(m2.reactions[2], "A", lb=1e6, ub=1e10)]
    r2 = mk.TransientRun(response=spectator, t=[0.5], y=[0.0], T=500.0,
                         pressures={m2._by_name["CO"]: 1.0, m2._by_name["O2"]: 0.5})
    with pytest.raises(ValueError, match="not net-produced"):
        mk.fit_kinetics_transient(m2, [r2], fit2)


def test_equilibrated_steps_rejected():
    m, _ = water_gas_shift_qea()
    CO2 = m._by_name["CO2"]
    fit = [mk.FitParam(m.reactions[3], "kf", lb=1.0, ub=1e6)]
    r = mk.TransientRun(response=CO2, t=[1.0], y=[0.0], T=480.0, pressures={})
    with pytest.raises(NotImplementedError, match="equilibrated"):
        mk.fit_kinetics_transient(m, [r], fit)


# ---------------------------------------------------------------------------
# round-trip recovery tests (full collocation NLP solves: slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_recover_A_single_run():
    """Recover the surface-step pre-exponential from one step-response run."""
    m, _ = co_oxidation(T=500.0)
    CO, O2, CO2 = m._by_name["CO"], m._by_name["O2"], m._by_name["CO2"]
    surf = m.reactions[2]  # A=1e8 (truth)

    co_input = (np.array([0.0, 1.0]), np.array([1.0, 0.4]))
    rng = np.random.default_rng(0)
    t_meas = np.sort(rng.uniform(0.05, 2.0, 40))
    y = _synthetic_run(m, co_input, 500.0, t_meas)

    run = mk.TransientRun(
        response=CO2, t=t_meas, y=y, T=500.0,
        pressures={CO: co_input, O2: 0.5, CO2: 0.0},
        sigma=0.01, t_span=(0.0, 2.0),
    )
    fit = [mk.FitParam(surf, "A", lb=1e6, ub=1e10, init=3e7)]
    res = mk.fit_kinetics_transient(m, [run], fit, nfe=20)

    A_key = next(k for k in res.parameters if k.startswith("A_"))
    assert res.parameters[A_key] == pytest.approx(1e8, rel=0.02)
    lo, hi = res.confidence_intervals[A_key]
    assert lo < 1e8 < hi
    assert res.n_observations == 40
    # fitted model reproduces the data and the trajectories are on the grid
    assert np.max(np.abs(res.predictions["run0"] - y)) < 0.01
    assert res.times["run0"].shape == res.trajectories["run0"]["CO*"].shape
    assert res.trajectories["run0"]["CO*"][0] == 0.0  # imposed initial condition


@pytest.mark.slow
def test_multi_run_shared_A_and_Ea():
    """Two runs at different temperatures identify A and Ea jointly
    (single-temperature transient data cannot separate them)."""
    m, _ = co_oxidation(T=500.0)
    CO, O2, CO2 = m._by_name["CO"], m._by_name["O2"], m._by_name["CO2"]
    surf = m.reactions[2]  # truth: A=1e8, Ea=0.7

    co_input = (np.array([0.0, 1.0]), np.array([1.0, 0.4]))
    rng = np.random.default_rng(1)
    runs = []
    for T in (480.0, 520.0):
        t_meas = np.sort(rng.uniform(0.05, 2.0, 30))
        y = _synthetic_run(m, co_input, T, t_meas)
        runs.append(mk.TransientRun(
            response=CO2, t=t_meas, y=y, T=T,
            pressures={CO: co_input, O2: 0.5, CO2: 0.0},
            sigma=0.01, t_span=(0.0, 2.0),
        ))
    fit = [
        mk.FitParam(surf, "A", lb=1e6, ub=1e10, init=3e7),
        mk.FitParam(surf, "Ea", lb=0.3, ub=1.2, init=0.6),
    ]
    res = mk.fit_kinetics_transient(m, runs, fit, nfe=16)

    A_key = next(k for k in res.parameters if k.startswith("A_"))
    Ea_key = next(k for k in res.parameters if k.startswith("Ea_"))
    assert res.parameters[Ea_key] == pytest.approx(0.7, abs=0.02)
    assert res.parameters[A_key] == pytest.approx(1e8, rel=0.25)
    assert res.n_observations == 60
    assert set(res.predictions) == {"run0", "run1"}


@pytest.mark.slow
def test_coverage_response():
    """Fitting a measured *coverage* trajectory recovers the constant. The
    coverage response contains no fitted constant explicitly; its entire
    sensitivity flows through the trajectory, so this exercises the
    implicit-function-theorem part of the FIM."""
    m, _ = co_oxidation(T=500.0)
    CO, O2, CO2 = m._by_name["CO"], m._by_name["O2"], m._by_name["CO2"]
    COs = m._by_name["CO*"]
    surf = m.reactions[2]

    co_input = (np.array([0.0, 1.0]), np.array([1.0, 0.4]))
    rng = np.random.default_rng(2)
    t_meas = np.sort(rng.uniform(0.05, 2.0, 30))
    y = _synthetic_run(m, co_input, 500.0, t_meas, response="coverage")

    run = mk.TransientRun(
        response=COs, t=t_meas, y=y, T=500.0,
        pressures={CO: co_input, O2: 0.5, CO2: 0.0},
        sigma=0.005, t_span=(0.0, 2.0),
    )
    fit = [mk.FitParam(surf, "A", lb=1e6, ub=1e10, init=3e7)]
    res = mk.fit_kinetics_transient(m, [run], fit, nfe=16)

    A_key = next(k for k in res.parameters if k.startswith("A_"))
    assert res.parameters[A_key] == pytest.approx(1e8, rel=0.05)
    # the implicit trajectory sensitivity makes the FIM non-degenerate even
    # though the response expression contains no fitted variable
    assert res.std_errors[A_key] > 0
    lo, hi = res.confidence_intervals[A_key]
    assert lo < res.parameters[A_key] < hi and np.isfinite(lo) and np.isfinite(hi)
