"""End-to-end steady-state, thermodynamic-consistency, DRC and transient tests.

These require the discopt package (and its pure-JAX IPM backend).
"""

import numpy as np
import pytest

import discopt.mkm as mk
from discopt.mkm.analysis import degree_of_rate_control, thermo_rate_control
from discopt.mkm.examples import co_oxidation


# ground truth from an independent scipy fsolve of the same mechanism
SS_REF = {"CO*": 0.58517, "O*": 0.23136, "free": 0.18347, "tof": 1.19016}


@pytest.fixture(scope="module")
def solved():
    m, reactor = co_oxidation(T=500.0)
    sol = mk.solve_steady_state(m, reactor)
    return m, sol


def test_status_and_sensitivities(solved):
    _, sol = solved
    assert sol.status == "optimal"
    assert sol.sensitivities_available


def test_coverages_match_reference(solved):
    m, sol = solved
    assert sol.coverage(m._by_name["CO*"]) == pytest.approx(SS_REF["CO*"], abs=1e-3)
    assert sol.coverage(m._by_name["O*"]) == pytest.approx(SS_REF["O*"], abs=1e-3)
    assert sol.free_coverage(m._by_name["Pt"]) == pytest.approx(SS_REF["free"], abs=1e-3)


def test_site_balance_closes(solved):
    m, sol = solved
    total = sum(sol.coverage(a) for a in m.adsorbates) + sol.free_coverage(m._by_name["Pt"])
    assert total == pytest.approx(1.0, abs=1e-6)


def test_turnover_frequency(solved):
    m, sol = solved
    assert sol.production_rate(m._by_name["CO2"]) == pytest.approx(SS_REF["tof"], rel=1e-3)


def test_drc_sums_to_one_and_identifies_rds(solved):
    m, sol = solved
    X = degree_of_rate_control(sol, species=m._by_name["CO2"])
    # Campbell's theorem: the degrees of rate control sum to 1
    assert sum(X.values()) == pytest.approx(1.0, abs=1e-3)
    # the surface reaction is rate-determining here
    surface = next(r for r in m.reactions if r.name == "surface reaction")
    assert X[surface] == pytest.approx(1.0, abs=1e-2)


def test_drc_matches_finite_difference(solved):
    m, sol = solved
    CO2 = m._by_name["CO2"]
    X = degree_of_rate_control(sol, species=CO2)
    r0 = sol.production_rate(CO2)

    def tof_scaled(idx, factor):
        mm, rr = co_oxidation(T=500.0)
        mm.reactions[idx].A *= factor
        s = mk.solve_steady_state(mm, rr)
        return s.production_rate(mm._by_name["CO2"])

    d = 0.01
    for i, rxn in enumerate(m.reactions):
        fd = (tof_scaled(i, 1 + d) - tof_scaled(i, 1 - d)) / (2 * d * r0)
        assert X[rxn] == pytest.approx(fd, abs=2e-3)


def test_thermo_rate_control_available(solved):
    m, sol = solved
    T = thermo_rate_control(sol, species=m._by_name["CO2"])
    # every species has a TRC entry, all finite
    assert set(T.keys()) == set(m.species)
    assert all(np.isfinite(v) for v in T.values())


def test_transient_converges_to_steady_state():
    m, reactor = co_oxidation(T=500.0)
    COs, Os = m._by_name["CO*"], m._by_name["O*"]
    tr = mk.solve_transient(m, reactor, t_span=(0.0, 5.0), theta0={COs: 0.0, Os: 0.0}, nfe=25, ncp=3)
    assert tr.status == "optimal"
    assert tr.final_coverage(COs) == pytest.approx(SS_REF["CO*"], abs=1e-2)
    assert tr.final_coverage(Os) == pytest.approx(SS_REF["O*"], abs=1e-2)
