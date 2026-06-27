"""Mechanism selection: recover a minimal mechanism from an over-complete set.

The over-complete CO-oxidation candidate has the true three-step Langmuir-
Hinshelwood mechanism plus four decoy steps that carry negligible flux. Three
independent methods should all recover the minimal mechanism:

- flux screen ranked by degree of rate control (:func:`reduce_by_drc`),
- best-subgraph selection that fits the turnover data (:func:`select_subgraph`),
- a symbolic rate law consistent with the apparent orders (:func:`fit_rate_law`).
"""

import pytest

import discopt_mkm as mk
from discopt_mkm.examples import co_oxidation, overcomplete_co_oxidation

MINIMAL = {"CO adsorption", "O2 dissociation", "surface reaction"}
CONDS = [{"CO": 1.0, "O2": 0.5}, {"CO": 2.0, "O2": 0.5}, {"CO": 0.5, "O2": 1.0}]


def _data():
    mt, _ = co_oxidation(500.0)
    CO, O2, CO2 = (mt._by_name[n] for n in ("CO", "O2", "CO2"))
    out = []
    for c in CONDS:
        sol = mk.solve_steady_state(mt, mk.DifferentialReactor({CO: c["CO"], O2: c["O2"], CO2: 0.0}))
        out.append(sol.production_rate(CO2))
    return out


def test_reduce_by_drc_recovers_minimal_mechanism():
    m, _ = overcomplete_co_oxidation(500.0)
    res = mk.reduce_by_drc(m, CONDS, "CO2")
    assert set(res.kept) == MINIMAL
    assert all(name.startswith("[decoy]") for name in res.dropped)
    assert res.max_tof_error < 1e-6
    # the surface reaction is rate-determining among the retained steps
    assert res.drc["surface reaction"] == pytest.approx(1.0, abs=1e-2)


def test_select_subgraph_recovers_minimal_mechanism():
    m, _ = overcomplete_co_oxidation(500.0)
    res = mk.select_subgraph(m, CONDS, "CO2", _data(), engine="greedy")
    assert set(res.selected) == MINIMAL
    assert res.n_steps == 3
    assert res.misfit < 1e-6


def test_milp_flux_selection_recovers_minimal_mechanism():
    # the linear flux-space MILP, with rate-constant capacity bounds, rejects the
    # kinetically incapable decoy routes and recovers the minimal mechanism
    m, _ = overcomplete_co_oxidation(500.0)
    res = mk.select_subgraph(m, CONDS, "CO2", _data(), engine="milp")
    assert set(res.selected) == MINIMAL
    assert res.status == "optimal"
    assert res.misfit < 1e-6


def test_pareto_front_knee_is_three_steps():
    m, _ = overcomplete_co_oxidation(500.0)
    front = mk.pareto_subgraph(m, CONDS, "CO2", _data())
    # misfit collapses to ~0 only once all three real steps are present
    assert front[2][0] > 1e-2
    assert front[3][0] < 1e-6
    assert set(front[3][1]) == MINIMAL


def test_fit_rate_law_recovers_co_inhibited_kinetics():
    # richer pressure grid so the 2-D rate law is identifiable
    conds = [{"CO": co, "O2": o2} for co in (0.5, 1.0, 2.0) for o2 in (0.5, 1.0, 2.0)]
    mt, _ = co_oxidation(500.0)
    CO, O2, CO2 = (mt._by_name[n] for n in ("CO", "O2", "CO2"))
    data = [mk.solve_steady_state(mt, mk.DifferentialReactor({CO: c["CO"], O2: c["O2"], CO2: 0.0})
            ).production_rate(CO2) for c in conds]
    res = mk.fit_rate_law(conds, data, ["CO", "O2"])
    # CO inhibits at high coverage (negative apparent order), O2 promotes
    assert res.orders["CO"] < 0.2
    assert res.orders["O2"] > 0.0
