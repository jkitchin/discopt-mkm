"""Parameter-estimation round-trip: recover known constants from synthetic data."""

import pytest

import discopt.mkm as mk
from discopt.mkm.examples import co_oxidation

TS = [480.0, 500.0, 520.0]


def _synthetic_tof():
    data = []
    for T in TS:
        m, r = co_oxidation(T=T)
        sol = mk.solve_steady_state(m, r)
        data.append(sol.production_rate(m._by_name["CO2"]))
    return data


def test_recover_A_and_Ea():
    data = _synthetic_tof()  # generated with surface step A=1e8, Ea=0.7

    m, _ = co_oxidation(T=500.0)
    CO, O2, CO2 = m._by_name["CO"], m._by_name["O2"], m._by_name["CO2"]
    surf = m.reactions[2]
    obs = [
        mk.Observation(response=CO2, value=v, T=T, pressures={CO: 1.0, O2: 0.5, CO2: 0.0}, sigma=0.01)
        for T, v in zip(TS, data)
    ]
    fit = [
        mk.FitParam(surf, "A", lb=1e6, ub=1e10, init=3e7),
        mk.FitParam(surf, "Ea", lb=0.3, ub=1.2, init=0.5),
    ]
    res = mk.fit_kinetics(m, obs, fit)

    A_key = next(k for k in res.parameters if k.startswith("A_"))
    Ea_key = next(k for k in res.parameters if k.startswith("Ea_"))
    assert res.parameters[Ea_key] == pytest.approx(0.7, abs=1e-2)
    assert res.parameters[A_key] == pytest.approx(1e8, rel=0.1)
    assert res.objective < 1e-10  # zero-noise data -> near-perfect fit
    # confidence intervals bracket the truth
    lo, hi = res.confidence_intervals[Ea_key]
    assert lo <= 0.7 <= hi
