"""Stoichiometric structure: routes, independence, conservation laws."""

import numpy as np
import pytest

from discopt_mkm.analysis import stoichiometry as st
from discopt_mkm.examples import co_oxidation, water_gas_shift


def test_wgs_route_is_overall_reaction():
    m, _ = water_gas_shift()
    routes = st.reaction_routes(m)
    assert len(routes) == 1
    _, overall = routes[0]
    by_name = {s.name: round(v) for s, v in overall.items()}
    # CO + H2O -> CO2 + H2  (sign: reactants negative)
    assert by_name == {"CO": -1, "H2O": -1, "CO2": 1, "H2": 1}


def test_co_oxidation_route():
    m, _ = co_oxidation()
    routes = st.reaction_routes(m)
    assert len(routes) == 1
    _, overall = routes[0]
    by_name = {s.name: round(v) for s, v in overall.items()}
    # 2 CO + O2 -> 2 CO2
    assert by_name == {"CO": -2, "O2": -1, "CO2": 2}


def test_independent_reaction_count():
    m, _ = water_gas_shift()
    assert st.n_independent_reactions(m) == len(st.independent_reactions(m))
    assert st.n_independent_reactions(m) == 7  # all 7 WGS steps are independent


def test_site_balance_is_recovered_and_valid():
    m, _ = co_oxidation()
    nu, species, _ = st.stoichiometric_matrix(m)
    laws = st.site_conservation_laws(m)
    assert len(laws) == len(m.sites)
    idx = {sp: i for i, sp in enumerate(species)}
    for law in laws:
        v = np.zeros(len(species))
        for sp, c in law.items():
            v[idx[sp]] = c
        # a true conservation law: every reaction leaves it unchanged
        assert np.allclose(nu.T @ v, 0.0, atol=1e-9)
    # the recovered Pt site balance: Pt(free) + CO* + O*
    pt_law = laws[0]
    assert {s.name for s in pt_law} == {"Pt", "CO*", "O*"}


def test_conservation_law_count():
    m, _ = water_gas_shift()
    # n_species - rank = site (1) + elements C, H, O (3) = 4
    assert st.n_conservation_laws(m) == 4
    assert len(st.conservation_laws(m)) == 4
