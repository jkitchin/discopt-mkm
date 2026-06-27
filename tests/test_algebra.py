"""Operator-algebra and stoichiometry tests (no solver needed)."""

import discopt_mkm as mk


def build():
    m = mk.Model("t", T=500)
    s = m.site("Pt", density=1.0)
    CO = m.gas("CO")
    CO2 = m.gas("CO2")
    COs = m.adsorbate("CO*", site=s)
    Os = m.adsorbate("O*", site=s)
    return m, s, CO, CO2, COs, Os


def test_simple_adsorption_stoichiometry():
    _, s, CO, _, COs, _ = build()
    r = CO + s >> COs
    assert r.reactants == {CO: 1.0, s: 1.0}
    assert r.products == {COs: 1.0}
    assert r.net_stoich() == {CO: -1.0, s: -1.0, COs: 1.0}


def test_coefficients_via_multiplication():
    _, s, _, _, _, Os = build()
    assert (2 * s).coeff == 2.0
    assert (s * 2).coeff == 2.0
    assert (2 * s).species is s


def test_surface_reaction_with_coefficients():
    _, s, _, CO2, COs, Os = build()
    r = COs + Os >> CO2 + 2 * s
    net = r.net_stoich()
    assert net[COs] == -1.0
    assert net[Os] == -1.0
    assert net[CO2] == 1.0
    assert net[s] == 2.0


def test_dissociative_adsorption_merges_duplicates():
    m = mk.Model("t2", T=500)
    s = m.site("Pt", density=1.0)
    O2 = m.gas("O2")
    Os = m.adsorbate("O*", site=s)
    r = O2 + 2 * s >> 2 * Os
    assert r.reactants[s] == 2.0
    assert r.products[Os] == 2.0
    assert r.net_stoich()[s] == -2.0
    assert r.net_stoich()[Os] == 2.0


def test_duplicate_species_name_rejected():
    m = mk.Model("t3", T=500)
    m.gas("CO")
    try:
        m.gas("CO")
    except ValueError:
        return
    raise AssertionError("duplicate species name should raise")
