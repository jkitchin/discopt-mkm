"""Elemental composition: formula parsing, element conservation laws, balance checks."""

import pytest

import discopt.mkm as mk
from discopt.mkm.analysis import (
    check_element_balance,
    conserved_quantities,
    element_conservation_laws,
    n_conservation_laws,
)
from discopt.mkm.examples import co_oxidation, water_gas_shift
from discopt.mkm.formula import looks_like_formula, parse_formula


def test_formula_parsing():
    assert parse_formula("CO2") == {"C": 1, "O": 2}
    assert parse_formula("H2O") == {"H": 2, "O": 1}
    assert parse_formula("CO*") == {"C": 1, "O": 1}  # site marker stripped
    assert parse_formula("Ca(OH)2") == {"Ca": 1, "O": 2, "H": 2}
    assert looks_like_formula("C3H6") and not looks_like_formula("A")


def test_infer_composition_flags_non_formulas():
    m = mk.Model("x")
    s = m.site("s", density=1.0)
    m.gas("CO2")  # parses
    m.adsorbate("A*", site=s)  # 'A' is not a real element
    unparsed = m.infer_composition()
    assert m._by_name["CO2"].composition == {"C": 1, "O": 2}
    assert [sp.name for sp in unparsed] == ["A*"]


@pytest.mark.parametrize("builder,expected", [(co_oxidation, {"C", "O"}), (water_gas_shift, {"C", "H", "O"})])
def test_element_conservation_laws(builder, expected):
    m, _ = builder()  # builders call infer_composition()
    laws = element_conservation_laws(m)
    assert set(laws) == expected
    # the full clean basis = sites + elements, matching the null-space dimension
    cq = conserved_quantities(m)
    assert len(cq) == n_conservation_laws(m)
    assert any(k.startswith("site:") for k in cq)


def test_mechanism_is_balanced():
    m, _ = water_gas_shift()
    assert check_element_balance(m) == []


def test_unbalanced_reaction_is_detected():
    m = mk.Model("bad", R=8.314)
    s = m.site("s", density=1.0)
    CO = m.gas("CO", composition={"C": 1, "O": 1})
    CO2 = m.gas("CO2", composition={"C": 1, "O": 2})
    cos = m.adsorbate("CO*", site=s, composition={"C": 1, "O": 1})
    m.step(CO + s >> cos, A=1e4, Ea=0.0)
    m.step(cos >> CO2 + s, A=1e4, Ea=0.0)  # CO* -> CO2 + * loses an O atom!
    violations = check_element_balance(m)
    assert any(e == "O" for _, e, _ in violations)
