"""Agent-native layer: spec parsing, JSON tool functions, MCP wiring."""

import json

import pytest

import discopt.mkm as mk
from discopt.mkm import agent
from discopt.mkm.spec import from_spec, from_yaml, parse_equation, to_spec

SPEC = {
    "name": "co_ox", "T": 500, "R": 8.617e-5,
    "sites": [{"name": "*", "density": 1.0}],
    "gas": [{"name": "CO", "H": 0, "S": 0.0020}, {"name": "O2", "H": 0, "S": 0.0021},
            {"name": "CO2", "H": -3.0, "S": 0.0023}],
    "adsorbates": [{"name": "CO*", "site": "*", "H": -0.8, "S": 0.0005},
                   {"name": "O*", "site": "*", "H": -0.3, "S": 0.0005}],
    "reactions": [
        {"equation": "CO + * <=> CO*", "A": 1e4, "Ea": 0.0},
        {"equation": "O2 + 2 * <=> 2 O*", "A": 1e4, "Ea": 0.0},
        {"equation": "CO* + O* -> CO2 + 2 *", "A": 1e8, "Ea": 0.7}],
    "reactor": {"type": "differential", "pressures": {"CO": 1.0, "O2": 0.5, "CO2": 0.0}},
}


def test_equation_parser():
    r, p, irr = parse_equation("O2 + 2 * <=> 2 O*")
    assert r == {"O2": 1.0, "*": 2.0} and p == {"O*": 2.0} and irr is False
    assert parse_equation("A -> B")[2] is True


def test_from_spec_builds_and_solves():
    m, reactor = from_spec(SPEC)
    assert len(m.reactions) == 3 and m.reactions[2].irreversible  # '->'
    sol = mk.solve_steady_state(m, reactor)
    assert sol.production_rate(m._by_name["CO2"]) == pytest.approx(1.19016, rel=1e-3)


def test_yaml_and_roundtrip():
    import yaml
    m, r = from_yaml(yaml.safe_dump(SPEC))
    assert mk.solve_steady_state(m, r).status == "optimal"
    m2, _ = from_spec(to_spec(*from_spec(SPEC)))  # roundtrip
    assert len(m2.reactions) == 3


def test_agent_tools_are_json_serializable():
    for result in (agent.validate(SPEC), agent.structure(SPEC), agent.solve(SPEC),
                   agent.degree_of_rate_control(SPEC, "CO2"), agent.apparent_kinetics(SPEC, "CO2"),
                   agent.analyze(SPEC, "CO2")):
        json.dumps(result)  # must not raise


def test_analyze_contents():
    out = agent.analyze(SPEC, target="CO2")
    assert out["tof"] == pytest.approx(1.19016, rel=1e-3)
    assert sum(out["drc"].values()) == pytest.approx(1.0, abs=1e-3)
    assert set(out["apparent_orders"]) == {"CO", "O2"}  # CO2 at P=0 -> order undefined, skipped
    assert out["structure"]["routes"][0]["overall_reaction"] == {"CO": -2.0, "O2": -1.0, "CO2": 2.0}


def test_validate_catches_element_imbalance():
    bad = json.loads(json.dumps(SPEC))
    bad["reactions"][2]["equation"] = "CO* + O* -> CO2 + CO2 + 2 *"  # extra C and O
    v = agent.validate(bad)
    assert not v["ok"] and any("conserve element" in e for e in v["errors"])


def test_validate_catches_site_imbalance():
    bad = json.loads(json.dumps(SPEC))
    bad["reactions"][2]["equation"] = "CO* + O* -> CO2 + *"  # 2 sites in, 1 out
    v = agent.validate(bad)
    assert not v["ok"] and any("conserve site" in e for e in v["errors"])


def test_spec_schema_and_mcp_tools():
    schema = agent.spec_schema()
    assert "reactions" in schema["properties"] and schema["title"] == "ModelSpec"
    # the MCP server registers the tools
    from discopt.mkm import mcp_server
    assert mcp_server.mcp is not None
