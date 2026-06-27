"""Figures (energy diagram, network) and the HTML report."""

import matplotlib

matplotlib.use("Agg")

import discopt_mkm as mk
from discopt_mkm.examples import co_oxidation, water_gas_shift


def test_energy_diagram_returns_axes():
    m, _ = co_oxidation()
    ax = mk.energy_diagram(m)
    assert ax.get_figure() is not None
    # one connector polyline per reaction
    assert len(ax.lines) == len(m.reactions)


def test_network_graph_runs_with_and_without_solution():
    m, reactor = co_oxidation()
    assert mk.network_graph(m) is not None
    sol = mk.solve_steady_state(m, reactor)
    assert mk.network_graph(m, solution=sol) is not None


def test_to_dot_structure():
    m, _ = co_oxidation()
    dot = mk.to_dot(m)
    assert dot.startswith('digraph')
    assert dot.count("ellipse") == len(m.species)
    assert dot.count("shape=box") == len(m.reactions)


def test_report_html_contains_all_sections():
    m, reactor = co_oxidation()
    sol = mk.solve_steady_state(m, reactor)
    html = mk.report_html(m, solution=sol, target=m._by_name["CO2"], figures=True)
    for section in ("Mechanism", "Stoichiometric structure", "overall reaction",
                    "site balance", "Free-energy diagram", "Reaction network",
                    "Degree of rate control", "apparent orders", "data:image/svg+xml"):
        assert section in html


def test_report_handles_explicit_rate_mechanism():
    # WGS has explicit kf/Keq steps (no Ea) -> barrierless connectors, no figures crash
    m, _ = water_gas_shift()
    html = mk.report_html(m, figures=True)
    assert "water_gas_shift" in html and "Free-energy diagram" in html
