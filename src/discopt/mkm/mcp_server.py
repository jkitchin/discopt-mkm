"""Model Context Protocol server exposing discopt-mkm as agent tools.

Run with ``discopt-mkm-mcp`` (console script) or ``python -m discopt.mkm.mcp_server``.
Each tool takes a declarative model spec (a JSON object; call ``spec_schema`` for
its shape, or see AGENTS.md) and returns JSON. Requires the ``mcp`` extra:
``uv sync --extra mcp``.
"""

from __future__ import annotations

from discopt.mkm import agent

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover
    raise SystemExit("the MCP server needs the 'mcp' extra: uv sync --extra mcp") from e

mcp = FastMCP("discopt-mkm")


@mcp.tool()
def spec_schema() -> dict:
    """Return the JSON schema for a microkinetic model spec (the input to every other tool)."""
    return agent.spec_schema()


@mcp.tool()
def validate(spec: dict) -> dict:
    """Validate a model spec WITHOUT solving: parse, element + site mass balance, and
    stoichiometric structure. Returns {ok, errors, warnings, info}. Call this first."""
    return agent.validate(spec)


@mcp.tool()
def structure(spec: dict) -> dict:
    """Stoichiometric structure: overall reaction (Horiuti-Temkin route), number of
    independent reactions, conservation laws (site + element), and mass-balance status."""
    return agent.structure(spec)


@mcp.tool()
def solve(spec: dict, coordinates: str = "linear") -> dict:
    """Solve the steady state. Returns coverages, free-site coverage, gas concentrations,
    rates of progress, and status. Use coordinates='log' for stiff near-equilibrium
    mechanisms (e.g. water-gas shift)."""
    return agent.solve(spec, coordinates=coordinates)


@mcp.tool()
def degree_of_rate_control(spec: dict, target: str | None = None, coordinates: str = "linear") -> dict:
    """Campbell degree of rate control for each step w.r.t. the production rate of
    `target` (a species name; defaults to a gas product). The values sum to ~1."""
    return agent.degree_of_rate_control(spec, target=target, coordinates=coordinates)


@mcp.tool()
def apparent_kinetics(spec: dict, target: str | None = None) -> dict:
    """Apparent reaction orders (d ln r / d ln P_i) and apparent activation energy
    (R T^2 d ln r / dT) for `target`. Needs a differential reactor."""
    return agent.apparent_kinetics(spec, target=target)


@mcp.tool()
def analyze(spec: dict, target: str | None = None, coordinates: str = "linear") -> dict:
    """One-call analysis: structure + steady state + degree of rate control + apparent
    kinetics for `target`. The most useful single tool."""
    return agent.analyze(spec, target=target, coordinates=coordinates)


@mcp.tool()
def report(spec: dict, target: str | None = None, path: str | None = None) -> str:
    """Render a self-contained HTML mechanism report (mechanism, structure, figures,
    steady state, DRC). Writes to `path` if given, else returns the HTML."""
    return agent.report(spec, target=target, path=path)


def main():  # pragma: no cover
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
