"""Agent-native tool functions: JSON spec in, JSON result out.

Each function takes a declarative model spec (see :mod:`discopt.mkm.spec`) and
returns plain JSON-serializable data — no object references, no holding state.
These are the implementations the MCP server exposes, and they are usable
directly by any agent that can produce a dict.
"""

from __future__ import annotations

from discopt.mkm.spec import from_spec


def _resolve_target(m, target):
    if target is None:
        # default to a gas product (a species net-produced by some reaction)
        for g in m.gas_species:
            if any(rxn.net_stoich().get(g, 0) > 0 for rxn in m.reactions):
                return g
        if m.gas_species:
            return m.gas_species[-1]
        raise ValueError("no gas species to use as a target; pass target=<name>")
    if target not in m._by_name:
        raise ValueError(f"unknown target species {target!r}")
    return m._by_name[target]


def _solve(m, reactor, coordinates="linear", method="auto", active_tol=1e-3):
    from discopt.mkm import numeric
    from discopt.mkm.steady_state import solve_steady_state

    if reactor is None:
        raise ValueError("the spec needs a 'reactor' section to solve")
    if coordinates == "log":
        pressures = getattr(reactor, "pressures", {})
        theta0, _ = numeric.steady_state_numeric(m, pressures, m.T, theta0={a: 1e-3 for a in m.adsorbates})
        return solve_steady_state(m, reactor, coordinates="log", theta0=theta0, active_tol=active_tol)
    return solve_steady_state(m, reactor, method=method, active_tol=active_tol)


def _names(d):
    return {k.name: float(v) for k, v in d.items()}


# --------------------------------------------------------------------------- tools
def solve(spec, coordinates="linear", method="auto"):
    """Solve the steady state. Returns coverages, gas, rates, status."""
    m, reactor = from_spec(spec)
    return _solve(m, reactor, coordinates, method).to_dict()


def degree_of_rate_control(spec, target=None, coordinates="linear"):
    """Campbell degree of rate control for each step (w.r.t. ``target``'s rate)."""
    from discopt.mkm.analysis import degree_of_rate_control as drc
    from discopt.mkm.analysis.drc import SensitivityUnavailable

    m, reactor = from_spec(spec)
    sol = _solve(m, reactor, coordinates, active_tol=1e-13 if coordinates == "log" else 1e-3)
    tgt = _resolve_target(m, target)
    try:
        X = drc(sol, species=tgt)
        return {"target": tgt.name, "drc": {r.name: float(x) for r, x in X.items()},
                "sum": float(sum(X.values()))}
    except SensitivityUnavailable as e:
        return {"target": tgt.name, "drc": None, "note": str(e)}


def apparent_kinetics(spec, target=None):
    """Apparent reaction orders and apparent activation energy (differential reactor)."""
    from discopt.mkm.analysis import apparent_activation_energy, apparent_orders

    m, reactor = from_spec(spec)
    sol = _solve(m, reactor)
    tgt = _resolve_target(m, target)
    out = {"target": tgt.name}
    try:
        out["orders"] = {g.name: float(n) for g, n in apparent_orders(sol, tgt).items()}
        out["apparent_Ea"] = float(apparent_activation_energy(sol, tgt))
    except Exception as e:
        out["note"] = str(e)
    return out


def structure(spec):
    """Stoichiometric structure: overall reaction, independence, conservation, balance."""
    from discopt.mkm.analysis import stoichiometry as st

    m, _ = from_spec(spec)
    routes = [{"stoichiometric_numbers": [int(round(x)) for x in sigma],
               "overall_reaction": _names(overall)} for sigma, overall in st.reaction_routes(m)]
    return {
        "n_reactions": len(m.reactions),
        "n_independent_reactions": st.n_independent_reactions(m),
        "routes": routes,
        "conservation_laws": {k: _names(v) for k, v in st.conserved_quantities(m).items()},
        "element_balanced": st.check_element_balance(m) == [],
    }


def validate(spec):
    """Validate a spec without solving: parse, mass balance, structure. Returns
    ``{ok, errors, warnings, info}``."""
    from discopt.mkm.analysis import stoichiometry as st

    errors, warnings = [], []
    try:
        m, reactor = from_spec(spec)
    except Exception as e:
        return {"ok": False, "errors": [str(e)], "warnings": [], "info": {}}

    for rxn, elem, residual in st.check_element_balance(m):
        errors.append(f"reaction '{rxn.name}' does not conserve element {elem} (residual {residual:+g})")
    for rxn, site, residual in st.check_site_balance(m):
        errors.append(f"reaction '{rxn.name}' does not conserve site {site.name} (residual {residual:+g})")
    unparsed = [sp.name for sp in [*m.gas_species, *m.adsorbates] if not sp.composition]
    if unparsed:
        warnings.append(f"no elemental composition for {unparsed} (mass balance not checked for these); "
                        "set composition= explicitly")
    if reactor is None:
        warnings.append("no reactor section; add one to solve")

    info = {
        "n_species": len(m.species),
        "n_reactions": len(m.reactions),
        "n_independent_reactions": st.n_independent_reactions(m),
        "overall_reaction": [_names(o) for _, o in st.reaction_routes(m)],
        "n_conservation_laws": st.n_conservation_laws(m),
    }
    return {"ok": not errors, "errors": errors, "warnings": warnings, "info": info}


def analyze(spec, target=None, coordinates="linear"):
    """One-call analysis: structure + steady state + DRC + apparent kinetics."""
    from discopt.mkm.analysis import (
        apparent_activation_energy,
        apparent_orders,
        degree_of_rate_control as drc,
    )
    from discopt.mkm.analysis.drc import SensitivityUnavailable

    m, reactor = from_spec(spec)
    out = {"structure": structure(spec)}
    if reactor is None:
        out["note"] = "no reactor; returned structure only"
        return out

    sol = _solve(m, reactor, coordinates, active_tol=1e-13 if coordinates == "log" else 1e-3)
    out["steady_state"] = sol.to_dict()
    tgt = _resolve_target(m, target)
    out["target"] = tgt.name
    out["tof"] = float(sol.production_rate(tgt))
    try:
        X = drc(sol, species=tgt)
        out["drc"] = {r.name: float(x) for r, x in X.items()}
    except SensitivityUnavailable as e:
        out["drc"] = None
        out["drc_note"] = str(e)
    try:
        out["apparent_orders"] = {g.name: float(n) for g, n in apparent_orders(sol, tgt).items()}
        out["apparent_Ea"] = float(apparent_activation_energy(sol, tgt))
    except Exception:
        pass
    return out


def report(spec, target=None, coordinates="linear", path=None):
    """Render the HTML mechanism report. Writes to ``path`` if given; returns HTML."""
    from discopt.mkm.report import report_html, write_report

    m, reactor = from_spec(spec)
    sol = _solve(m, reactor, coordinates) if reactor is not None else None
    tgt = _resolve_target(m, target) if (sol is not None) else None
    if path:
        return write_report(m, path, solution=sol, target=tgt)
    return report_html(m, solution=sol, target=tgt)


def spec_schema() -> dict:
    """The JSON schema for a model spec (for tool definitions / agent validation)."""
    from discopt.mkm.spec import ModelSpec

    return ModelSpec.model_json_schema()
