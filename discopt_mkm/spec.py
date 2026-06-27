"""Declarative, agent-friendly model specification.

Agents reliably emit structured data and strings, so a microkinetic model can be
described as a dict/YAML with string reaction equations, instead of the
object-reference operator DSL. ``from_spec`` / ``from_yaml`` parse that into a
``MicrokineticModel`` (+ reactor); ``to_spec`` exports one back. The ``ModelSpec``
Pydantic model validates input and provides a JSON schema (``ModelSpec.model_json_schema()``)
for tool definitions.

Reaction strings use ``<=>`` (reversible) or ``->`` (irreversible); terms are
``coeff species`` with the site/adsorbate ``*`` allowed, e.g.
``"O2 + 2 * <=> 2 O*"`` or ``"CO* + O* -> CO2 + 2 *"``.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from discopt_mkm.reaction import Reaction

_ARROWS = [("<=>", False), ("<->", False), ("=>", True), ("->", True)]
_TERM = re.compile(r"^(\d+\.?\d*)?\s*(\S.*)$")


def parse_equation(eq: str):
    """Parse ``"A + 2 B <=> C"`` -> ``(reactants, products, irreversible)``.

    ``reactants``/``products`` are ``{species_name: coefficient}`` dicts.
    """
    for tok, irreversible in _ARROWS:
        if tok in eq:
            lhs, rhs = eq.split(tok, 1)
            return _parse_side(lhs), _parse_side(rhs), irreversible
    raise ValueError(f"reaction {eq!r} has no arrow (use '<=>' for reversible or '->' for irreversible)")


def _parse_side(side: str) -> dict:
    terms: dict[str, float] = {}
    for part in side.split("+"):
        part = part.strip()
        if not part:
            continue
        m = _TERM.match(part)
        coeff = float(m.group(1)) if m.group(1) else 1.0
        name = m.group(2).strip()
        terms[name] = terms.get(name, 0.0) + coeff
    return terms


# --------------------------------------------------------------------------- schema
class ThermoSpec(BaseModel):
    type: Literal["nasa7", "shomate"]
    low: Optional[list[float]] = None
    high: Optional[list[float]] = None
    Tmid: float = 1000.0
    coeffs: Optional[list[float]] = Field(default=None, description="Shomate A..H (8 values)")


class SiteSpec(BaseModel):
    name: str
    density: float = 1.0


class SpeciesSpec(BaseModel):
    name: str
    H: float = 0.0
    S: float = 0.0
    Cp: float = 0.0
    composition: Optional[dict[str, int]] = None
    thermo: Optional[ThermoSpec] = None


class AdsorbateSpec(SpeciesSpec):
    site: str


class ReactionSpec(BaseModel):
    equation: str
    A: Optional[float] = None
    Ea: Optional[float] = None
    kf: Optional[float] = None
    Keq: Optional[float] = None
    irreversible: bool = False
    equilibrated: bool = False
    alpha: float = 0.0
    n_electrons: Optional[float] = None
    beta: float = 0.5
    name: Optional[str] = None


class InteractionSpec(BaseModel):
    a: str
    b: str
    eps: float


class ReactorSpec(BaseModel):
    type: Literal["differential", "cstr", "batch"] = "differential"
    pressures: dict[str, float] = {}
    inlet: dict[str, float] = {}
    initial: dict[str, float] = {}
    tau: float = 1.0
    cat_density: float = 1.0


class ModelSpec(BaseModel):
    """A complete microkinetic model as data (validated, JSON-schema-able)."""

    name: str = "model"
    T: float = 500.0
    R: float = 8.617e-5
    Tref: float = 298.15
    U: float = 0.0
    F: float = 1.0
    sites: list[SiteSpec] = []
    gas: list[SpeciesSpec] = []
    adsorbates: list[AdsorbateSpec] = []
    reactions: list[ReactionSpec] = []
    interactions: list[InteractionSpec] = []
    reactor: Optional[ReactorSpec] = None
    infer_composition: bool = True


# --------------------------------------------------------------------------- build
def _thermo(ts: ThermoSpec):
    from discopt_mkm.thermo_models import NASA7, Shomate

    if ts.type == "nasa7":
        return NASA7(ts.low, ts.high, ts.Tmid)
    return Shomate(*ts.coeffs)


def from_spec(spec):
    """Build ``(model, reactor)`` from a spec dict (or ``ModelSpec``).

    ``reactor`` is ``None`` if the spec has no ``reactor`` section.
    """
    from discopt_mkm.model import MicrokineticModel
    from discopt_mkm.reactors import CSTR, Batch, DifferentialReactor

    s = spec if isinstance(spec, ModelSpec) else ModelSpec(**spec)
    m = MicrokineticModel(s.name, T=s.T, R=s.R, Tref=s.Tref, U=s.U, F=s.F)

    for site in s.sites:
        m.site(site.name, density=site.density)
    for g in s.gas:
        m.gas(g.name, H=g.H, S=g.S, Cp=g.Cp, composition=g.composition,
              thermo=_thermo(g.thermo) if g.thermo else None)
    for a in s.adsorbates:
        if a.site not in m._by_name:
            raise ValueError(f"adsorbate {a.name!r} references unknown site {a.site!r}")
        m.adsorbate(a.name, site=m._by_name[a.site], H=a.H, S=a.S, Cp=a.Cp,
                    composition=a.composition, thermo=_thermo(a.thermo) if a.thermo else None)
    for it in s.interactions:
        m.interaction(_resolve(m, it.a), _resolve(m, it.b), it.eps)

    for r in s.reactions:
        reactants, products, arrow_irr = parse_equation(r.equation)
        rxn = Reaction(_stoich(m, reactants, r.equation), _stoich(m, products, r.equation))
        m.step(rxn, A=r.A, Ea=r.Ea, kf=r.kf, Keq=r.Keq,
               irreversible=r.irreversible or arrow_irr, equilibrated=r.equilibrated,
               alpha=r.alpha, n_electrons=r.n_electrons, beta=r.beta, name=r.name)

    if s.infer_composition:
        m.infer_composition()

    reactor = _reactor(m, s.reactor) if s.reactor else None
    return m, reactor


def _resolve(m, name):
    if name not in m._by_name:
        raise ValueError(f"unknown species {name!r}; declare it in sites/gas/adsorbates")
    return m._by_name[name]


def _stoich(m, terms, eq):
    out = {}
    for name, coeff in terms.items():
        if name not in m._by_name:
            raise ValueError(f"reaction {eq!r} references unknown species {name!r}")
        out[m._by_name[name]] = coeff
    return out


def _reactor(m, rs: ReactorSpec):
    from discopt_mkm.reactors import CSTR, Batch, DifferentialReactor

    if rs.type == "differential":
        return DifferentialReactor({_resolve(m, k): v for k, v in rs.pressures.items()})
    if rs.type == "cstr":
        return CSTR({_resolve(m, k): v for k, v in rs.inlet.items()}, tau=rs.tau, cat_density=rs.cat_density)
    return Batch({_resolve(m, k): v for k, v in rs.initial.items()}, cat_density=rs.cat_density)


def from_yaml(text_or_path):
    """Build ``(model, reactor)`` from a YAML string or file path."""
    import os

    import yaml

    text = text_or_path
    if os.path.exists(str(text_or_path)):
        with open(text_or_path) as f:
            text = f.read()
    return from_spec(yaml.safe_load(text))


def to_spec(mkm, reactor=None) -> dict:
    """Export a model (and optional reactor) back to a spec dict."""
    s = {
        "name": mkm.name, "T": mkm.T, "R": mkm.R, "Tref": mkm.Tref, "U": mkm.U, "F": mkm.F,
        "sites": [{"name": x.name, "density": x.density} for x in mkm.sites],
        "gas": [_sp_dict(x) for x in mkm.gas_species],
        "adsorbates": [{**_sp_dict(x), "site": x.site.name} for x in mkm.adsorbates],
        "reactions": [_rxn_dict(r) for r in mkm.reactions],
        "interactions": [{"a": a.name, "b": b.name, "eps": e} for a, b, e in mkm._interactions],
    }
    if reactor is not None and hasattr(reactor, "pressures"):
        s["reactor"] = {"type": "differential",
                        "pressures": {g.name: float(v) for g, v in reactor.pressures.items()}}
    return s


def _sp_dict(sp):
    d = {"name": sp.name, "S": sp.S, "Cp": sp.Cp}
    if not callable(sp.H):
        d["H"] = sp.H
    if sp.composition:
        d["composition"] = dict(sp.composition)
    return d


def _rxn_dict(r):
    d = {"equation": r.equation(), "name": r.name}
    if r.equilibrated:
        d["equilibrated"] = True
        if r.Keq is not None:
            d["Keq"] = r.Keq
    elif r.explicit_rate:
        d["kf"] = r.kf
        if not r.irreversible:
            d["Keq"] = r.Keq
    else:
        d["A"], d["Ea"] = r.A, r.Ea
    if r.alpha:
        d["alpha"] = r.alpha
    if getattr(r, "is_electrochemical", False):
        d["n_electrons"] = r.n_electrons
        d["beta"] = r.beta
    return d
