"""Symbolic derivation of overall (lumped) rate expressions with SymPy.

For a quasi-equilibrium mechanism (the fast steps marked ``equilibrated`` and one
or a few rate-determining steps), the surface coverages are fixed by the
equilibrium relations plus the site balance, independent of the slow step. SymPy
solves that algebraic system in closed form and substitutes the coverages into
the rate-determining step, yielding the classic Langmuir-Hinshelwood-Hougen-
Watson lumped rate ``r(P, k, K)``.

This is exact only when the coverage-determining subsystem (equilibrium relations
+ site balance) is square and solvable symbolically — i.e. the standard LHHW
setting. For a general all-kinetic network the steady state has no closed form;
there the *numerical* model is the lumped rate, and apparent orders / apparent
activation energy (:mod:`discopt_mkm.analysis.apparent`) characterize it locally.
"""

from __future__ import annotations

import sympy as sp

from discopt_mkm.model import _safe


def lumped_rate_expression(mkm, target_species):
    """Derive the overall rate of production of ``target_species`` in closed form.

    Requires a quasi-equilibrium mechanism (some steps ``equilibrated``). Returns
    ``(rate, symbols)`` where ``rate`` is a SymPy expression and ``symbols`` maps
    the model objects to the SymPy symbols used (gas pressures ``P_*``, per-step
    forward constants ``kf*`` and equilibrium constants ``Keq*``).
    """
    if not any(r.equilibrated for r in mkm.reactions):
        raise ValueError(
            "lumped_rate_expression needs a quasi-equilibrium mechanism "
            "(mark the fast steps equilibrated=True); a general network has no "
            "closed-form steady state."
        )

    theta = {a: sp.Symbol(f"theta_{_safe(a.name)}", positive=True) for a in mkm.adsorbates}
    free = {s: sp.Symbol(f"thetaf_{_safe(s.name)}", positive=True) for s in mkm.sites}
    P = {g: sp.Symbol(f"P_{_safe(g.name)}", positive=True) for g in mkm.gas_species}
    Keq = {r: sp.Symbol(f"Keq{j}", positive=True) for j, r in enumerate(mkm.reactions)}
    kf = {r: sp.Symbol(f"kf{j}", positive=True) for j, r in enumerate(mkm.reactions)}

    def act(s):
        return P.get(s) or theta.get(s) or free.get(s)

    def mass(stoich):
        e = sp.Integer(1)
        for s, c in stoich.items():
            e *= act(s) ** int(c)
        return e

    # coverage-determining subsystem: equilibrium relations + site balances
    eqs = [mass(r.products) - Keq[r] * mass(r.reactants) for r in mkm.reactions if r.equilibrated]
    for s in mkm.sites:
        occ = sum((theta[a] for a in mkm.adsorbates_on(s)), sp.Integer(0))
        eqs.append(free[s] + occ - 1)

    unknowns = list(theta.values()) + list(free.values())
    solutions = sp.solve(eqs, unknowns, dict=True)
    if not solutions:
        raise RuntimeError("could not solve the coverage subsystem symbolically (not LHHW-reducible)")
    cover = solutions[0]

    # production of the target through the rate-determining (kinetic) steps
    rate = sp.Integer(0)
    for r in mkm.reactions:
        if r.equilibrated:
            continue
        nu = r.net_stoich().get(target_species, 0)
        if nu == 0:
            continue
        rop = kf[r] * mass(r.reactants)
        if not r.irreversible:
            rop = rop - (kf[r] / Keq[r]) * mass(r.products)
        rate += int(nu) * rop

    rate = sp.simplify(rate.subs(cover))
    symbols = {"P": P, "kf": kf, "Keq": Keq, "theta": theta}
    return rate, symbols
