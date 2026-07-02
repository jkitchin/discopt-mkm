"""Mechanism discovery: a POUNCE reaction-network on a potential energy surface
turned into a microkinetic model spec.

``pounce.reaction_network`` maps a smooth energy surface ``fun(x)`` into its stable
states (minima) and the index-1 transition states (saddles) between them, with
barrier heights. This bridge turns that into a `discopt.mkm` **spec**: each
minimum becomes a surface state (an adsorbate) with ``H`` = its energy, and each
connecting saddle becomes a reversible isomerization step ``S_i <=> S_j`` with
forward ``Ea`` = the barrier. Because the reverse rate is derived from the species
thermodynamics, the reverse barrier comes out as ``barrier(j -> i)`` automatically
— consistent with the PES.

The result is the *mechanism skeleton with energetics* (renders as a free-energy
diagram, exposes routes/structure). To make it a solvable kinetic model, couple
some states to a gas phase (adsorption/desorption) in the returned spec.
"""

from __future__ import annotations


def spec_from_reaction_network(rn, name="discovered", site="*", A=1e13, T=500.0,
                               R=8.617e-5, Tref=298.15, energy_scale=1.0, state_names=None):
    """Build a `discopt.mkm` spec dict from a POUNCE ``ReactionNetwork``."""
    minima = rn.minima
    names = list(state_names) if state_names else [f"S{i}" for i in range(len(minima))]

    adsorbates = [
        {"name": names[i], "site": site, "H": float(minima[i].f) * energy_scale, "S": 0.0}
        for i in range(len(minima))
    ]
    reactions = []
    for (i, j) in rn.edges:
        reactions.append({
            "equation": f"{names[i]} <=> {names[j]}",
            "A": float(A),
            "Ea": float(rn.barrier(i, j)) * energy_scale,  # forward barrier; reverse from thermo
            "name": f"{names[i]} ⇌ {names[j]}",
        })
    return {
        "name": name, "T": T, "R": R, "Tref": Tref,
        "sites": [{"name": site, "density": 1.0}],
        "gas": [],
        "adsorbates": adsorbates,
        "reactions": reactions,
        "infer_composition": False,
    }


def discover_mechanism(fun, x0, grad, hess, *, A=1e13, T=500.0, R=8.617e-5,
                       energy_scale=1.0, state_names=None, **rn_kwargs):
    """Map the reaction network of a PES and return ``(reaction_network, spec)``.

    ``fun``/``grad``/``hess`` are the energy, gradient, and Hessian (NumPy
    callables); ``x0`` a starting point. Extra keyword args (``n_states``,
    ``n_transition_states``, ``seed``, ...) pass through to
    ``pounce.reaction_network``.
    """
    import pounce

    rn = pounce.reaction_network(fun, x0, grad=grad, hess=hess, **rn_kwargs)
    spec = spec_from_reaction_network(rn, A=A, T=T, R=R, energy_scale=energy_scale, state_names=state_names)
    return rn, spec
