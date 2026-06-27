"""Thermodynamics: species free energy, reaction free energy, equilibrium constant.

All quantities are built as discopt expressions in terms of the temperature
parameter so that temperature drives *both* the kinetics (via Arrhenius) and the
thermodynamics (via ``K_eq``) from a single consistent source.

Units are the caller's responsibility: ``H``, ``S``, ``Cp``, ``Ea`` and the gas
constant ``R`` must be mutually consistent (e.g. eV and eV/K with
``R = 8.617e-5``, or J/mol and J/mol/K with ``R = 8.314``).
"""

from __future__ import annotations

import discopt.modeling as dm

from discopt_mkm.reaction import Reaction
from discopt_mkm.species import Species


def delta_H(sp: Species, theta) -> object:
    """Coverage-induced enthalpy shift ``sum_j eps_ij theta_j`` (0 if none).

    Comes from lateral interactions registered with ``Model.interaction``.
    """
    terms = getattr(sp, "_interaction_params", None)
    if not terms or theta is None:
        return 0.0
    contribs = [eps * theta[partner] for partner, eps in terms if partner in theta]
    return dm.sum(contribs) if contribs else 0.0


def base_H(sp: Species, theta):
    """Base enthalpy: the ``H`` parameter, or a user callable ``H(theta)``."""
    if callable(sp.H):
        if theta is None:
            raise ValueError(f"species {sp.name!r} has a coverage-dependent H; theta is required")
        return sp.H(theta)
    return sp.H_param


def delta_reaction_H(rxn: Reaction, theta):
    """Coverage-induced reaction-energy shift ``sum_i nu_i delta_H_i(theta)`` (for BEP)."""
    if theta is None:
        return 0.0
    contribs = []
    for s, nu in rxn.net_stoich().items():
        d = delta_H(s, theta)
        if not (isinstance(d, float) and d == 0.0):
            contribs.append(nu * d)
    return dm.sum(contribs) if contribs else 0.0


def g_species(sp: Species, T_expr, R: float, Tref: float, theta=None):
    """Standard Gibbs free energy ``G(T) = H(T) - T S(T)`` as a discopt expression.

    With a non-zero heat capacity the enthalpy and entropy are corrected from the
    reference temperature ``Tref``::

        H(T) = H + Cp (T - Tref)
        S(T) = S + Cp ln(T / Tref)

    With lateral interactions the enthalpy gains the coverage-dependent term
    ``sum_j eps_ij theta_j``. A per-species free-energy offset parameter ``dG``
    (default 0) is added; it is the perturbation handle for thermodynamic rate
    control.
    """
    if getattr(sp, "thermo", None) is not None:
        G = sp.thermo.g(T_expr, R, Tref, dm.log)
    else:
        H = base_H(sp, theta)
        S = sp.S_param
        if sp.Cp_param is not None:
            H = H + sp.Cp_param * (T_expr - Tref)
            S = S + sp.Cp_param * dm.log(T_expr / Tref)
        G = H - T_expr * S
    G = G + delta_H(sp, theta)  # lateral-interaction term (enthalpic)
    if sp.dG_param is not None:
        G = G + sp.dG_param
    return G


def ec_shift(rxn: Reaction):
    """Computational-hydrogen-electrode shift of the reaction free energy,
    ``n_electrons * F * U`` (a discopt expression). Zero for a chemical step.

    Putting the *full* shift here (and only the ``beta`` fraction in the forward
    barrier, see :func:`discopt_mkm.kinetics.k_forward`) makes the derived reverse
    rate carry the complementary Butler-Volmer factor automatically, so detailed
    balance ``k_f / k_r = K_eq`` holds at every potential.
    """
    if not getattr(rxn, "is_electrochemical", False) or rxn._U_param is None:
        return 0.0
    return rxn.n_electrons * rxn._F * rxn._U_param


def dG_rxn(rxn: Reaction, T_expr, R: float, Tref: float, theta=None):
    """Reaction free energy ``sum_i nu_i G_i(T)`` (plus the electrochemical
    ``n_electrons * F * U`` shift for a faradaic step)."""
    terms = [nu * g_species(s, T_expr, R, Tref, theta) for s, nu in rxn.net_stoich().items()]
    return dm.sum(terms) + ec_shift(rxn)


def K_eq(rxn: Reaction, T_expr, R: float, Tref: float, theta=None):
    """Equilibrium constant ``exp(-dG_rxn / (R T))``."""
    return dm.exp(-dG_rxn(rxn, T_expr, R, Tref, theta) / (R * T_expr))
