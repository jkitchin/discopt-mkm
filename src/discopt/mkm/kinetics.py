"""Rate laws: Arrhenius forward rate, derived reverse rate, mass-action rates.

These functions build discopt expressions and are reused *verbatim* by both the
steady-state residual assembler and the transient ODE right-hand side. They use
only element-wise discopt operations (``*``, ``**``, ``exp``) so they broadcast
correctly over the ``(nfe, ncp)`` collocation arrays the DAE builder supplies,
as well as over scalar steady-state variables.
"""

from __future__ import annotations

from functools import reduce

import discopt.modeling as dm

from discopt.mkm.reaction import Reaction
from discopt.mkm.species import Adsorbate, GasSpecies, Site, Species
from discopt.mkm.thermo import K_eq, delta_reaction_H, ec_shift


def k_forward(rxn: Reaction, T_expr, R: float, theta=None):
    """Forward rate constant.

    Explicit-rate steps return ``kf`` directly. Arrhenius steps return
    ``A exp(-Ea(theta) / (R T))`` where the barrier picks up a Brønsted-Evans-
    Polanyi shift ``alpha * delta(reaction energy)`` from lateral interactions
    when ``alpha`` and interactions are present.

    For an electrochemical step the forward barrier additionally shifts by the
    Butler-Volmer fraction ``beta * n_electrons * F * U`` of the potential term,
    applied as a multiplicative ``exp(-beta * n F U / RT)`` factor. The reverse
    rate (``k_f / K_eq``) then carries the complementary ``(1-beta)`` factor,
    since ``K_eq`` holds the full ``n F U`` shift.
    """
    if rxn.explicit_rate:
        kf = rxn.kf_param
    else:
        Ea = rxn.Ea_param
        if rxn.alpha_param is not None:
            Ea = Ea + rxn.alpha_param * delta_reaction_H(rxn, theta)
        kf = rxn.A_param * dm.exp(-Ea / (R * T_expr))
    if getattr(rxn, "is_electrochemical", False) and rxn.beta_param is not None:
        kf = kf * dm.exp(-rxn.beta_param * ec_shift(rxn) / (R * T_expr))
    return kf


def k_reverse(rxn: Reaction, T_expr, R: float, Tref: float, theta=None):
    """Reverse rate constant, *derived* as ``k_forward / K_eq`` (0 if irreversible).

    For explicit-rate steps ``K_eq`` is the supplied constant; for Arrhenius
    steps it is ``exp(-dG_rxn(theta)/RT)`` from the species thermodynamics
    (coverage-dependent when interactions are present). Either way the reverse
    rate is consistent and the equilibrium constant is held fixed when the
    forward rate constant is perturbed for degree of rate control.
    """
    if rxn.irreversible:
        return 0.0
    if rxn.explicit_rate:
        # explicit K_eq is the *bare* (U = 0) constant; for a faradaic step the
        # effective equilibrium constant carries the full n F U shift so that the
        # derived reverse rate picks up the complementary (1 - beta) Butler-Volmer
        # factor (detailed balance k_f / k_r = K_eq(U); mirrors numeric.py:69).
        keq = rxn.Keq_param
        if getattr(rxn, "is_electrochemical", False) and rxn._U_param is not None:
            keq = keq * dm.exp(-ec_shift(rxn) / (R * T_expr))
    else:
        keq = K_eq(rxn, T_expr, R, Tref, theta)
    return k_forward(rxn, T_expr, R, theta) / keq


def activity(sp: Species, conc: dict, theta: dict, free_cov: dict):
    """Activity used in the mass-action law.

    Gas species use concentration / partial pressure, adsorbates use coverage,
    and a bare site uses its free-site coverage.
    """
    if isinstance(sp, GasSpecies):
        return conc[sp]
    if isinstance(sp, Adsorbate):
        return theta[sp]
    if isinstance(sp, Site):
        return free_cov[sp]
    raise TypeError(f"species {sp!r} has no defined activity")


def _mass_action(stoich: dict, conc: dict, theta: dict, free_cov: dict):
    """Element-wise product ``prod_i activity_i ** coeff_i``.

    Built by Python ``*``-reduction (not ``dm.prod``) so it stays element-wise
    over array-valued activities rather than collapsing them to a scalar.
    """
    factors = [activity(s, conc, theta, free_cov) ** c for s, c in stoich.items()]
    if not factors:
        return 1.0
    return reduce(lambda a, b: a * b, factors)


def equilibrium_residual(rxn: Reaction, conc: dict, theta: dict, free_cov: dict, T_expr, R: float, Tref: float):
    """Equilibrium-quotient residual ``Pi act(products) - K_eq Pi act(reactants)``.

    Zero when the step is at equilibrium. Written in quotient form (no rate
    constants), so it carries no huge-number cancellation — that is the whole
    numerical point of the quasi-equilibrium approximation.
    """
    if rxn.Keq is not None:
        # explicit K_eq: shift by the full n F U for a faradaic equilibrated step
        # so the equilibrium coverages respond to the electrode potential.
        keq = rxn.Keq_param
        if getattr(rxn, "is_electrochemical", False) and rxn._U_param is not None:
            keq = keq * dm.exp(-ec_shift(rxn) / (R * T_expr))
    else:
        keq = K_eq(rxn, T_expr, R, Tref, theta)
    return _mass_action(rxn.products, conc, theta, free_cov) - keq * _mass_action(
        rxn.reactants, conc, theta, free_cov
    )


def rate_of_progress(
    rxn: Reaction, conc: dict, theta: dict, free_cov: dict, T_expr, R: float, Tref: float, extents=None
):
    """Net rate of progress ``r_j = k_f Pi act(reactants) - k_r Pi act(products)``.

    For an irreversible step the reverse term (and its ``K_eq`` expression) is
    omitted entirely. For a quasi-equilibrated step the rate is the unknown
    extent supplied in ``extents`` (its rate law is not used).
    """
    if extents is not None and rxn in extents:
        return extents[rxn]
    fwd = k_forward(rxn, T_expr, R, theta) * _mass_action(rxn.reactants, conc, theta, free_cov)
    if rxn.irreversible:
        return fwd
    kr = k_reverse(rxn, T_expr, R, Tref, theta)
    rev = kr * _mass_action(rxn.products, conc, theta, free_cov)
    return fwd - rev


def net_rate(
    sp: Species, reactions, conc: dict, theta: dict, free_cov: dict, T_expr, R: float, Tref: float, extents=None
):
    """Net production rate of a species ``R_i = sum_j nu_ij r_j``."""
    terms = []
    for rxn in reactions:
        nu = rxn.net_stoich().get(sp, 0.0)
        if nu != 0.0:
            terms.append(nu * rate_of_progress(rxn, conc, theta, free_cov, T_expr, R, Tref, extents))
    if not terms:
        return 0.0
    return dm.sum(terms)
