"""Shared assembly helpers reused by the steady-state and transient builders."""

from __future__ import annotations

import discopt.modeling as dm

from discopt_mkm.kinetics import equilibrium_residual
from discopt_mkm.model import MicrokineticModel, _safe
from discopt_mkm.species import Site


def quasi_equilibrium(m, mkm: MicrokineticModel, conc: dict, theta: dict, free_cov: dict, T_expr):
    """Build extent-rate variables and equilibrium constraints for QEA steps.

    Returns ``(extents, eq_residuals)`` where ``extents`` maps each quasi-
    equilibrated reaction to its unknown rate-of-progress variable (used in place
    of a rate law in the species balances) and ``eq_residuals`` are the
    equilibrium-quotient expressions constrained to zero.
    """
    extents, eq_residuals = {}, []
    for j, rxn in enumerate(mkm.reactions):
        if rxn.equilibrated:
            extents[rxn] = m.continuous(f"extent_{j}", lb=-1e8, ub=1e8)
            eq_residuals.append(equilibrium_residual(rxn, conc, theta, free_cov, T_expr, mkm.R, mkm.Tref))
    return extents, eq_residuals


def site_balance_residual(mkm: MicrokineticModel, site: Site, theta: dict, free_cov: dict):
    """Residual ``theta_free + sum(theta_ads) - 1`` for one site type.

    Equals zero when the site conservation ``sum(theta) = 1`` holds. Using an
    explicit free-coverage variable keeps this constraint linear and well posed,
    and lets it *replace* one redundant coverage steady-state equation per site.
    """
    occupied = [theta[a] for a in mkm.adsorbates_on(site)]
    total = free_cov[site] + (dm.sum(occupied) if occupied else 0.0)
    return total - 1.0
