"""Non-isothermal energy balance: reaction enthalpy, heat release, thermal mass.

Temperature is carried as a discopt expression (a Variable for a steady CSTR, a
state for a transient/PFR solve) and flows into every rate constant (Arrhenius)
and equilibrium constant, so the energy and species balances are fully coupled.

Heat capacities are taken from each species' ``Cp`` (so gas species need a
non-zero ``Cp`` for a non-isothermal solve). Units are the caller's
responsibility but must be mutually consistent with the species enthalpies.
"""

from __future__ import annotations

from dataclasses import dataclass

import discopt.modeling as dm

from discopt.mkm.kinetics import rate_of_progress
from discopt.mkm.model import MicrokineticModel
from discopt.mkm.reaction import Reaction


@dataclass
class EnergyBalance:
    """Configuration for a non-isothermal solve.

    Parameters
    ----------
    T_in : float
        Inlet (CSTR/PFR) or initial (batch) temperature.
    Q : float
        External heat input per unit reactor volume (0 = adiabatic; positive
        heats the reactor).
    """

    T_in: float
    Q: float = 0.0


def h_species(sp, T_expr, R, Tref, theta=None):
    """Species enthalpy ``H(T)`` as a discopt expression (thermo model or constant)."""
    from discopt.mkm.thermo import base_H, delta_H

    if getattr(sp, "thermo", None) is not None:
        H = sp.thermo.H(T_expr, R, dm.log)
    else:
        H = base_H(sp, theta)
        if getattr(sp, "Cp_param", None) is not None:
            H = H + sp.Cp_param * (T_expr - Tref)
    return H + delta_H(sp, theta)


def dH_rxn(rxn: Reaction, T_expr, R, Tref, theta=None):
    """Reaction enthalpy ``sum_i nu_i H_i(T)``."""
    return dm.sum([nu * h_species(s, T_expr, R, Tref, theta) for s, nu in rxn.net_stoich().items()])


def heat_release_rate(mkm: MicrokineticModel, conc, theta, free, T_expr, cat_density):
    """Volumetric heat release ``cat * sum_j (-dH_j) r_j`` (positive = exothermic)."""
    terms = []
    for rxn in mkm.reactions:
        r = rate_of_progress(rxn, conc, theta, free, T_expr, mkm.R, mkm.Tref)
        terms.append(-dH_rxn(rxn, T_expr, mkm.R, mkm.Tref, theta) * r)
    return cat_density * dm.sum(terms)


def mixture_heat_capacity(mkm: MicrokineticModel, conc):
    """Volumetric gas heat capacity ``sum_i C_i Cp_i``."""
    terms = []
    for g in mkm.gas_species:
        cp = g.Cp_param if getattr(g, "Cp_param", None) is not None else g.Cp
        terms.append(conc[g] * cp)
    return dm.sum(terms)
