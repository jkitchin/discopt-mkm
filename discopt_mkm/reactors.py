"""Reactor models that supply gas-phase treatment and mole balances.

Each reactor is reactor-agnostic about the surface chemistry: it only decides
how the gas-phase concentrations enter the model.

Conventions
-----------
Rates of progress are on a *per active site* basis. Hence an adsorbate coverage
evolves as ``dtheta_i/dt = sum_j nu_ij r_j`` directly, while a gas concentration
gains ``cat_density * sum_j nu_ij r_j`` from the catalyst plus any flow term,
where ``cat_density`` is the active-site amount per reactor volume. Units are the
caller's responsibility (see the package README).
"""

from __future__ import annotations

import re

import discopt.modeling as dm

from discopt_mkm.kinetics import net_rate
from discopt_mkm.model import MicrokineticModel


def _safe(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", name)


class Reactor:
    """Base reactor. Subclasses define gas treatment for steady/transient solves."""

    dynamic_gas = False

    def create_gas(self, m, mkm: MicrokineticModel) -> dict:
        """Create and return the gas concentration expressions ``{gas: expr}``."""
        raise NotImplementedError

    def gas_residuals(self, conc, theta, free_cov, T_param, mkm: MicrokineticModel, extents=None) -> list:
        """Steady-state gas balance residual expressions (each constrained ``== 0``)."""
        return []

    # transient hooks (defaults: gas is constant) ------------------------
    def initial_concentration(self, g) -> float:
        return 0.0

    def gas_rhs(self, g, conc, theta, free_cov, T_param, mkm: MicrokineticModel):
        """``dC_g/dt`` contribution for a dynamic gas state."""
        return 0.0


class DifferentialReactor(Reactor):
    """Gas held at fixed partial pressures / concentrations.

    This is the canonical setting for steady-state coverage solves and Campbell
    degree of rate control: the gas conditions are fixed (but exposed as
    differentiable parameters), and only the surface coverages are unknowns.

    Parameters
    ----------
    pressures : dict
        Mapping ``{gas_species: value}`` of fixed gas concentration / partial
        pressure. Missing species default to 0.
    """

    dynamic_gas = False

    def __init__(self, pressures: dict):
        self.pressures = dict(pressures)
        self._params: dict = {}

    def create_gas(self, m, mkm: MicrokineticModel) -> dict:
        self._params = {
            g: m.parameter(f"P_{_safe(g.name)}", float(self.pressures.get(g, 0.0)))
            for g in mkm.gas_species
        }
        return self._params

    def initial_concentration(self, g) -> float:
        return float(self.pressures.get(g, 0.0))


class CSTR(Reactor):
    """Continuous stirred-tank reactor with inflow/outflow and a catalyst.

    Steady-state gas balance per species::

        (C_in - C) / tau + cat_density * sum_j nu_ij r_j = 0

    Parameters
    ----------
    inlet : dict
        Inlet concentrations ``{gas_species: value}``.
    tau : float
        Residence time.
    cat_density : float
        Active-site amount per reactor volume coupling surface rate to the gas
        balance. Default 1.
    """

    dynamic_gas = True

    def __init__(self, inlet: dict, tau: float, cat_density: float = 1.0):
        self.inlet = dict(inlet)
        self.tau = float(tau)
        self.cat = float(cat_density)

    def create_gas(self, m, mkm: MicrokineticModel) -> dict:
        return {g: m.continuous(f"C_{_safe(g.name)}", lb=0.0, ub=1e6) for g in mkm.gas_species}

    def gas_residuals(self, conc, theta, free_cov, T_param, mkm: MicrokineticModel, extents=None) -> list:
        res = []
        for g in mkm.gas_species:
            Cin = float(self.inlet.get(g, 0.0))
            rxn_term = self.cat * net_rate(
                g, mkm.reactions, conc, theta, free_cov, T_param, mkm.R, mkm.Tref, extents
            )
            res.append((Cin - conc[g]) / self.tau + rxn_term)
        return res

    def initial_concentration(self, g) -> float:
        return float(self.inlet.get(g, 0.0))

    def gas_rhs(self, g, conc, theta, free_cov, T_param, mkm: MicrokineticModel):
        Cin = float(self.inlet.get(g, 0.0))
        rxn_term = self.cat * net_rate(
            g, mkm.reactions, conc, theta, free_cov, T_param, mkm.R, mkm.Tref
        )
        return (Cin - conc[g]) / self.tau + rxn_term


def levich_coefficient(D: float, omega: float, kinematic_viscosity: float = 1.0e-2) -> float:
    """Levich mass-transfer coefficient of a rotating disk electrode,
    ``k_m = 0.62 D^(2/3) nu^(-1/6) omega^(1/2)``.

    ``D`` diffusivity, ``nu`` kinematic viscosity, ``omega`` angular rotation rate
    (rad/s). Units are the caller's responsibility; with CGS (cm^2/s, cm^2/s,
    rad/s) ``k_m`` is in cm/s and the limiting current is ``n F k_m C_bulk``.
    """
    return 0.620 * D ** (2.0 / 3.0) * kinematic_viscosity ** (-1.0 / 6.0) * omega ** 0.5


class MassTransferReactor(Reactor):
    """Solution species coupled to a well-mixed bulk through a finite mass-transfer
    coefficient (a stagnant Nernst diffusion layer).

    Transport-limited species' *surface* activities are unknowns set by balancing
    the diffusion flux against consumption; every other species is held at its
    bulk value (a fixed parameter). Per transport-limited species ``i`` the
    steady-state balance is::

        k_m,i (C_bulk,i - C_surf,i) + cat_density * sum_j nu_ij r_j = 0

    so at large reaction rate ``C_surf -> 0`` and the current saturates at the
    mass-transport limit ``j_lim,i = n F k_m,i C_bulk,i``. Combined with the
    surface (Butler-Volmer) kinetics this is the Koutecky-Levich mixed-control
    picture ``1/j = 1/j_kinetic + 1/j_lim``. Works for surface mechanisms and for
    adsorbate-free *outer-sphere* electron transfers.

    Parameters
    ----------
    bulk : dict
        Bulk activities ``{species: C_bulk}`` (missing species default to 0).
    mass_transfer : dict
        ``{species: k_m}`` for the transport-limited species. Species absent here
        are held fixed at their bulk value.
    cat_density : float
        Surface-rate-to-flux scale (1.0 for an outer-sphere reaction whose rate is
        already a flux; the active-site density for an adsorbed mechanism).
    """

    dynamic_gas = True

    def __init__(self, bulk: dict, mass_transfer: dict, cat_density: float = 1.0):
        self.bulk = dict(bulk)
        self.km = dict(mass_transfer)
        self.cat = float(cat_density)

    def create_gas(self, m, mkm: MicrokineticModel) -> dict:
        out = {}
        for g in mkm.gas_species:
            if g in self.km:
                out[g] = m.continuous(f"Cs_{_safe(g.name)}", lb=0.0, ub=1e6)
            else:
                out[g] = m.parameter(f"Cb_{_safe(g.name)}", float(self.bulk.get(g, 0.0)))
        return out

    def gas_residuals(self, conc, theta, free_cov, T_param, mkm: MicrokineticModel, extents=None) -> list:
        res = []
        for g in mkm.gas_species:
            if g in self.km:
                Cb = float(self.bulk.get(g, 0.0))
                rxn_term = self.cat * net_rate(
                    g, mkm.reactions, conc, theta, free_cov, T_param, mkm.R, mkm.Tref, extents
                )
                res.append(float(self.km[g]) * (Cb - conc[g]) + rxn_term)
        return res

    def initial_concentration(self, g) -> float:
        return float(self.bulk.get(g, 0.0))

    def gas_rhs(self, g, conc, theta, free_cov, T_param, mkm: MicrokineticModel):
        if g not in self.km:
            return 0.0
        Cb = float(self.bulk.get(g, 0.0))
        rxn_term = self.cat * net_rate(g, mkm.reactions, conc, theta, free_cov, T_param, mkm.R, mkm.Tref)
        return float(self.km[g]) * (Cb - conc[g]) + rxn_term


class RotatingDiskElectrode(MassTransferReactor):
    """A rotating disk electrode: a :class:`MassTransferReactor` whose mass-transfer
    coefficients are the Levich values ``k_m = 0.62 D^(2/3) nu^(-1/6) omega^(1/2)``.

    Parameters
    ----------
    bulk : dict
        Bulk activities ``{species: C_bulk}``.
    omega : float
        Angular rotation rate (rad/s).
    diffusivities : dict
        ``{species: D}`` for the transport-limited species (those coupled to the
        bulk through the rotating diffusion layer).
    kinematic_viscosity : float
        ``nu`` of the electrolyte (default 1e-2, water in CGS).
    cat_density : float
        See :class:`MassTransferReactor`.
    """

    def __init__(self, bulk: dict, omega: float, diffusivities: dict,
                 kinematic_viscosity: float = 1.0e-2, cat_density: float = 1.0):
        self.omega = float(omega)
        self.diffusivities = dict(diffusivities)
        self.kinematic_viscosity = float(kinematic_viscosity)
        km = {g: levich_coefficient(float(D), self.omega, kinematic_viscosity)
              for g, D in diffusivities.items()}
        super().__init__(bulk, km, cat_density)

    def limiting_current(self, species, n_electrons: float = 1.0, F: float = 1.0) -> float:
        """Levich limiting current ``n F k_m C_bulk`` for a transport-limited species."""
        return float(n_electrons * F * self.km[species] * self.bulk.get(species, 0.0))


class Batch(Reactor):
    """Closed, fixed-volume batch reactor (transient only).

    Gas concentrations evolve by reaction alone::

        dC_g/dt = cat_density * sum_j nu_ij r_j
    """

    dynamic_gas = True

    def __init__(self, initial: dict, cat_density: float = 1.0):
        self.initial = dict(initial)
        self.cat = float(cat_density)

    def create_gas(self, m, mkm: MicrokineticModel) -> dict:
        # batch has no nontrivial steady state; provided for API symmetry
        return {g: m.continuous(f"C_{_safe(g.name)}", lb=0.0, ub=1e6) for g in mkm.gas_species}

    def initial_concentration(self, g) -> float:
        return float(self.initial.get(g, 0.0))

    def gas_rhs(self, g, conc, theta, free_cov, T_param, mkm: MicrokineticModel):
        return self.cat * net_rate(
            g, mkm.reactions, conc, theta, free_cov, T_param, mkm.R, mkm.Tref
        )
