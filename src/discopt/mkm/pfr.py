"""Plug-flow reactor: a spatial DAE solved with orthogonal collocation.

A steady-state PFR is a boundary-value problem in the axial coordinate ``z``:
the gas concentrations evolve along the reactor while the surface coverages sit
at a local quasi-steady state. We map this onto :mod:`discopt.dae` with ``z`` as
the continuous set, the gas concentrations as spatial *states*, and the
coverages (plus free-site coverage) as *algebraic* variables pinned by the
surface steady-state and site-balance equations at every collocation point::

    u dC_i/dz = cat_density * sum_j nu_ij r_j        (gas, states)
    0          = sum_j nu_aj r_j                      (each adsorbate, algebraic)
    1          = theta_free + sum theta_ads           (each site, algebraic)

An optional adiabatic energy balance carries temperature as an additional
spatial state (see ``energy``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import discopt.modeling as dm
from discopt.dae import ContinuousSet, DAEBuilder

from discopt.mkm.assemble import site_balance_residual
from discopt.mkm.energy import EnergyBalance, heat_release_rate, mixture_heat_capacity
from discopt.mkm.kinetics import net_rate
from discopt.mkm.model import MicrokineticModel, _safe
from discopt.mkm.transient import _aligned_grid, _aligned_state, _solve_feasibility


@dataclass
class PFRSolution:
    mkm: MicrokineticModel
    dm_model: dm.Model
    cs: ContinuousSet
    dae: DAEBuilder
    result: object
    _name_g: dict
    _name_a: dict
    _feed: dict
    _has_energy: bool

    @property
    def status(self) -> str:
        return self.result.status

    def positions(self):
        """1D axial grid, aligned point-for-point with the profile accessors."""
        return _aligned_grid(self.dae)

    def concentration(self, gas):
        """Axial concentration profile of a gas species (1D, aligned with
        :meth:`positions`)."""
        return _aligned_state(self.dae, self.result.x[self.dae.get_state(self._name_g[gas]).name])

    def outlet(self, gas) -> float:
        return float(np.asarray(self.concentration(gas)).reshape(-1)[-1])

    def conversion(self, gas) -> float:
        cin = float(self._feed.get(gas, 0.0))
        return (cin - self.outlet(gas)) / cin if cin > 0 else 0.0

    def coverage(self, adsorbate):
        """Axial coverage profile (1D, aligned with :meth:`positions`)."""
        return _aligned_state(self.dae, self.result.x[self.dae.get_state(self._name_a[adsorbate]).name])

    def temperature(self):
        """Axial temperature profile (1D, aligned with :meth:`positions`)."""
        if not self._has_energy:
            raise ValueError("isothermal PFR has no temperature profile")
        return _aligned_state(self.dae, self.result.x[self.dae.get_state("T").name])

    def outlet_temperature(self) -> float:
        return float(np.asarray(self.temperature()).reshape(-1)[-1])


def solve_pfr(
    mkm: MicrokineticModel,
    feed: dict,
    length: float,
    velocity: float,
    cat_density: float = 1.0,
    nfe: int = 40,
    ncp: int = 3,
    scheme: str = "radau",
    energy: EnergyBalance | None = None,
    nlp_solver: str = "pounce",
    solver_options: dict | None = None,
) -> PFRSolution:
    """Solve a steady-state plug-flow reactor.

    Parameters
    ----------
    feed : dict
        Inlet gas concentrations ``{gas_species: C_in}``.
    length : float
        Reactor length.
    velocity : float
        Superficial velocity ``u``.
    cat_density : float
        Active-site amount per reactor volume coupling surface rate to the gas.
    energy : EnergyBalance, optional
        If given, carries temperature as a spatial state with an adiabatic
        energy balance (otherwise isothermal at ``mkm.T``).
    """
    if any(r.equilibrated for r in mkm.reactions):
        raise ValueError("equilibrated (quasi-equilibrium) steps are only supported in steady-state solves")
    m = dm.Model(f"{mkm.name}_pfr")
    T_param = mkm.wire_parameters(m)
    cs = ContinuousSet("z", bounds=(0.0, length), nfe=nfe, ncp=ncp, scheme=scheme)
    dae = DAEBuilder(m, cs)

    for g in mkm.gas_species:
        dae.add_state(f"C_{_safe(g.name)}", bounds=(0.0, 1e6), initial=float(feed.get(g, 0.0)))
    if energy is not None:
        dae.add_state("T", bounds=(1.0, 1e5), initial=float(energy.T_in))
    for a in mkm.adsorbates:
        dae.add_algebraic(f"th_{_safe(a.name)}", bounds=(0.0, 1.0))
    for s in mkm.sites:
        dae.add_algebraic(f"fr_{_safe(s.name)}", bounds=(0.0, 1.0))

    name_g = {g: f"C_{_safe(g.name)}" for g in mkm.gas_species}
    name_a = {a: f"th_{_safe(a.name)}" for a in mkm.adsorbates}
    name_s = {s: f"fr_{_safe(s.name)}" for s in mkm.sites}

    def build(states, alg):
        conc = {g: states[name_g[g]] for g in mkm.gas_species}
        theta = {a: alg[name_a[a]] for a in mkm.adsorbates}
        free = {s: alg[name_s[s]] for s in mkm.sites}
        T_expr = states["T"] if energy is not None else T_param
        return conc, theta, free, T_expr

    def ode(z, s, a, c):
        conc, theta, free, T_expr = build(s, a)
        d = {}
        for g in mkm.gas_species:
            d[name_g[g]] = (cat_density / velocity) * net_rate(
                g, mkm.reactions, conc, theta, free, T_expr, mkm.R, mkm.Tref
            )
        if energy is not None:
            cp_mix = mixture_heat_capacity(mkm, conc)
            q = heat_release_rate(mkm, conc, theta, free, T_expr, cat_density)
            d["T"] = (q + energy.Q) / (velocity * cp_mix)
        return d

    def alg(z, s, a, c):
        conc, theta, free, T_expr = build(s, a)
        res = {
            name_a[ad]: net_rate(ad, mkm.reactions, conc, theta, free, T_expr, mkm.R, mkm.Tref)
            for ad in mkm.adsorbates
        }
        for st in mkm.sites:
            res[name_s[st]] = site_balance_residual(mkm, st, theta, free)
        return res

    dae.set_ode(ode)
    dae.set_algebraic(alg)
    dae.discretize()
    m.minimize(0.0)

    # physical warm start: inlet gas, inlet temperature, inlet-composition
    # coverages (discopt's default start would clip a ~500 K state to ~10).
    from discopt.mkm import numeric

    T0 = float(energy.T_in) if energy is not None else mkm.T
    seed = {a: 0.5 / max(len(mkm.adsorbates), 1) for a in mkm.adsorbates}
    theta_g, free_g = numeric.steady_state_numeric(mkm, feed, T0, theta0=seed)
    fills = {dae.get_state(name_g[g]).name: float(feed.get(g, 0.0)) for g in mkm.gas_species}
    if energy is not None:
        fills[dae.get_state("T").name] = T0
    for a in mkm.adsorbates:
        fills[dae.get_state(name_a[a]).name] = float(max(theta_g[a], 1e-9))
    for s in mkm.sites:
        fills[dae.get_state(name_s[s]).name] = float(max(free_g[s], 1e-9))

    result = _solve_feasibility(m, nlp_solver=nlp_solver, solver_options=solver_options, fills=fills)
    return PFRSolution(mkm, m, cs, dae, result, name_g, name_a, dict(feed), energy is not None)
