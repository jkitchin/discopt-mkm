"""Steady-state solve: coverages (and CSTR gas) as a feasibility NLP.

discopt has no standalone root-finder, so the square steady-state system is
posed as a feasibility problem with a constant (zero) objective and solved
through ``differentiable_solve_l3``, whose KKT/implicit-function-theorem
machinery also yields ``dx*/dp`` for degree of rate control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import discopt.modeling as dm
from discopt._jax.differentiable import differentiable_solve_l3

from discopt.mkm.assemble import quasi_equilibrium, site_balance_residual
from discopt.mkm.kinetics import net_rate, rate_of_progress
from discopt.mkm.model import MicrokineticModel, _safe
from discopt.mkm.reactors import Reactor

from discopt.modeling.core import Expression  # type: ignore


def _scalar(v) -> float:
    """Coerce a discopt scalar result (often stored as a shape-(1,) array)."""
    return float(np.asarray(v).reshape(-1)[0])


@dataclass
class SteadyStateSolution:
    """Bundle of everything produced by :func:`solve_steady_state`.

    Holds the discopt model, the L3 differentiable result, and the variable /
    parameter maps so that :mod:`discopt.mkm.analysis.drc` can build output-rate
    expressions and differentiate them through the steady state.
    """

    mkm: MicrokineticModel
    dm_model: dm.Model
    result: object
    T_param: object
    conc: dict
    theta: dict
    free_cov: dict
    reactor: Reactor
    extents: dict = None
    U_param: object = None   # electrode-potential handle (electrochemical models)

    # -- convenience accessors -------------------------------------------
    @property
    def status(self) -> str:
        return self.result.status

    @property
    def sensitivities_available(self) -> bool:
        """True if L3 implicit differentiation succeeded (DRC/TRC usable)."""
        return self.result.sensitivity_matrix() is not None

    def coverage(self, adsorbate) -> float:
        # evaluate the stored activity expression; works for linear (a Variable)
        # and log (an exp(z)) coordinates alike.
        return self._eval(self.theta[adsorbate])

    def free_coverage(self, site) -> float:
        return self._eval(self.free_cov[site])

    def gas_concentration(self, gas) -> float:
        return self._eval(self.conc[gas])

    def rate_of_progress_expr(self, rxn):
        """discopt expression for reaction ``rxn``'s rate of progress at the solution."""
        return rate_of_progress(
            rxn, self.conc, self.theta, self.free_cov, self.T_param, self.mkm.R, self.mkm.Tref,
            self.extents,
        )

    def rate_of_progress(self, rxn) -> float:
        return self._eval(self.rate_of_progress_expr(rxn))

    def production_rate_expr(self, species):
        """discopt expression for the net production rate of ``species``."""
        return net_rate(
            species, self.mkm.reactions, self.conc, self.theta, self.free_cov,
            self.T_param, self.mkm.R, self.mkm.Tref, self.extents,
        )

    def production_rate(self, species) -> float:
        return self._eval(self.production_rate_expr(species))

    def temperature(self) -> float:
        """Solved temperature (constant for isothermal, the energy-balance T otherwise)."""
        return self._eval(self.T_param)

    def _eval(self, expr) -> float:
        from discopt.mkm.analysis.sensitivity import evaluate_expression

        return evaluate_expression(expr, self.result, self.dm_model)

    def to_dict(self) -> dict:
        """JSON-serializable summary of the solved steady state."""
        return {
            "status": self.status,
            "temperature": self.temperature(),
            "coverages": {a.name: self.coverage(a) for a in self.mkm.adsorbates},
            "free_coverage": {s.name: self.free_coverage(s) for s in self.mkm.sites},
            "gas": {g.name: self.gas_concentration(g) for g in self.mkm.gas_species},
            "rates_of_progress": {r.name: self.rate_of_progress(r) for r in self.mkm.reactions},
            "sensitivities_available": self.sensitivities_available,
        }

    def to_html(self) -> str:
        from discopt.mkm import render

        return render.solution_html(self)

    def _repr_html_(self) -> str:
        return self.to_html()


def solve_steady_state(
    mkm: MicrokineticModel,
    reactor: Reactor,
    method: str = "auto",
    coordinates: str = "linear",
    theta0: dict | None = None,
    log_box: float = 6.0,
    reg_weight: float = 1.0,
    energy=None,
    active_tol: float = 1e-3,
    nlp_solver: str = "pounce",
    solver_options: dict | None = None,
) -> SteadyStateSolution:
    """Build and solve the steady-state model for ``mkm`` in ``reactor``.

    Parameters
    ----------
    method : {"auto", "feasibility", "least_squares"}
        Used in ``coordinates="linear"`` mode. ``"feasibility"`` poses the square
        root-find with a zero objective (most accurate sensitivities);
        ``"least_squares"`` minimizes the residual norm subject to site
        conservation (robust near saturated coverages); ``"auto"`` tries
        feasibility then falls back.
    coordinates : {"linear", "log"}
        ``"log"`` solves in log-coverages ``z = ln(theta)`` with a search box and
        a regularizer centered on the warm start ``theta0``. This is required for
        stiff, near-equilibrium mechanisms (e.g. water-gas shift) where physical
        coverages span many orders of magnitude and a naive linear solve drifts
        to a poisoned spurious root.
    theta0 : dict, optional
        Warm start ``{adsorbate: coverage}`` (required for ``coordinates="log"``);
        free-site coverages are inferred from the site balance. Obtain one from
        :func:`discopt.mkm.numeric.steady_state_numeric` or a transient solve.
    log_box : float
        Half-width (in natural-log units) of the coverage search box around the
        warm start, for ``coordinates="log"``.
    reg_weight : float
        Weight of the ``sum (z - z0)^2`` regularizer toward the warm start.
    active_tol : float
        Bound-activity tolerance for the L3 implicit-differentiation sensitivity.
        At a physical MKM steady state no coverage is ever genuinely at a 0/1
        bound, so a *small* value avoids false positives. Increase only the
        rare case where a bound is truly active. Tiny coverages (e.g. ~1e-12 in
        linear coordinates) need ``active_tol`` below them for correct degree of
        rate control.
    """
    if coordinates == "log":
        return _solve_log(
            mkm, reactor, theta0, log_box, reg_weight, active_tol, nlp_solver, solver_options
        )

    def build(least_squares: bool):
        m = dm.Model(f"{mkm.name}_ss")
        T_param = mkm.wire_parameters(m)
        # non-isothermal: temperature becomes an unknown driven by an energy balance
        T_expr = m.continuous("T_var", lb=1.0, ub=1e5) if energy is not None else T_param
        theta = {a: m.continuous(f"theta_{_safe(a.name)}", lb=0.0, ub=1.0) for a in mkm.adsorbates}
        free_cov = {s: m.continuous(f"thetafree_{_safe(s.name)}", lb=0.0, ub=1.0) for s in mkm.sites}
        conc = reactor.create_gas(m, mkm)

        # quasi-equilibrated steps contribute an unknown extent + equilibrium relation
        extents, eq_residuals = quasi_equilibrium(m, mkm, conc, theta, free_cov, T_expr)

        # steady-state residuals: net adsorbate production + reactor gas balance
        residuals = []
        for a in mkm.adsorbates:
            r = net_rate(a, mkm.reactions, conc, theta, free_cov, T_expr, mkm.R, mkm.Tref, extents)
            if isinstance(r, Expression):
                residuals.append(r)
        residuals.extend(reactor.gas_residuals(conc, theta, free_cov, T_expr, mkm, extents))
        residuals.extend(eq_residuals)

        # energy balance closes the system when temperature is unknown
        if energy is not None:
            residuals.append(_cstr_energy_residual(mkm, reactor, conc, theta, free_cov, T_expr, energy))

        # site conservation is always a hard (linear) constraint
        for s in mkm.sites:
            m.subject_to(
                site_balance_residual(mkm, s, theta, free_cov) == 0.0, name=f"site_{_safe(s.name)}"
            )

        if least_squares:
            m.minimize(dm.sum([r * r for r in residuals]))
        else:
            for k, r in enumerate(residuals):
                m.subject_to(r == 0.0, name=f"ss_{k}")
            m.minimize(0.0)
        return m, T_expr, conc, theta, free_cov, extents

    def run(least_squares: bool):
        m, T_param, conc, theta, free_cov, extents = build(least_squares)
        result = differentiable_solve_l3(
            m, active_tol=active_tol, nlp_solver=nlp_solver, solver_options=solver_options or {}
        )
        return SteadyStateSolution(mkm, m, result, T_param, conc, theta, free_cov, reactor,
                                   extents, U_param=mkm._U_param)

    if method == "least_squares":
        return run(least_squares=True)
    if method == "feasibility":
        return run(least_squares=False)
    if method == "auto":
        try:
            return run(least_squares=False)
        except RuntimeError:
            return run(least_squares=True)
    raise ValueError(f"unknown method {method!r}")


def _cstr_energy_residual(mkm, reactor, conc, theta, free_cov, T_expr, energy):
    """CSTR adiabatic/heated energy balance residual (== 0 at steady state).

    ``(sum_i C_in,i Cp_i)(T_in - T)/tau + cat * sum_j (-dH_j) r_j + Q``
    """
    from discopt.mkm.energy import heat_release_rate

    if not hasattr(reactor, "inlet") or not hasattr(reactor, "tau"):
        raise ValueError("energy balance requires a CSTR reactor (inlet, tau)")
    cp_in = dm.sum(
        [
            float(reactor.inlet.get(g, 0.0))
            * (g.Cp_param if getattr(g, "Cp_param", None) is not None else g.Cp)
            for g in mkm.gas_species
        ]
    )
    q = heat_release_rate(mkm, conc, theta, free_cov, T_expr, reactor.cat)
    return cp_in * (energy.T_in - T_expr) / reactor.tau + q + energy.Q


def _solve_log(mkm, reactor, theta0, log_box, reg_weight, active_tol, nlp_solver, solver_options):
    """Log-coverage steady-state solve with a warm-start box and regularizer."""
    from discopt.mkm import numeric

    if theta0 is None:
        raise ValueError("coordinates='log' requires a warm start theta0")
    if any(r.equilibrated for r in mkm.reactions):
        raise ValueError(
            "quasi-equilibrium (equilibrated steps) is supported with "
            "coordinates='linear' — it removes the stiffness that log coordinates "
            "are for, so the two are not combined."
        )

    # complete the warm start: free-site coverages from the site balance
    free0 = {s: max(1.0 - sum(theta0[a] for a in mkm.adsorbates_on(s)), 1e-300) for s in mkm.sites}
    # numeric one-way forward magnitudes at the warm start -> residual scaling
    T = mkm.T
    pressures = getattr(reactor, "pressures", {})
    kf, kr = numeric.rate_constants(mkm, T, theta0)
    conc0 = {g: float(pressures.get(g, 0.0)) for g in mkm.gas_species}

    def fwd_mag(a):
        mags = [1e-30]
        for rxn in mkm.reactions:
            nu = rxn.net_stoich().get(a, 0.0)
            if nu != 0.0:
                f = kf[rxn] * numeric._prod(rxn.reactants, theta0, free0, conc0)
                mags.append(abs(nu) * abs(f))
        return max(mags)

    m = dm.Model(f"{mkm.name}_ss_log")
    T_param = mkm.wire_parameters(m)

    z = {}
    theta = {}
    for a in mkm.adsorbates:
        z0 = float(np.log(max(theta0[a], 1e-300)))
        za = m.continuous(f"z_{_safe(a.name)}", lb=z0 - log_box, ub=min(0.0, z0 + log_box))
        z[a] = (za, z0)
        theta[a] = dm.exp(za)
    free_cov = {}
    for s in mkm.sites:
        z0 = float(np.log(free0[s]))
        zs = m.continuous(f"zfree_{_safe(s.name)}", lb=z0 - log_box, ub=min(0.0, z0 + log_box))
        z[s] = (zs, z0)
        free_cov[s] = dm.exp(zs)

    conc = reactor.create_gas(m, mkm)

    # scaled steady-state residuals (O(1)) as hard constraints
    for a in mkm.adsorbates:
        r = net_rate(a, mkm.reactions, conc, theta, free_cov, T_param, mkm.R, mkm.Tref)
        if isinstance(r, Expression):
            m.subject_to(r / fwd_mag(a) == 0.0, name=f"ss_{_safe(a.name)}")
    for s in mkm.sites:
        m.subject_to(site_balance_residual(mkm, s, theta, free_cov) == 0.0, name=f"site_{_safe(s.name)}")

    # regularizer toward the warm start picks the physical root
    m.minimize(reg_weight * dm.sum([(za - z0) ** 2 for (za, z0) in z.values()]))

    result = differentiable_solve_l3(
        m, active_tol=active_tol, nlp_solver=nlp_solver, solver_options=solver_options or {}
    )
    return SteadyStateSolution(mkm, m, result, T_param, conc, theta, free_cov, reactor,
                               U_param=mkm._U_param)
