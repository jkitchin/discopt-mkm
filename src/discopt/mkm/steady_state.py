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
    # Snapshots captured at construction time so this solution stays valid after
    # the model is re-solved. ``wire_parameters`` overwrites the parameter handles
    # stored on the shared ``Species``/``Reaction`` objects on every solve, so a
    # rate expression rebuilt *later* from those live handles would reference a
    # different discopt model and break (``Parameter 'A_2' not found in model``).
    # We therefore build every rate-of-progress / production-rate expression and
    # snapshot the rate-constant / free-energy handles now, while the handles
    # still belong to *this* solution's model.
    rop_exprs: dict = None
    prod_exprs: dict = None
    rate_constant_params: dict = None
    dG_params: dict = None

    def __post_init__(self):
        self._capture_expressions()

    def _capture_expressions(self) -> None:
        """Eagerly build and store this solution's rate expressions and parameter
        handles, decoupling it from later re-solves that rewire the shared objects."""
        R, Tref = self.mkm.R, self.mkm.Tref
        self.rop_exprs = {
            rxn: rate_of_progress(rxn, self.conc, self.theta, self.free_cov,
                                  self.T_param, R, Tref, self.extents)
            for rxn in self.mkm.reactions
        }
        # Reuse the snapshot's rate-of-progress expressions in the production-rate
        # expressions instead of rebuilding each reaction subtree per species. The
        # cached objects are the exact same expressions ``net_rate`` would build,
        # so the production/DRC results are identical.
        self.prod_exprs = {
            sp: net_rate(sp, self.mkm.reactions, self.conc, self.theta, self.free_cov,
                         self.T_param, R, Tref, self.extents, rate_cache=self.rop_exprs)
            for sp in self.mkm.species
        }
        self.rate_constant_params = {rxn: rxn.rate_constant_param() for rxn in self.mkm.reactions}
        self.dG_params = {sp: sp.dG_param for sp in self.mkm.species}

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
        """discopt expression for reaction ``rxn``'s rate of progress at the solution.

        Returns the snapshot captured at construction time (stable across later
        re-solves); falls back to a fresh build only if the reaction is unknown.
        """
        if self.rop_exprs is not None and rxn in self.rop_exprs:
            return self.rop_exprs[rxn]
        return rate_of_progress(
            rxn, self.conc, self.theta, self.free_cov, self.T_param, self.mkm.R, self.mkm.Tref,
            self.extents,
        )

    def rate_of_progress(self, rxn) -> float:
        return self._eval(self.rate_of_progress_expr(rxn))

    def production_rate_expr(self, species):
        """discopt expression for the net production rate of ``species``.

        Returns the snapshot captured at construction time (stable across later
        re-solves); falls back to a fresh build only for an unknown species.
        """
        if self.prod_exprs is not None and species in self.prod_exprs:
            return self.prod_exprs[species]
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
        linear coordinates) fall *below* the default ``1e-3`` and are then flagged
        as bound-active, which nulls or degrades their degree of rate control. For
        such near-zero coverages either pass a smaller ``active_tol`` (below the
        smallest physical coverage) or, better, solve in ``coordinates="log"``
        with a warm start, where coverages are represented as ``exp(z)`` and are
        never near the linear 0-bound.
    """
    if not reactor.supports_steady_state:
        raise ValueError(
            "batch reactor has no steady state; use solve_transient "
            "(a closed batch reactor's only steady state is complete "
            "consumption/equilibrium)"
        )
    if coordinates == "log":
        if energy is not None:
            raise ValueError("energy balance is not supported with coordinates='log'")
        return _solve_log(
            mkm, reactor, theta0, log_box, reg_weight, active_tol, nlp_solver, solver_options
        )

    def build(least_squares: bool):
        m = dm.Model(f"{mkm.name}_ss")
        T_param = mkm.wire_parameters(m)
        # non-isothermal: temperature becomes an unknown driven by an energy
        # balance. discopt's ``_safe_x0`` clips every variable's start to [-10, 10]
        # in its *own* units, so a bare ``T_var`` in kelvin would begin near 10 K
        # where Arrhenius factors underflow (numeric.py / transient.py / pfr.py all
        # special-case this). We instead carry a dimensionless ``T / T_nom`` whose
        # O(1) scale starts at a physical temperature: with the bounds below the
        # clipped start lands a few times ``T_nom`` rather than at 10 K.
        if energy is not None:
            T_nom = float(getattr(energy, "T_in", 0.0)) or mkm.T
            T_scaled = m.continuous("T_scaled", lb=1.0 / T_nom, ub=4.0)
            T_expr = T_scaled * T_nom
        else:
            T_expr = T_param
        theta = {a: m.continuous(f"theta_{_safe(a.name)}", lb=0.0, ub=1.0) for a in mkm.adsorbates}
        free_cov = {s: m.continuous(f"thetafree_{_safe(s.name)}", lb=0.0, ub=1.0) for s in mkm.sites}
        conc = reactor.create_gas(m, mkm)

        # quasi-equilibrated steps contribute an unknown extent + equilibrium relation
        extents, eq_residuals = quasi_equilibrium(m, mkm, conc, theta, free_cov, T_expr)

        # Build each reaction's rate of progress ONCE and share it across every
        # adsorbate and gas balance (see kinetics.net_rate). Identical expressions,
        # smaller graph.
        rate_cache = {
            rxn: rate_of_progress(rxn, conc, theta, free_cov, T_expr, mkm.R, mkm.Tref, extents)
            for rxn in mkm.reactions
        }

        # steady-state residuals: net adsorbate production + reactor gas balance
        residuals = []
        for a in mkm.adsorbates:
            r = net_rate(a, mkm.reactions, conc, theta, free_cov, T_expr, mkm.R, mkm.Tref, extents,
                         rate_cache=rate_cache)
            if isinstance(r, Expression):
                residuals.append(r)
        residuals.extend(reactor.gas_residuals(conc, theta, free_cov, T_expr, mkm, extents,
                                               rate_cache=rate_cache))
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
        if least_squares:
            # a least-squares solve reports "optimal" at *any* local minimum of the
            # residual norm; the objective here IS the residual sum of squares, so
            # verify it is essentially zero rather than trusting the status. At a
            # true steady state the net rates vanish, so this is an absolute check.
            sse = result.objective
            if sse is not None and float(sse) > 1e-9:
                raise RuntimeError(
                    f"least-squares steady-state solve converged to a non-steady point "
                    f"(residual sum of squares {float(sse):.3e} >> 0); the reported "
                    "'optimal' status is a local minimum of the residual norm, not a "
                    "steady state. Try coordinates='log' with a warm start."
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
    # inlet-stream heat capacity. A species whose Cp lives in a temperature-
    # dependent thermo model has ``Cp_param is None``; falling back to the constant
    # ``g.Cp`` (0.0) there silently zeros the flow term and degenerates the balance
    # to ``q + Q = 0`` (unphysical solved T). Take Cp from the thermo model instead,
    # evaluated at the inlet temperature (the stream entering at ``T_in``).
    cp_terms = []
    for g in mkm.gas_species:
        amount = float(reactor.inlet.get(g, 0.0))
        if amount == 0.0:
            continue
        if getattr(g, "thermo", None) is not None:
            cp = g.thermo.Cp(energy.T_in, mkm.R)
        elif getattr(g, "Cp_param", None) is not None:
            cp = g.Cp_param
        else:
            cp = g.Cp
        cp_terms.append(amount * cp)
    cp_in = dm.sum(cp_terms) if cp_terms else 0.0
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
    # numeric one-way flux magnitudes at the warm start -> residual scaling.
    # The nominal gas composition is reactor-specific (fixed pressures for a
    # differential reactor, the inlet feed for a CSTR, the bulk for a
    # mass-transfer / RDE cell); a bare ``getattr(reactor, "pressures", ...)``
    # would read an all-zero gas for any flow reactor and center the search box
    # at log(0).
    T = mkm.T
    nominal = reactor.nominal_gas(mkm)
    kf, kr = numeric.rate_constants(mkm, T, theta0)
    conc0 = {g: float(nominal.get(g, 0.0)) for g in mkm.gas_species}

    def flux_mag(sp):
        """Largest one-way (forward *or* reverse) flux touching ``sp`` at the warm
        start. Including the reverse term is essential: for a reverse-dominated or
        zero-reactant warm start the forward-only magnitude collapses to the
        floor and the scaled residual becomes ~1e30-overscaled."""
        mags = [1e-30]
        for rxn in mkm.reactions:
            nu = rxn.net_stoich().get(sp, 0.0)
            if nu != 0.0:
                f = kf[rxn] * numeric._prod(rxn.reactants, theta0, free0, conc0)
                r = kr[rxn] * numeric._prod(rxn.products, theta0, free0, conc0)
                mags.append(abs(nu) * abs(f))
                mags.append(abs(nu) * abs(r))
        return max(mags)

    fwd_mag = flux_mag  # backward-compatible alias

    m = dm.Model(f"{mkm.name}_ss_log")
    T_param = mkm.wire_parameters(m)

    # coverages at/below this floor carry no trustworthy warm-start information
    # (the numeric seed returned ~0, common for minor intermediates). Pinning
    # them in a degenerate trust box at log(0) ≈ -690 excludes their true (small
    # but non-zero) value and makes the NLP infeasible ("solver error"), so
    # instead center the regularizer at the floor and let the variable range over
    # the whole physical interval.
    FLOOR = 1e-12
    z = {}
    theta = {}
    for a in mkm.adsorbates:
        th0a = theta0[a]
        if th0a > FLOOR:
            z0 = float(np.log(th0a))
            lb, ub = z0 - log_box, min(0.0, z0 + log_box)
        else:
            z0 = float(np.log(FLOOR))
            lb, ub = float(np.log(1e-30)), 0.0
        za = m.continuous(f"z_{_safe(a.name)}", lb=lb, ub=ub)
        z[a] = (za, z0)
        theta[a] = dm.exp(za)
    free_cov = {}
    for s in mkm.sites:
        z0 = float(np.log(free0[s]))
        zs = m.continuous(f"zfree_{_safe(s.name)}", lb=z0 - log_box, ub=min(0.0, z0 + log_box))
        z[s] = (zs, z0)
        free_cov[s] = dm.exp(zs)

    conc = reactor.create_gas(m, mkm)

    # share each reaction's rate of progress across the adsorbate + gas balances
    # (see kinetics.net_rate); log mode rejects equilibrated steps, so no extents.
    rate_cache = {
        rxn: rate_of_progress(rxn, conc, theta, free_cov, T_param, mkm.R, mkm.Tref)
        for rxn in mkm.reactions
    }

    # scaled steady-state residuals (O(1)) as hard constraints
    for a in mkm.adsorbates:
        r = net_rate(a, mkm.reactions, conc, theta, free_cov, T_param, mkm.R, mkm.Tref,
                     rate_cache=rate_cache)
        if isinstance(r, Expression):
            m.subject_to(r / flux_mag(a) == 0.0, name=f"ss_{_safe(a.name)}")
    for s in mkm.sites:
        m.subject_to(site_balance_residual(mkm, s, theta, free_cov) == 0.0, name=f"site_{_safe(s.name)}")

    # reactor gas balance, exactly as the linear path imposes it. Without this a
    # flow reactor's (CSTR / mass-transfer / RDE) gas concentrations are free
    # variables constrained by nothing and drift to their bound midpoints. Added
    # unscaled (like the linear path) — the balance already mixes an O(1) flow /
    # transport term with the reaction term, so its natural scale is O(1).
    for k, gres in enumerate(
        reactor.gas_residuals(conc, theta, free_cov, T_param, mkm, extents=None, rate_cache=rate_cache)
    ):
        m.subject_to(gres == 0.0, name=f"gas_{k}")

    # regularizer toward the warm start picks the physical root
    m.minimize(reg_weight * dm.sum([(za - z0) ** 2 for (za, z0) in z.values()]))

    result = differentiable_solve_l3(
        m, active_tol=active_tol, nlp_solver=nlp_solver, solver_options=solver_options or {}
    )
    return SteadyStateSolution(mkm, m, result, T_param, conc, theta, free_cov, reactor,
                               U_param=mkm._U_param)
