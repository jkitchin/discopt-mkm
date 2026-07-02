"""Transient solve via discopt orthogonal-collocation DAE discretization.

Reuses the exact same rate core (:mod:`discopt.mkm.kinetics`) as the
steady-state path. The DAE right-hand side is invoked once with vector-shaped
state entries, so the rate builders must be (and are) written with element-wise
discopt operations only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import discopt.modeling as dm
from discopt.dae import ContinuousSet, DAEBuilder

from discopt.mkm.assemble import site_balance_residual
from discopt.mkm.kinetics import net_rate
from discopt.mkm.model import MicrokineticModel, _safe
from discopt.mkm.reactors import Reactor


def _aligned_grid(dae, include_start=False):
    """1D grid of the ``nfe*ncp`` collocation points, in order, that lines up
    point-for-point with the flattened values from :func:`_aligned_state`.

    Both *state* variables (shape ``(nfe, ncp+1)``: an element-start column plus
    the ``ncp`` collocation columns) and *algebraic* variables (shape
    ``(nfe, ncp)``: collocation columns only) are reported on this common
    collocation grid, so any pair of profiles can be plotted against it.
    (``dae.time_points()`` deduplicates boundaries differently and does *not*
    line up with either raw block.)

    ``include_start=True`` prepends the very first element-start point (the
    initial condition / inlet). Use it for a transient solve, where every
    coverage is a state and showing the ``t=0`` value matters; leave it False for
    the PFR, where states and algebraics must share one grid.
    """
    ep = np.asarray(dae._element_points())  # (nfe, ncp+1)
    coll = ep[:, 1:].ravel()
    return np.concatenate([ep[:1, 0].ravel(), coll]) if include_start else coll


def _aligned_state(dae, arr, include_start=False):
    """Flatten a collocation block to a 1D profile on the :func:`_aligned_grid`
    collocation points, handling both state ``(nfe, ncp+1)`` and algebraic
    ``(nfe, ncp)`` arrays. ``include_start`` prepends the element-start value (the
    initial/inlet point) to match ``_aligned_grid(include_start=True)``."""
    ncp = np.asarray(dae._element_points()).shape[1] - 1
    a = np.asarray(arr)
    coll = a[:, 1:].ravel() if a.shape[1] == ncp + 1 else a.ravel()
    return np.concatenate([a[:1, 0].ravel(), coll]) if include_start else coll


@dataclass
class _FeasibilityResult:
    """Minimal solve result (status + variable-value dict)."""

    status: str
    x: dict


def _build_x0(m: dm.Model, fills: dict) -> np.ndarray:
    """Assemble an initial point, filling named variables and using clipped
    bound-midpoints (NOT discopt's universal clip-to-10) for the rest.

    Needed for absolute-scale states like temperature (~500 K), which discopt's
    default ``_safe_x0`` would otherwise initialize near 10.
    """
    parts = []
    for v in m._variables:
        if v.name in fills:
            parts.append(np.full(v.size, float(fills[v.name])))
        else:
            lb = np.asarray(v.lb, float).reshape(-1)
            ub = np.asarray(v.ub, float).reshape(-1)
            mid = 0.5 * (np.clip(lb, -1e3, 1e3) + np.clip(ub, -1e3, 1e3))
            parts.append(np.clip(mid, lb, ub))
    return np.concatenate([np.atleast_1d(p).astype(float) for p in parts])


def _solve_feasibility(
    m: dm.Model, nlp_solver: str = "pounce", solver_options: dict | None = None, fills: dict | None = None
) -> _FeasibilityResult:
    """Solve a continuous feasibility model directly through the NLP backend.

    Bypasses ``Model.solve`` and ``differentiable_solve``: the latter mis-handles
    mixed-shape DAE constraints when building its (here-unused) envelope-theorem
    sensitivity (discopt#324). We only need the primal solution, so we call the
    evaluator/NLP dispatch directly.
    """
    from discopt._jax.differentiable import _dispatch_nlp_solve, _safe_x0
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.solvers import SolveStatus

    m.validate()
    evaluator = NLPEvaluator(m)
    x0 = _build_x0(m, fills) if fills else _safe_x0(evaluator)
    opts = dict(solver_options or {})
    opts.setdefault("print_level", 0)
    nlp_result = _dispatch_nlp_solve(nlp_solver, evaluator, x0, opts)

    x_star = np.asarray(nlp_result.x)
    x_dict, offset = {}, 0
    for v in m._variables:
        size = v.size
        val = x_star[offset : offset + size]
        x_dict[v.name] = val.reshape(v.shape) if v.shape != () else val
        offset += size
    status = "optimal" if nlp_result.status == SolveStatus.OPTIMAL else nlp_result.status.value
    return _FeasibilityResult(status, x_dict)


@dataclass
class TransientSolution:
    mkm: MicrokineticModel
    dm_model: dm.Model
    cs: ContinuousSet
    dae: DAEBuilder
    result: object
    reactor: Reactor
    _name_a: dict
    _name_g: dict

    @property
    def status(self) -> str:
        return self.result.status

    def times(self):
        """1D time grid (including ``t=0``), aligned point-for-point with
        :meth:`coverage` and :meth:`concentration`."""
        return _aligned_grid(self.dae, include_start=True)

    def coverage(self, adsorbate):
        """Coverage trajectory as a 1D array aligned with :meth:`times`,
        starting from the initial coverage at ``t=0``."""
        var = self.dae.get_state(self._name_a[adsorbate])
        return _aligned_state(self.dae, self.result.x[var.name], include_start=True)

    def final_coverage(self, adsorbate) -> float:
        return float(self.coverage(adsorbate).reshape(-1)[-1])

    def concentration(self, gas):
        """Concentration trajectory as a 1D array aligned with :meth:`times`,
        starting from the initial concentration at ``t=0``."""
        if not self.reactor.dynamic_gas:
            raise ValueError("gas is held constant in this reactor")
        var = self.dae.get_state(self._name_g[gas])
        return _aligned_state(self.dae, self.result.x[var.name], include_start=True)

    def final_concentration(self, gas) -> float:
        return float(self.concentration(gas).reshape(-1)[-1])


def solve_transient(
    mkm: MicrokineticModel,
    reactor: Reactor,
    t_span: tuple[float, float],
    theta0: dict | None = None,
    nfe: int = 40,
    ncp: int = 3,
    scheme: str = "radau",
    nlp_solver: str = "pounce",
    solver_options: dict | None = None,
) -> TransientSolution:
    """Integrate the microkinetic model in time over ``t_span``."""
    if any(r.equilibrated for r in mkm.reactions):
        raise ValueError("equilibrated (quasi-equilibrium) steps are only supported in steady-state solves")
    m = dm.Model(f"{mkm.name}_transient")
    T_param = mkm.wire_parameters(m)
    cs = ContinuousSet("t", bounds=t_span, nfe=nfe, ncp=ncp, scheme=scheme)
    dae = DAEBuilder(m, cs)

    # fixed-gas reactors expose gas concentrations as constant parameters
    fixed_conc = {} if reactor.dynamic_gas else reactor.create_gas(m, mkm)

    theta0 = theta0 or {}
    for a in mkm.adsorbates:
        dae.add_state(f"theta_{_safe(a.name)}", bounds=(0.0, 1.0), initial=float(theta0.get(a, 0.0)))
    if reactor.dynamic_gas:
        for g in mkm.gas_species:
            dae.add_state(
                f"C_{_safe(g.name)}", bounds=(0.0, 1e6), initial=reactor.initial_concentration(g)
            )
    for s in mkm.sites:
        dae.add_algebraic(f"thetafree_{_safe(s.name)}", bounds=(0.0, 1.0))

    name_a = {a: f"theta_{_safe(a.name)}" for a in mkm.adsorbates}
    name_g = {g: f"C_{_safe(g.name)}" for g in mkm.gas_species}
    name_s = {s: f"thetafree_{_safe(s.name)}" for s in mkm.sites}

    def build_dicts(s, a):
        theta = {ad: s[name_a[ad]] for ad in mkm.adsorbates}
        free_cov = {st: a[name_s[st]] for st in mkm.sites}
        conc = {g: s[name_g[g]] for g in mkm.gas_species} if reactor.dynamic_gas else fixed_conc
        return conc, theta, free_cov

    def ode(t, s, a, c):
        conc, theta, free_cov = build_dicts(s, a)
        d = {}
        for ad in mkm.adsorbates:
            d[name_a[ad]] = net_rate(ad, mkm.reactions, conc, theta, free_cov, T_param, mkm.R, mkm.Tref)
        if reactor.dynamic_gas:
            for g in mkm.gas_species:
                d[name_g[g]] = reactor.gas_rhs(g, conc, theta, free_cov, T_param, mkm)
        return d

    def alg(t, s, a, c):
        conc, theta, free_cov = build_dicts(s, a)
        return {name_s[st]: site_balance_residual(mkm, st, theta, free_cov) for st in mkm.sites}

    dae.set_ode(ode)
    dae.set_algebraic(alg)
    dae.discretize()
    m.minimize(0.0)

    result = _solve_feasibility(m, nlp_solver=nlp_solver, solver_options=solver_options)
    return TransientSolution(mkm, m, cs, dae, result, reactor, name_a, name_g)
