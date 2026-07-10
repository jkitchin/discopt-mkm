"""Transient (time-series) parameter estimation via simultaneous collocation.

Fits kinetic/thermodynamic constants of a microkinetic model to measured
*transient* responses (e.g. a product formation rate versus time under a
PRBS/pulse-modulated feed). This is the all-at-once counterpart of the
sequential/shooting pattern (``numeric.integrate_coverages`` inside a curve
fit): each experimental run's coverage ODEs are transcribed to algebraic
collocation constraints with :class:`discopt.dae.DAEBuilder`, the fitted
constants become discopt **Variables shared across all runs**, and one weighted
least-squares NLP solves for the constants and every run's coverage
trajectories simultaneously, with exact AD derivatives.

Design
------
- One :class:`~discopt.dae.ContinuousSet`/:class:`~discopt.dae.DAEBuilder`
  block per :class:`TransientRun`, with element boundaries aligned to the
  switching times of the piecewise-constant gas inputs (so the discontinuous
  input never falls inside an element) and refined to at least ``nfe`` elements.
- Fitted constants are promoted to shared Variables exactly as the
  steady-state estimator does (:func:`~discopt.mkm.estimate.wire_fit_constants`;
  log-space by default for pre-exponentials); every other constant is a fixed
  Parameter. Multi-run fits at different temperatures therefore share ``A``
  and ``Ea`` across runs (Arrhenius-consistent estimates).
- The model response is evaluated at the **exact measurement times** by
  interpolating each element's collocation polynomial (the vectorized analog of
  :meth:`DAEBuilder.state_at`), so irregular sampling schedules need no grid
  alignment and introduce no time-misalignment bias. Residuals are
  ``(y_i - scale * response(t_i)) / sigma_i``.
- The coverage trajectories are warm-started from a numeric (implicit-Radau)
  integration at the nominal parameter values (the transient analog of
  ``Observation.theta0``), and the fitted variables start at ``FitParam.init``.
- Covariance and 95% confidence intervals come from the same
  Fisher-information convention as :func:`~discopt.mkm.estimate.fit_kinetics`
  (explicit response Jacobian w.r.t. the fitted variables), reported in
  physical units via the delta method.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

import discopt.modeling as dm
from discopt.dae import ContinuousSet, DAEBuilder
from discopt.dae.polynomials import lagrange_basis
from discopt.estimate import EstimationResult
from discopt.modeling.core import Constant, MatMulExpression

from discopt.mkm import numeric
from discopt.mkm.assemble import site_balance_residual
from discopt.mkm.estimate import (
    FitParam,
    MKMEstimationResult,
    physical_units_result,
    wire_fit_constants,
)
from discopt.mkm.kinetics import net_rate
from discopt.mkm.model import MicrokineticModel, _safe
from discopt.mkm.species import Adsorbate, Site
from discopt.mkm.transient import _aligned_grid, _aligned_state, _solve_feasibility


@dataclass
class TransientRun:
    """One transient experiment: a measured time series at one condition.

    Parameters
    ----------
    response : Species
        What was measured. A :class:`~discopt.mkm.species.GasSpecies` means the
        net production rate (turnover frequency) of that species; an
        :class:`~discopt.mkm.species.Adsorbate` means its coverage; a
        :class:`~discopt.mkm.species.Site` means its free-site coverage.
    t : array
        Measurement times (need not align with input switches or the
        collocation mesh; must lie within ``t_span``).
    y : array
        Measured values at ``t`` (``y = scale * model_response + noise``).
    T : float
        Temperature for this run.
    pressures : dict
        Gas activities ``{gas_species: value}``. A value is either a constant
        float or a piecewise-constant input ``(times, values)`` where
        ``values[i]`` is held on ``[times[i], times[i+1])`` and the last value
        is held to the end of the run (a trailing extra edge, PRBS style, is
        also accepted). Values are right-continuous at switches. Missing gas
        species default to 0.
    sigma : float or array
        Measurement standard deviation, scalar or per-point (default 1.0).
    scale : float
        Calibration factor mapping the model response to the measured
        observable, ``y ~ scale * response`` (default 1.0).
    theta0 : dict, optional
        Initial coverages ``{adsorbate (or its name): coverage}`` at the start
        of the run (default: clean surface). These are imposed as the fixed
        initial condition of the coverage ODEs.
    t_span : (float, float), optional
        Time window of the run. Defaults to ``(0.0, max(t))``.
    label : str, optional
        Unique key for this run (defaults to ``run{index}``).
    U : float, optional
        Electrode potential for this run (electrochemical fits; default 0).
    """

    response: object
    t: object
    y: object
    T: float
    pressures: dict
    sigma: object = 1.0
    scale: float = 1.0
    theta0: dict = field(default_factory=dict)
    t_span: tuple | None = None
    label: str | None = None
    U: float = 0.0


@dataclass(repr=False)  # keep MKMEstimationResult's compact summary repr
class TransientFitResult(MKMEstimationResult):
    """Transient estimation result: parameter estimates plus fitted trajectories.

    In addition to the :class:`MKMEstimationResult` fields (parameters, standard
    errors, and confidence intervals in physical units):

    - ``predictions[label]``: the fitted model response at that run's
      measurement times (``scale`` applied, directly comparable to ``run.y``);
    - ``times[label]``: the run's collocation time grid (including ``t=0``);
    - ``trajectories[label][adsorbate name]``: the fitted coverage trajectory
      on that grid.
    """

    predictions: dict = field(default_factory=dict)
    times: dict = field(default_factory=dict)
    trajectories: dict = field(default_factory=dict)


def _step_input(entry, gas, t0: float, tf: float):
    """Normalize one ``pressures`` entry to ``(interior switch times, value_at)``.

    ``value_at`` is vectorized: it accepts a scalar or array of times and
    returns matching values. The held value is right-continuous at a switch,
    and ``values[0]`` extends back to ``t0`` if the first switch time is later.
    """
    if isinstance(entry, (tuple, list)):
        if len(entry) != 2:
            raise ValueError(
                f"piecewise input for {gas.name!r} must be a (times, values) pair, got {entry!r}")
        times = np.asarray(entry[0], dtype=float)
        values = np.asarray(entry[1], dtype=float)
        if times.ndim != 1 or values.ndim != 1 or len(times) not in (len(values), len(values) + 1):
            raise ValueError(
                f"piecewise input for {gas.name!r}: expected len(times) == len(values) "
                f"(or one trailing edge), got {len(times)} times / {len(values)} values")
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError(f"piecewise input for {gas.name!r}: times must be strictly increasing")
        lead = times[: len(values)]

        def value_at(t, lead=lead, values=values):
            idx = np.clip(np.searchsorted(lead, np.asarray(t, dtype=float), side="right") - 1,
                          0, len(values) - 1)
            return values[idx]

        interior = times[(times > t0) & (times < tf)]
        return interior, value_at

    val = float(entry)
    return np.empty(0), lambda t, val=val: np.full(np.shape(t), val) if np.ndim(t) else val


# Geometric split of the first element after each input switch: a step input
# drives a fast coverage relaxation whose boundary layer would otherwise make
# the first element's collocation polynomial oscillate and pollute nearby
# response interpolations. Fractions of the first element's width.
_BOUNDARY_LAYER = (0.02, 0.14)


def _element_boundaries(t0: float, tf: float, switches, nfe: int) -> np.ndarray:
    """Element boundaries: every input switch time, refined to >= ``nfe`` elements.

    Each inter-switch segment is uniformly subdivided so no element is wider
    than ``(tf - t0) / nfe``; switch times themselves are always boundaries, so
    the piecewise-constant inputs are exactly constant within every element.
    The first element after ``t0`` and after each switch is further split
    geometrically (:data:`_BOUNDARY_LAYER`) to resolve the fast coverage
    relaxation a step input excites. Switch times (nearly) coincident with each
    other or with the endpoints are deduplicated rather than producing sliver
    elements.
    """
    span = tf - t0
    h_max = span / max(int(nfe), 1)
    sw = np.unique(np.asarray(switches, dtype=float))
    sw = sw[(sw > t0 + 1e-9 * span) & (sw < tf - 1e-9 * span)]
    if len(sw) > 1:
        sw = sw[np.concatenate([[True], np.diff(sw) > 1e-9 * span])]
    knots = np.concatenate([[t0], sw, [tf]])
    parts = []
    for a, b in zip(knots[:-1], knots[1:]):
        n = max(1, int(np.ceil((b - a) / h_max - 1e-9)))
        seg = np.linspace(a, b, n + 1)[:-1]
        w = (b - a) / n
        parts.append(np.concatenate([[a], a + w * np.asarray(_BOUNDARY_LAYER), seg[1:]]))
    return np.append(np.concatenate(parts), tf)


def _interp_matrices(dae: DAEBuilder, t_data: np.ndarray) -> list[np.ndarray]:
    """Collocation-polynomial interpolation matrices onto arbitrary times.

    Returns ``ncp + 1`` matrices ``E_k`` of shape ``(n_meas, nfe)`` such that a
    state's value at ``t_data[i]`` is ``sum_k (E_k @ var[:, k])[i]``, the
    vectorized form of :meth:`DAEBuilder.state_at`, exact to the collocation
    order within the containing element.
    """
    tp = dae._element_points()  # (nfe, ncp+1)
    nfe, ncols = tp.shape
    Es = [np.zeros((len(t_data), nfe)) for _ in range(ncols)]
    for i, t in enumerate(t_data):
        e = dae._locate_element(float(t))
        for k in range(ncols):
            Es[k][i, e] = float(lagrange_basis(tp[e], float(t), k))
    return Es


def _state_at_times(dae: DAEBuilder, name: str, Es: list[np.ndarray]):
    """A ``(n_meas, 1)`` expression for state ``name`` at the interpolation times."""
    var = dae.get_state(name)
    expr = None
    for k, E in enumerate(Es):
        term = MatMulExpression(Constant(E), var[:, k : k + 1])
        expr = term if expr is None else expr + term
    return expr


class _RunBlock:
    """Per-run build artifacts needed after the solve (grouped for clarity)."""

    def __init__(self, run, label, dae, cs, name_a, name_s, response_expr,
                 t, y, sigma, knots, value_at):
        self.run = run
        self.label = label
        self.dae = dae
        self.cs = cs
        self.name_a = name_a          # adsorbate -> state name (without cs prefix)
        self.name_s = name_s          # site -> algebraic name (without cs prefix)
        self.response_expr = response_expr
        self.t = t
        self.y = y
        self.sigma = sigma
        self.knots = knots            # segment edges: t0, interior switches, tf
        self.value_at = value_at      # gas -> vectorized step function


def _build_run(m, mkm, run, i: int, nfe: int, ncp: int, scheme: str) -> _RunBlock:
    """Add one run's DAE block + measurement-time response expression to ``m``."""
    label = run.label or f"run{i}"
    t = np.asarray(run.t, dtype=float).reshape(-1)
    y = np.asarray(run.y, dtype=float).reshape(-1)
    if len(t) != len(y) or len(t) == 0:
        raise ValueError(f"run {label!r}: t and y must be equal-length, non-empty arrays")
    t0, tf = run.t_span if run.t_span is not None else (0.0, float(np.max(t)))
    t0, tf = float(t0), float(tf)
    if not tf > t0:
        raise ValueError(f"run {label!r}: t_span must have tf > t0, got ({t0:g}, {tf:g})")
    if np.any(t < t0) or np.any(t > tf):
        raise ValueError(f"run {label!r}: measurement times must lie within t_span ({t0:g}, {tf:g})")
    sigma = np.broadcast_to(np.asarray(run.sigma, dtype=float), t.shape).copy()
    if np.any(sigma <= 0):
        raise ValueError(f"run {label!r}: sigma must be positive")

    T_i = m.parameter(f"T_{_safe(label)}", float(run.T))
    # per-run electrode potential: point the faradaic steps at this run's U
    # before their rate expressions are built below (mirrors the steady-state
    # estimator's per-observation wiring).
    U_i = m.parameter(f"U_{_safe(label)}", float(run.U))
    for rxn in mkm.reactions:
        if rxn.is_electrochemical:
            rxn._U_param = U_i
            rxn._F = mkm.F

    # piecewise-constant inputs; element boundaries aligned to their switches
    value_at, switches = {}, []
    for g in mkm.gas_species:
        sw, fn = _step_input(run.pressures.get(g, 0.0), g, t0, tf)
        value_at[g] = fn
        switches.extend(sw.tolist())
    eb = _element_boundaries(t0, tf, switches, nfe)
    cs = ContinuousSet(f"t_{_safe(label)}", bounds=(t0, tf), nfe=len(eb) - 1, ncp=ncp,
                       scheme=scheme, element_boundaries=eb)
    dae = DAEBuilder(m, cs)

    name_a = {a: f"th_{_safe(a.name)}" for a in mkm.adsorbates}
    name_s = {s: f"fr_{_safe(s.name)}" for s in mkm.sites}
    theta0 = {a: float(run.theta0.get(a, run.theta0.get(a.name, 0.0))) for a in mkm.adsorbates}
    for a in mkm.adsorbates:
        dae.add_state(name_a[a], bounds=(0.0, 1.0), initial=theta0[a])
    for s in mkm.sites:
        dae.add_algebraic(name_s[s], bounds=(0.0, 1.0))

    # gas values per element, shaped (nfe, 1) to broadcast over the (nfe, ncp)
    # collocation arrays inside the element-wise rate builders
    mid = 0.5 * (eb[:-1] + eb[1:])
    conc_elem = {g: Constant(np.asarray(value_at[g](mid), dtype=float).reshape(-1, 1))
                 for g in mkm.gas_species}

    def ode(t_, s_, a_, c_):
        theta = {ad: s_[name_a[ad]] for ad in mkm.adsorbates}
        free = {st: a_[name_s[st]] for st in mkm.sites}
        return {name_a[ad]: net_rate(ad, mkm.reactions, conc_elem, theta, free, T_i, mkm.R, mkm.Tref)
                for ad in mkm.adsorbates}

    def alg(t_, s_, a_, c_):
        theta = {ad: s_[name_a[ad]] for ad in mkm.adsorbates}
        free = {st: a_[name_s[st]] for st in mkm.sites}
        return {name_s[st]: site_balance_residual(mkm, st, theta, free) for st in mkm.sites}

    dae.set_ode(ode)
    dae.set_algebraic(alg)
    dae.discretize()

    # response at the exact measurement times (collocation-polynomial interpolation)
    Es = _interp_matrices(dae, t)
    theta_meas = {a: _state_at_times(dae, name_a[a], Es) for a in mkm.adsorbates}
    # free-site coverage from the site balance (algebraics exist only at
    # collocation points, so they are reconstructed rather than interpolated)
    free_meas = {}
    for s in mkm.sites:
        occ = [theta_meas[a] for a in mkm.adsorbates_on(s)]
        free_meas[s] = 1.0 - dm.sum(occ) if occ else 1.0
    if isinstance(run.response, Adsorbate):
        response = theta_meas[run.response]
    elif isinstance(run.response, Site):
        response = free_meas[run.response]
    else:
        if all(run.response not in rxn.net_stoich() for rxn in mkm.reactions):
            raise ValueError(
                f"run {label!r}: response species {run.response.name!r} is not net-produced or "
                "-consumed by any reaction, so its rate is identically zero")
        conc_meas = {g: Constant(np.asarray(value_at[g](t), dtype=float).reshape(-1, 1))
                     for g in mkm.gas_species}
        response = net_rate(run.response, mkm.reactions, conc_meas, theta_meas, free_meas,
                            T_i, mkm.R, mkm.Tref)

    sw = np.unique(np.asarray(switches, dtype=float))
    knots = np.concatenate([[t0], sw[(sw > t0) & (sw < tf)], [tf]])
    return _RunBlock(run, label, dae, cs, name_a, name_s, response, t, y, sigma, knots, value_at)


def _warm_start_fills(mkm, blocks: list[_RunBlock], fit: list[FitParam], warm_start: bool) -> dict:
    """Initial point: fitted constants at their nominal values, coverage
    trajectories from a numeric (implicit-Radau) integration at those values.

    Falls back to a flat ``theta0`` trajectory (with a warning) if the numeric
    integration fails; the fit then starts from the flat profile.
    """
    fills: dict = {}
    for fp in fit:
        v0 = min(max(fp.current_value(), fp.lb), fp.ub)
        if fp.is_log():
            fills[fp.resolved_name()] = float(np.log(v0)) if v0 > 0 else float(np.log(fp.lb))
        else:
            fills[fp.resolved_name()] = float(v0)

    for blk in blocks:
        tp = blk.dae._element_points()  # (nfe, ncp+1) node times
        theta0 = {a: float(blk.run.theta0.get(a, blk.run.theta0.get(a.name, 0.0)))
                  for a in mkm.adsorbates}
        traj = None
        if warm_start:
            traj = _integrate_trajectory(mkm, blk, theta0, tp)
        for a in mkm.adsorbates:
            vals = traj[a] if traj is not None else np.full(tp.shape, theta0[a])
            fills[f"{blk.cs.name}_{blk.name_a[a]}"] = vals
        for s in mkm.sites:
            occ = sum(fills[f"{blk.cs.name}_{blk.name_a[a]}"][:, 1:] for a in mkm.adsorbates_on(s))
            occ = occ if isinstance(occ, np.ndarray) else np.zeros(tp[:, 1:].shape)
            fills[f"{blk.cs.name}_{blk.name_s[s]}"] = np.clip(1.0 - occ, 0.0, 1.0)
    return fills


def _integrate_trajectory(mkm, blk: _RunBlock, theta0: dict, tp: np.ndarray):
    """Integrate the coverages segment-by-segment at the current constants and
    sample onto the collocation node times ``tp``; ``None`` on failure."""
    nodes = np.unique(tp.ravel())
    theta = dict(theta0)
    ts, ys = [], []
    try:
        for a, b in zip(blk.knots[:-1], blk.knots[1:]):
            t_eval = np.union1d(nodes[(nodes >= a) & (nodes <= b)], [a, b])
            conc = {g: float(blk.value_at[g](0.5 * (a + b))) for g in mkm.gas_species}
            sol = numeric.integrate_coverages(mkm, conc, blk.run.T, t_eval, theta0=theta)
            ts.append(np.asarray(sol.t, dtype=float))
            ys.append(np.asarray(sol.y, dtype=float))
            theta = {ad: float(sol.y[k, -1]) for k, ad in enumerate(mkm.adsorbates)}
    except Exception as exc:  # numeric warm start is best-effort
        warnings.warn(
            f"numeric warm-start integration failed for run {blk.label!r} ({exc}); "
            "starting the fit from a flat theta0 trajectory instead.",
            stacklevel=2,
        )
        return None
    t_all = np.concatenate(ts)
    y_all = np.concatenate(ys, axis=1)
    order = np.argsort(t_all)
    return {a: np.interp(tp, t_all[order], y_all[k][order]) for k, a in enumerate(mkm.adsorbates)}


def fit_kinetics_transient(
    mkm: MicrokineticModel,
    runs: list[TransientRun],
    fit: list[FitParam],
    nfe: int = 40,
    ncp: int = 3,
    scheme: str = "radau",
    warm_start: bool = True,
    nlp_solver: str = "pounce",
    solver_options: dict | None = None,
) -> TransientFitResult:
    """Estimate kinetic/thermodynamic constants from transient time-series data.

    Builds one collocation block per run (element boundaries aligned to the
    piecewise-constant input switches), shares the fitted constants across all
    runs, and solves a single weighted least-squares NLP for the constants and
    every run's coverage trajectories simultaneously. See the module docstring
    for the formulation.

    Parameters
    ----------
    mkm : MicrokineticModel
        The mechanism. Constants not in ``fit`` are held at their declared
        values. Gas is prescribed by each run's ``pressures`` (the gas phase is
        not a dynamic state), the transient analog of a differential reactor.
    runs : list of TransientRun
        Measured time series; multiple runs (e.g. at different temperatures)
        are fit jointly with shared constants.
    fit : list of FitParam
        Which constants to estimate, with bounds. ``init`` (defaulting to the
        target's current value) is used as the starting point of the fitted
        variable *and* of the warm-start integration.
    nfe : int
        Minimum number of finite elements per run; every input switch time is
        always an element boundary and segments are refined so no element is
        wider than ``t_span/nfe`` (default 40).
    ncp : int
        Collocation points per element (default 3).
    scheme : str
        ``"radau"`` (default) or ``"legendre"``.
    warm_start : bool
        Warm-start each run's coverage trajectory from a numeric implicit-Radau
        integration at the nominal constants (default True).
    nlp_solver, solver_options
        Passed to the direct NLP solve (default ``"pounce"``).

    Returns
    -------
    TransientFitResult
        Estimates, standard errors and 95% confidence intervals in physical
        units, plus per-run model predictions and fitted coverage trajectories.
    """
    if not runs or not fit:
        raise ValueError("fit_kinetics_transient needs at least one run and one fit parameter")
    if any(r.equilibrated for r in mkm.reactions):
        raise NotImplementedError("fitting models with equilibrated steps is not supported")
    labels = [r.label or f"run{i}" for i, r in enumerate(runs)]
    if len(set(labels)) != len(labels):
        raise ValueError(f"run labels must be unique, got {labels}")

    m = dm.Model(f"{mkm.name}_transient_fit")
    unknown, _log_names = wire_fit_constants(m, mkm, fit)

    blocks = [_build_run(m, mkm, run, i, nfe, ncp, scheme) for i, run in enumerate(runs)]

    # weighted least-squares objective over every measurement of every run
    obj_terms = []
    for blk in blocks:
        resid = (Constant(blk.y.reshape(-1, 1)) - blk.run.scale * blk.response_expr) \
            * Constant((1.0 / blk.sigma).reshape(-1, 1))
        obj_terms.append(dm.sum(resid ** 2))
    m.minimize(dm.sum(obj_terms))

    # warm start: nominal constants + numerically integrated trajectories. The
    # integration must see the nominal values of the *fitted* constants, so set
    # them temporarily on the shared Reaction/Species objects.
    saved = [(fp.target, fp.attr, getattr(fp.target, fp.attr)) for fp in fit]
    try:
        for fp in fit:
            setattr(fp.target, fp.attr, fp.current_value())
        fills = _warm_start_fills(mkm, blocks, fit, warm_start)
    finally:
        for target, attr, value in saved:
            setattr(target, attr, value)

    result = _solve_feasibility(m, nlp_solver=nlp_solver, solver_options=solver_options, fills=fills)
    if result.status not in ("optimal", "feasible"):
        raise ValueError(f"transient estimation solve failed with status: {result.status}")

    raw = _covariance_and_raw_result(m, blocks, unknown, result)
    out = physical_units_result(raw, fit)

    predictions, times, trajectories = {}, {}, {}
    y_pred = np.asarray(raw.solve_result_predictions)
    off = 0
    for blk in blocks:
        n = len(blk.t)
        predictions[blk.label] = y_pred[off : off + n].copy()
        off += n
        times[blk.label] = _aligned_grid(blk.dae, include_start=True)
        trajectories[blk.label] = {
            a.name: _aligned_state(blk.dae, result.x[f"{blk.cs.name}_{blk.name_a[a]}"],
                                   include_start=True)
            for a in mkm.adsorbates
        }
    return TransientFitResult(
        parameters=out.parameters,
        std_errors=out.std_errors,
        confidence_intervals=out.confidence_intervals,
        objective=out.objective,
        n_observations=out.n_observations,
        raw=raw,
        predictions=predictions,
        times=times,
        trajectories=trajectories,
    )


def _covariance_and_raw_result(m, blocks: list[_RunBlock], unknown: dict, result) -> EstimationResult:
    """FIM-based covariance at the solution, packaged as a raw ``EstimationResult``.

    Mirrors ``discopt.estimate._compute_estimation_fim``: the response Jacobian
    is the explicit partial derivative w.r.t. the fitted variables (holding the
    trajectory variables at the solution), weighted by the per-point measurement
    error: ``FIM = J^T diag(1/sigma^2) J``. Computed with forward-mode JAX over
    the (few) fitted-variable columns of the compiled response expressions.
    """
    import jax
    import jax.numpy as jnp

    from discopt._jax.differentiable import _compile_parametric_node

    x_parts = [np.asarray(result.x[v.name], dtype=float).reshape(-1) for v in m._variables]
    x_flat = jnp.asarray(np.concatenate(x_parts))
    p_parts = [np.asarray(p.value, dtype=float).ravel() for p in m._parameters]
    p_flat = jnp.asarray(np.concatenate(p_parts)) if p_parts else jnp.zeros(0)

    # flat-vector indices of the fitted variables, in parameter_names order
    param_indices = []
    for name, var in unknown.items():
        off = 0
        for v in m._variables:
            if v is var:
                param_indices.extend(range(off, off + v.size))
                break
            off += v.size
    idx = jnp.asarray(param_indices)

    response_fns = [_compile_parametric_node(blk.response_expr, m) for blk in blocks]
    scales = [float(blk.run.scale) for blk in blocks]

    def stacked_responses(u):
        xf = x_flat.at[idx].set(u)
        return jnp.concatenate([s * jnp.ravel(fn(xf, p_flat))
                                for s, fn in zip(scales, response_fns)])

    u_star = x_flat[idx]
    y_pred = np.asarray(stacked_responses(u_star))            # scale applied
    J = np.asarray(jax.jacfwd(stacked_responses)(u_star)).reshape(len(y_pred), len(param_indices))

    y_obs = np.concatenate([blk.y for blk in blocks])
    sigma = np.concatenate([blk.sigma for blk in blocks])
    weights = 1.0 / sigma**2
    fim = J.T @ (J * weights[:, None])
    if np.linalg.matrix_rank(fim) < len(param_indices):
        # The FIM convention uses the *explicit* response Jacobian (holding the
        # trajectory variables fixed). A response that does not directly contain
        # a fitted constant (e.g. a measured coverage, which is purely state
        # variables) contributes zero explicit sensitivity, so the covariance
        # degenerates. The point estimates are unaffected.
        warnings.warn(
            "Fisher information matrix is singular: at least one fitted constant has no "
            "explicit sensitivity in the measured responses (typical for coverage-only "
            "data). Reported standard errors / confidence intervals are not reliable; "
            "the point estimates are unaffected.",
            stacklevel=3,
        )
    try:
        covariance = np.linalg.inv(fim)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(fim)

    params = {name: float(np.asarray(result.x[name]).flat[0]) for name in unknown}
    objective = float(np.sum(weights * (y_obs - y_pred) ** 2))
    raw = EstimationResult(
        parameters=params,
        covariance=covariance,
        fim=fim,
        objective=objective,
        solve_result=result,
        parameter_names=list(unknown),
        n_observations=len(y_obs),
    )
    raw.solve_result_predictions = y_pred  # stacked model predictions (scale applied)
    return raw
