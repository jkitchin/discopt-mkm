"""Smarter mechanism selection: reduce or pick a sub-mechanism from a candidate
superset, instead of enumerating combinatorially.

Three complementary methods, all built on the existing assembly and analysis
machinery (none reimplements numerics):

- :func:`reduce_by_drc` -- screen a candidate mechanism by flux, then rank the
  surviving steps by the exact Campbell degree of rate control. Removing a step
  that carries negligible flux provably cannot change the steady state, so the
  flux screen is safe; the degree of rate control then explains which retained
  steps actually control the rate.
- :func:`select_subgraph` -- pose mechanism selection as a sparse MINLP: a binary
  in/out variable per step, big-M-gated rates, the steady-state balances as
  constraints, and an objective that trades data misfit against the number of
  steps. Solved by discopt's mixed-integer solver. :func:`pareto_subgraph` sweeps
  the parsimony weight to trace accuracy versus mechanism size.
- :func:`fit_rate_law` -- regress a compact closed-form rate law from a small
  library of mechanistic templates, scored by a parsimony-penalized criterion.

The headline is that relevance is decided by exact sensitivity and global
optimization, not by enumerate-then-threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from discopt_mkm.analysis import degree_of_rate_control
from discopt_mkm.analysis.drc import SensitivityUnavailable
from discopt_mkm.numeric import (
    integrate_coverages,
    net_rate,
    rate_constants,
    rates_of_progress,
    steady_state_numeric,
)
from discopt_mkm.reactors import DifferentialReactor
from discopt_mkm.spec import from_spec, to_spec
from discopt_mkm.steady_state import solve_steady_state


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _name(x):
    return x if isinstance(x, str) else x.name


def _pressures_by_name(condition) -> dict:
    return {_name(k): float(v) for k, v in condition.items()}


def _reactor(model, pres_by_name) -> DifferentialReactor:
    """A differential reactor for ``model`` from a name->pressure dict (0 default)."""
    return DifferentialReactor({g: pres_by_name.get(g.name, 0.0) for g in model.gas_species})


def _physical(theta, free):
    vals = list(theta.values()) + list(free.values())
    return all(-1e-6 <= float(v) <= 1.0 + 1e-6 for v in vals)


def _numeric_state(model, pres_by_name, T, theta0=None):
    """Steady-state coverages, robust *and* fast: try ``fsolve`` from a few seeds
    and accept the first physical root; fall back to integrating to steady state
    only if every seed lands on a non-physical root."""
    conc = {g: pres_by_name.get(g.name, 0.0) for g in model.gas_species}
    n = max(len(model.adsorbates), 1)
    seeds = ([theta0] if theta0 else []) + [
        {a: 0.05 for a in model.adsorbates},
        {a: 0.5 / n for a in model.adsorbates},
        {a: 0.9 / n for a in model.adsorbates},
    ]
    for s in seeds:
        try:
            theta, free = steady_state_numeric(model, conc, T, theta0=s)
        except Exception:
            continue
        if _physical(theta, free):
            return theta, free
    # last resort: integrate to steady state (stays physical) then polish
    t_eval = np.array([0.0, 1e-4, 1e-2, 1.0, 100.0])
    sol = integrate_coverages(model, conc, T, t_eval, theta0={a: 0.05 for a in model.adsorbates})
    warm = {a: float(sol.y[i, -1]) for i, a in enumerate(model.adsorbates)}
    return steady_state_numeric(model, conc, T, theta0=warm)


def _flux_and_tof(model, theta, free, pres_by_name, T, target_name):
    conc = {g: pres_by_name.get(g.name, 0.0) for g in model.gas_species}
    kf, kr = rate_constants(model, T, theta)
    rop = rates_of_progress(model, kf, kr, theta, free, conc)
    target = model._by_name[target_name]
    return rop, net_rate(model, target, rop)


def _submodel(model, keep_reactions, reactor=None):
    """Rebuild a model with only ``keep_reactions`` and the species they touch
    (all gas species and sites are retained; only dangling adsorbates drop)."""
    spec = to_spec(model, reactor)
    keep_names = {r.name for r in keep_reactions}
    spec["reactions"] = [rs for rs in spec["reactions"] if rs["name"] in keep_names]
    used = set()
    for r in keep_reactions:
        for sp in list(r.reactants) + list(r.products):
            used.add(sp.name)
    spec["adsorbates"] = [a for a in spec["adsorbates"] if a["name"] in used]
    spec["infer_composition"] = False
    return from_spec(spec)


def _drc_table(model, pres_by_name_list, target_name, T):
    """Aggregate max |X_RC| per reaction over conditions, on a (conditioned) model.

    Warm-starts the differentiable solve from the robust numeric root and falls
    back to log-coordinates if the linear KKT is ill-conditioned.
    """
    agg = {r: 0.0 for r in model.reactions}
    for pres in pres_by_name_list:
        theta, _ = _numeric_state(model, pres, T)
        reactor = _reactor(model, pres)
        X = None
        for kw in (dict(theta0=theta, active_tol=1e-12),
                   dict(coordinates="log", theta0=theta, log_box=4.0,
                        reg_weight=0.1, active_tol=1e-12)):
            try:
                sol = solve_steady_state(model, reactor, **kw)
                X = degree_of_rate_control(sol, species=model._by_name[target_name])
                break
            except (SensitivityUnavailable, RuntimeError):
                continue
        if X is None:
            continue
        for r, v in X.items():
            agg[r] = max(agg[r], abs(float(v)))
    return agg


@dataclass
class ReductionResult:
    """Outcome of :func:`reduce_by_drc`."""

    reduced: object                       # MicrokineticModel (minimal mechanism)
    reactor: object                       # a DifferentialReactor for the reduced model
    kept: list                            # reaction names retained
    dropped: list                         # reaction names removed
    flux: dict                            # {reaction name: normalized max|flux|}
    drc: dict                             # {reaction name: max|X_RC|} on the reduced model
    max_tof_error: float                  # worst relative TOF error vs the full model

    def __repr__(self):
        return (f"ReductionResult(kept={len(self.kept)}, dropped={len(self.dropped)}, "
                f"max_tof_error={self.max_tof_error:.2e})")


def reduce_by_drc(model, conditions, target, *, flux_tol=1e-6, tof_tol=0.05, T=None,
                  theta0=None) -> ReductionResult:
    """Reduce a candidate mechanism by flux screening, ranked by degree of rate control.

    Parameters
    ----------
    model : MicrokineticModel
        The (over-complete) candidate mechanism.
    conditions : list of dict
        Operating points as ``{gas: partial_pressure}`` dicts (gas as a species
        or a name). The reduction must hold across all of them.
    target : Species or str
        The product whose turnover rate must be preserved.
    flux_tol : float
        Drop steps whose largest normalized flux over the conditions is below
        this (relative to the most active step). Removing a negligible-flux step
        provably cannot change the steady state.
    tof_tol : float
        Sanity bound: raise if the reduced model's turnover rate differs from the
        full model's by more than this (relative) at any condition.

    Returns
    -------
    ReductionResult
    """
    T = model.T if T is None else float(T)
    target_name = _name(target)
    conds = [_pressures_by_name(c) for c in conditions]

    full_tof, agg_flux = [], {r: 0.0 for r in model.reactions}
    for pres in conds:
        theta, free = _numeric_state(model, pres, T, theta0)
        rop, tof = _flux_and_tof(model, theta, free, pres, T, target_name)
        full_tof.append(tof)
        for r, v in rop.items():
            agg_flux[r] = max(agg_flux[r], abs(float(v)))

    fmax = max(agg_flux.values()) or 1.0
    flux_norm = {r: agg_flux[r] / fmax for r in model.reactions}
    keep = [r for r in model.reactions if flux_norm[r] >= flux_tol]
    dropped = [r for r in model.reactions if r not in keep]

    reduced, reactor = _submodel(model, keep, _reactor(model, conds[0]))

    # verify the reduction preserves the turnover rate
    max_err = 0.0
    for pres, tof0 in zip(conds, full_tof):
        theta, free = _numeric_state(reduced, pres, T)
        _, tof = _flux_and_tof(reduced, theta, free, pres, T, target_name)
        if abs(tof0) > 1e-30:
            max_err = max(max_err, abs(tof - tof0) / abs(tof0))
    if max_err > tof_tol:
        raise ValueError(
            f"flux screen at flux_tol={flux_tol:g} changed the turnover rate by "
            f"{max_err:.2%} (> {tof_tol:.0%}); lower flux_tol to keep more steps")

    drc = _drc_table(reduced, conds, target_name, T)

    return ReductionResult(
        reduced=reduced, reactor=reactor,
        kept=[r.name for r in keep], dropped=[r.name for r in dropped],
        flux={r.name: flux_norm[r] for r in model.reactions},
        drc={r.name: v for r, v in drc.items()},
        max_tof_error=max_err,
    )


# --------------------------------------------------------------------------- #
# 2. MINLP best-subgraph selection
# --------------------------------------------------------------------------- #
@dataclass
class SubgraphResult:
    """Outcome of :func:`select_subgraph`."""

    selected: list                        # selected reaction names
    z: dict                               # {reaction name: 0/1}
    fitted_tof: list                      # turnover rate per condition (re-solved numerically)
    data: list                            # observed turnover rate per condition
    misfit: float                         # normalized sum of squared residuals (verified)
    n_steps: int                          # number of selected steps
    lam: float                            # parsimony weight used
    status: str

    def __repr__(self):
        return (f"SubgraphResult(selected={self.selected}, n_steps={self.n_steps}, "
                f"misfit={self.misfit:.2e}, status={self.status!r})")


def _rho_expr(rxn, kf_j, kr_j, theta, free, conc):
    """Mass-action rate of progress as a numeric-coefficient polynomial in the
    coverage variables (rate constants pre-evaluated at the fixed temperature, so
    the MILP relaxation sees only products of variables, not symbolic Arrhenius)."""
    def act(sp):
        if sp in conc:
            return conc[sp]          # gas: fixed partial pressure (float)
        if sp in free:
            return free[sp]          # bare site: free coverage (variable)
        return theta[sp]             # adsorbate coverage (variable)

    def side(stoich):
        e = 1.0
        for sp, c in stoich.items():
            a = act(sp)
            e = e * (a ** int(c) if c != 1 else a)
        return e

    rho = kf_j * side(rxn.reactants)
    if not rxn.irreversible and kr_j != 0.0:
        rho = rho - kr_j * side(rxn.products)
    return rho


def _build_select_model(model, conds, target_name, data, lam, max_steps, T, sigma):
    import discopt.modeling as dm

    m = dm.Model(f"select_{model.name}")
    rxns = model.reactions
    target = model._by_name[target_name]

    kf, kr = rate_constants(model, T)               # numeric at the fixed T
    pmax = max([max(list(c.values()) + [1.0]) for c in conds])
    # big-M per step: a bound on |rate of progress| (coverages <= 1)
    def order(stoich):
        return sum(c for sp, c in stoich.items())
    M = {r: 5.0 * (float(kf[r]) * pmax ** order(r.reactants)
                   + float(kr[r]) * pmax ** order(r.products)) + 1.0 for r in rxns}

    z = {r: m.boolean(f"z_{j}") for j, r in enumerate(rxns)}

    misfit_terms, fitted = [], []
    for c, (pres, obs) in enumerate(zip(conds, data)):
        conc = {g: pres.get(g.name, 0.0) for g in model.gas_species}
        theta = {a: m.continuous(f"th_{c}_{i}", lb=0.0, ub=1.0)
                 for i, a in enumerate(model.adsorbates)}
        free = {s: m.continuous(f"fr_{c}_{i}", lb=0.0, ub=1.0)
                for i, s in enumerate(model.sites)}
        # gated rate of progress per step: r == rho when z=1, r == 0 when z=0
        r = {}
        for j, rxn in enumerate(rxns):
            rho = _rho_expr(rxn, float(kf[rxn]), float(kr[rxn]), theta, free, conc)
            rv = m.continuous(f"r_{c}_{j}", lb=-M[rxn], ub=M[rxn])
            zb = z[rxn].variable
            m.subject_to(rv <= M[rxn] * zb, name=f"gp_{c}_{j}")
            m.subject_to(rv >= -M[rxn] * zb, name=f"gn_{c}_{j}")
            m.subject_to(rv <= rho + M[rxn] * (1 - zb), name=f"gu_{c}_{j}")
            m.subject_to(rv >= rho - M[rxn] * (1 - zb), name=f"gl_{c}_{j}")
            r[rxn] = rv
        # steady state: net adsorbate production from the gated rates = 0
        for a in model.adsorbates:
            bal = dm.sum([rxn.net_stoich().get(a, 0.0) * r[rxn] for rxn in rxns
                          if rxn.net_stoich().get(a, 0.0) != 0.0])
            m.subject_to(bal == 0.0, name=f"ss_{c}_{a.name}")
        # site balance
        for s in model.sites:
            ads_on = [a for a in model.adsorbates if a.site is s]
            m.subject_to(free[s] + dm.sum([theta[a] for a in ads_on]) == 1.0,
                         name=f"site_{c}_{s.name}")
        # target turnover from the gated rates (bound to a variable so it reads back)
        tofv = m.continuous(f"tof_{c}", lb=-max(M.values()), ub=max(M.values()))
        m.subject_to(tofv == dm.sum([rxn.net_stoich().get(target, 0.0) * r[rxn]
                     for rxn in rxns if rxn.net_stoich().get(target, 0.0) != 0.0]),
                     name=f"tof_{c}")
        fitted.append(tofv)
        # residual as its own variable so the objective is a genuine quadratic (MIQP)
        scale = max(abs(obs), sigma)
        ev = m.continuous(f"e_{c}", lb=-1e6, ub=1e6)
        m.subject_to(ev == (tofv - obs) / scale, name=f"resid_{c}")
        misfit_terms.append(ev * ev)

    if max_steps is not None:
        m.subject_to(dm.sum([z[r].variable for r in rxns]) <= max_steps, name="cardinality")

    m.minimize(dm.sum(misfit_terms) + lam * dm.sum([z[r].variable for r in rxns]))
    return m, z, fitted


def _score_subset(model, keep_reactions, conds, target_name, data, T, sigma):
    """Normalized turnover-rate misfit of a candidate sub-mechanism (numerical
    steady-state solve at each condition). Returns ``(misfit, fitted_tof)``;
    ``inf`` if the subset cannot reproduce the target (no route / singular)."""
    target = model._by_name[target_name]
    if not any(r.net_stoich().get(target, 0.0) > 0 for r in keep_reactions):
        return float("inf"), [0.0] * len(conds)
    sub, _ = _submodel(model, keep_reactions, _reactor(model, conds[0]))
    fitted, sse = [], 0.0
    for pres, obs in zip(conds, data):
        try:
            theta, free = _numeric_state(sub, pres, T)
            _, tof = _flux_and_tof(sub, theta, free, pres, T, target_name)
        except Exception:
            return float("inf"), [float("nan")] * len(conds)
        fitted.append(tof)
        sse += ((tof - obs) / max(abs(obs), sigma)) ** 2
    return sse, fitted


def select_subgraph(model, conditions, target, data, *, lam=0.05, fit_tol=1e-3,
                    T=None, sigma=1e-6, engine="greedy", time_limit=120,
                    solver_options=None) -> SubgraphResult:
    """Select the smallest sub-mechanism that reproduces the turnover data.

    Best-subgraph selection: find the fewest steps whose steady state still
    reproduces the observed turnover rate of ``target`` across all conditions,
    trading data misfit against the number of steps.

    Two engines:

    - ``"greedy"`` (default) / ``"exhaustive"`` -- search over *structure* and
      score each candidate with an exact numerical steady-state solve. Greedy
      backward elimination removes the step whose removal least hurts the fit
      until no further removal stays within ``fit_tol``; exhaustive enumerates all
      subsets (use only for small candidate sets). These are robust and find the
      minimal data-reproducing mechanism.
    - ``"milp"`` -- a true **MILP** that discopt solves directly (sub-second). The
      trick is to work in *flux* space, not coverage space: the steady-state
      balances on the net rates of progress are linear, a binary gates each step,
      and a capacity bound (a flux cannot exceed its rate constant) rejects
      kinetically incapable shortcut routes. Minimizes the number of steps subject
      to reproducing the observed turnover. This is the recommended global-solver
      route (see :func:`_select_milp`).
    - ``"minlp"`` -- pose it as a true MINLP for discopt (a binary in/out variable
      per step gating big-M-bounded rates, the nonlinear steady-state balances as
      constraints, objective = misfit + ``lam`` x step count). **Experimental:**
      embedding the nonlinear steady state in coverage space makes the spatial
      branch-and-bound intractable beyond a handful of steps (this is exactly why
      the ``"milp"`` flux reformulation exists). The returned selection is always
      re-verified by a numerical solve.

    Parameters
    ----------
    model : MicrokineticModel
        The over-complete candidate mechanism.
    conditions : list of dict
        Operating points ``{gas: partial_pressure}`` (isothermal at ``T``).
    target : Species or str
        The product whose turnover rate is fit.
    data : list of float
        Observed turnover rate of ``target`` at each condition.
    lam : float
        Parsimony weight (cost per retained step, normalized-misfit units).
    fit_tol : float
        A step is removable (greedy) if doing so keeps the total misfit below this.

    Returns
    -------
    SubgraphResult
    """
    T = model.T if T is None else float(T)
    target_name = _name(target)
    conds = [_pressures_by_name(c) for c in conditions]
    data = [float(d) for d in data]
    status = engine

    if engine == "milp":
        selected_names, status = _select_milp(model, conds, target_name, data, T,
                                              solver_options)
    elif engine == "minlp":
        selected_names, status = _select_minlp(model, conds, target_name, data,
                                               lam, T, sigma, time_limit, solver_options)
    elif engine == "exhaustive":
        selected_names = _select_exhaustive(model, conds, target_name, data, lam, T, sigma)
    elif engine == "greedy":
        selected_names = _select_greedy(model, conds, target_name, data, fit_tol, T, sigma)
    else:
        raise ValueError(f"unknown engine {engine!r}")

    selected = [r for r in model.reactions if r.name in selected_names]
    misfit, fit_tof = _score_subset(model, selected, conds, target_name, data, T, sigma)
    return SubgraphResult(
        selected=[r.name for r in selected],
        z={r.name: int(r in selected) for r in model.reactions},
        fitted_tof=fit_tof, data=data, misfit=float(misfit),
        n_steps=len(selected), lam=lam, status=str(status),
    )


def _select_greedy(model, conds, target_name, data, fit_tol, T, sigma):
    """Backward elimination: drop the least-damaging step until none can go."""
    keep = list(model.reactions)
    while len(keep) > 1:
        best, best_misfit = None, None
        for r in keep:
            trial = [x for x in keep if x is not r]
            msf, _ = _score_subset(model, trial, conds, target_name, data, T, sigma)
            if msf <= fit_tol and (best_misfit is None or msf < best_misfit):
                best, best_misfit = trial, msf
        if best is None:
            break
        keep = best
    return [r.name for r in keep]


def _select_exhaustive(model, conds, target_name, data, lam, T, sigma):
    """Exact: minimize misfit + lam*count over all subsets (small sets only)."""
    from itertools import combinations
    rxns = list(model.reactions)
    best, best_score = None, float("inf")
    for k in range(1, len(rxns) + 1):
        for combo in combinations(rxns, k):
            msf, _ = _score_subset(model, list(combo), conds, target_name, data, T, sigma)
            score = msf + lam * k
            if score < best_score:
                best, best_score = combo, score
    return [r.name for r in (best or rxns)]


def _select_milp(model, conds, target_name, data, T, solver_options):
    """Flux-space selection as a true MILP.

    Working in net rates of progress (fluxes) instead of coverages makes the
    steady-state balances *linear*, so the problem is a pure MILP that discopt
    solves directly. A continuous flux ``v_j`` per step is gated by a binary
    ``z_j`` and capped by the step's forward rate constant ``k_f,j`` (a flux can
    never exceed its rate constant, since the mass-action activities are <= 1).
    The constraints are: every surface intermediate is conserved
    (``sum_j nu_ij v_j == 0``), the target is produced at the observed turnover
    rate (``sum_j nu_target,j v_j == TOF_obs``), and ``|v_j| <= cap_j z_j``. The
    objective minimizes the number of steps. The capacity bound is what rejects
    kinetically incapable shortcuts (a low-barrier-on-paper decoy whose rate
    constant is tiny cannot carry the required flux), without any nonlinear
    kinetics in the model.
    """
    import discopt.modeling as dm

    rxns = model.reactions
    target = model._by_name[target_name]
    kf, kr = rate_constants(model, T)
    cap = {r: max(float(kf[r]), float(kr[r])) for r in rxns}

    m = dm.Model(f"milp_{model.name}")
    z = {r: m.boolean(f"z_{j}") for j, r in enumerate(rxns)}
    for c, (pres, obs) in enumerate(zip(conds, data)):
        v = {r: m.continuous(f"v_{c}_{j}", lb=-cap[r], ub=cap[r])
             for j, r in enumerate(rxns)}
        for r in rxns:
            zb = z[r].variable
            m.subject_to(v[r] <= cap[r] * zb, name=f"cap_p_{c}_{r.name}")
            m.subject_to(v[r] >= -cap[r] * zb, name=f"cap_n_{c}_{r.name}")
            if r.irreversible:
                m.subject_to(v[r] >= 0.0, name=f"irr_{c}_{r.name}")
        # surface intermediates are conserved at steady state (linear balances)
        for a in model.adsorbates:
            m.subject_to(dm.sum([r.net_stoich().get(a, 0.0) * v[r] for r in rxns
                         if r.net_stoich().get(a, 0.0) != 0.0]) == 0.0,
                         name=f"bal_{c}_{a.name}")
        # the target is produced at the observed turnover rate
        m.subject_to(dm.sum([r.net_stoich().get(target, 0.0) * v[r] for r in rxns
                     if r.net_stoich().get(target, 0.0) != 0.0]) == obs,
                     name=f"tof_{c}")
    m.minimize(dm.sum([z[r].variable for r in rxns]))
    res = m.solve(nlp_solver="pounce", **(solver_options or {}))
    names = [r.name for r in rxns
             if int(round(float(res.value(z[r].variable)))) == 1]
    return names, res.status


def _select_minlp(model, conds, target_name, data, lam, T, sigma, time_limit, solver_options):
    """Experimental discopt MINLP engine (see :func:`select_subgraph`)."""
    opts = {"time_limit": time_limit, **(solver_options or {})}
    m, z, _ = _build_select_model(model, conds, target_name, data, lam, None, T, sigma)
    res = m.solve(nlp_solver="pounce", **opts)
    names = [r.name for r in model.reactions
             if int(round(float(res.value(z[r].variable)))) == 1]
    return names, res.status


def pareto_subgraph(model, conditions, target, data, *, T=None, sigma=1e-6):
    """Accuracy-versus-size Pareto front: best achievable misfit at each number
    of steps. Returns ``{n_steps: (misfit, [reaction names])}``. The knee, the
    smallest mechanism whose misfit is essentially zero, is the one to keep."""
    from itertools import combinations
    T = model.T if T is None else float(T)
    target_name = _name(target)
    conds = [_pressures_by_name(c) for c in conditions]
    data = [float(d) for d in data]
    rxns = list(model.reactions)
    front = {}
    for k in range(1, len(rxns) + 1):
        best, best_misfit = None, float("inf")
        for combo in combinations(rxns, k):
            msf, _ = _score_subset(model, list(combo), conds, target_name, data, T, sigma)
            if msf < best_misfit:
                best, best_misfit = combo, msf
        front[k] = (best_misfit, [r.name for r in best] if best else [])
    return front


# --------------------------------------------------------------------------- #
# 3. symbolic rate-law regression
# --------------------------------------------------------------------------- #
@dataclass
class RateLawResult:
    """Outcome of :func:`fit_rate_law`."""

    template: str                         # winning template name
    expression: object                    # sympy expression for the rate
    params: dict                          # fitted constants
    orders: dict                          # apparent reaction order per gas
    aic: float                            # Akaike information criterion (lower better)
    candidates: list = field(default_factory=list)  # (template, aic) for all tried

    def __repr__(self):
        return f"RateLawResult(template={self.template!r}, aic={self.aic:.1f}, orders={self.orders})"


def _rate_law_templates(gases):
    """A small library of mechanistic rate laws over the listed gas names.

    Each entry is ``(name, n_params, fn(P, *params), sympy_builder)`` where ``P``
    is a dict of gas-name -> pressure array. Two-gas reactant laws (CO + O2 style)
    plus a generic power law."""
    import sympy as sp

    g = list(gases)
    syms = {x: sp.Symbol(f"P_{x}", positive=True) for x in g}
    k = sp.Symbol("k", positive=True)

    def power(P, k, *exps):
        out = k * np.ones_like(next(iter(P.values())))
        for x, a in zip(g, exps):
            out = out * P[x] ** a
        return out

    def power_expr(vals):
        e = sp.Symbol("k", positive=True)
        for x, a in zip(g, vals[1:]):
            e = e * syms[x] ** round(float(a), 2)
        return e

    templates = [("power_law", 1 + len(g), power, power_expr)]

    if len(g) >= 2:
        a, b = g[0], g[1]

        def lh(P, k, K):  # dual-site Langmuir-Hinshelwood, A-inhibited
            return k * P[a] * P[b] / (1 + K * P[a]) ** 2

        def lh_expr(vals):
            kk, KK = vals
            return (sp.Symbol("k", positive=True) * syms[a] * syms[b]
                    / (1 + round(float(KK), 3) * syms[a]) ** 2)

        templates.append(("langmuir_hinshelwood", 2, lh, lh_expr))
    return templates


def fit_rate_law(conditions, data, gases, *, ref=None):
    """Regress a compact closed-form rate law from a small template library.

    Fits each mechanistic template (power law, dual-site Langmuir-Hinshelwood) to
    the turnover data by least squares and selects the one with the lowest Akaike
    information criterion (fit penalized by parameter count). Returns the winning
    ``sympy`` expression, the fitted constants, and the apparent reaction orders
    it implies at a reference condition.

    Parameters
    ----------
    conditions : list of dict
        Operating points ``{gas: partial_pressure}``.
    data : list of float
        Observed turnover rate at each condition.
    gases : list of str
        The gas species (names) the rate may depend on, in order.
    ref : dict, optional
        Reference condition for reporting apparent orders (default: the geometric
        mean of the supplied conditions).
    """
    from scipy.optimize import curve_fit

    gases = [_name(x) for x in gases]
    P = {x: np.array([float(c[_name_in(c, x)]) for c in conditions]) for x in gases}
    y = np.asarray([float(d) for d in data], float)
    n = len(y)

    results = []
    for name, npar, fn, builder in _rate_law_templates(gases):
        try:
            p0 = [max(y.mean(), 1e-6)] + [0.5] * (npar - 1)
            popt, _ = curve_fit(lambda X, *p: fn(P, *p), xdata=np.arange(n), ydata=y,
                                p0=p0, maxfev=20000)
            resid = fn(P, *popt) - y
            sse = float(np.sum(resid ** 2))
            aic = n * np.log(sse / n + 1e-300) + 2 * npar
            results.append((name, aic, popt, builder))
        except Exception:
            continue
    if not results:
        raise RuntimeError("no rate-law template could be fit to the data")

    results.sort(key=lambda t: t[1])
    name, aic, popt, builder = results[0]
    expr = builder(popt)

    # apparent orders d ln r / d ln P_i from the fitted law at the reference point
    if ref is None:
        ref = {x: float(np.exp(np.mean(np.log(P[x])))) for x in gases}
    orders = {}
    fn = dict((nm, f) for nm, _, f, _ in _rate_law_templates(gases)).get  # noqa
    for x in gases:
        h = 1e-3
        rp = _eval_law(name, popt, gases, {**ref, x: ref[x] * (1 + h)})
        rm = _eval_law(name, popt, gases, {**ref, x: ref[x] * (1 - h)})
        orders[x] = round(float((np.log(rp) - np.log(rm)) / (2 * h)), 3)

    params = {"k": float(popt[0])}
    if name == "power_law":
        params.update({f"order_{x}": float(a) for x, a in zip(gases, popt[1:])})
    elif name == "langmuir_hinshelwood":
        params["K"] = float(popt[1])
    return RateLawResult(template=name, expression=expr, params=params, orders=orders,
                         aic=float(aic), candidates=[(nm, float(a)) for nm, a, _, _ in results])


def _name_in(cond, gas):
    for k in cond:
        if _name(k) == gas:
            return k
    raise KeyError(gas)


def _eval_law(name, popt, gases, pres):
    P = {x: np.array([pres[x]]) for x in gases}
    for nm, _, fn, _ in _rate_law_templates(gases):
        if nm == name:
            return float(fn(P, *popt)[0])
    raise KeyError(name)
