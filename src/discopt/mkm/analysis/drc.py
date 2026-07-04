"""Campbell degree of rate control and thermodynamic rate control via AD.

Both quantities are derivatives of a steady-state *output rate* with respect to
kinetic / thermodynamic parameters. The correct mechanism is implicit
differentiation of the steady-state system (``DiffSolveResultL3``), not
``SolveResult.gradient`` (which returns the objective sensitivity and is
identically zero for the zero-objective feasibility formulation).
"""

from __future__ import annotations

import warnings

from discopt.mkm.analysis.sensitivity import evaluate_expression, param_slice, total_derivative


class SensitivityUnavailable(RuntimeError):
    """Raised when L3 implicit differentiation failed (e.g. saturated coverage)."""


def _resolve_rate_expr(solution, rate_expr, species):
    if rate_expr is not None:
        return rate_expr
    if species is not None:
        return solution.production_rate_expr(species)
    raise ValueError("provide either rate_expr or species to analyze")


def degree_of_rate_control(solution, rate_expr=None, species=None) -> dict:
    """Campbell degree of rate control ``X_RC,i`` for each reaction step.

    ``X_RC,i = (k_i / r) (dr/dk_i)`` holding all other rate constants and every
    equilibrium constant fixed. Perturbing the forward pre-exponential ``A_i``
    scales ``k_f`` and the derived ``k_r`` together at fixed ``K_eq``, so the
    "hold other k and all K fixed" condition is satisfied by construction, and
    ``d r/d ln A_i = d r/d ln k_i``.

    Parameters
    ----------
    solution : SteadyStateSolution
    rate_expr : discopt expression, optional
        The output rate to analyze. If omitted, ``species``'s net production
        rate is used.
    species : Species, optional
        Convenience alternative to ``rate_expr``.

    Returns
    -------
    dict
        ``{reaction: X_RC}`` for every reaction in the model. Emits a warning if
        the values do not sum to ~1 (Campbell's theorem), which flags an
        ill-conditioned/saturated operating point where the result is unreliable.
    """
    expr = _resolve_rate_expr(solution, rate_expr, species)
    model, result = solution.dm_model, solution.result

    r_star = evaluate_expression(expr, result, model)
    dr_dp = total_derivative(expr, result, model)
    if dr_dp is None:
        raise SensitivityUnavailable(
            "L3 sensitivities unavailable (KKT system ill-conditioned, often a "
            "saturated coverage). DRC is undefined at this operating point."
        )

    out = {}
    for rxn in solution.mkm.reactions:
        if rxn.equilibrated:
            # a quasi-equilibrated step has no rate constant; its kinetic degree
            # of rate control is structurally zero (the QEA asserts it).
            out[rxn] = 0.0
            continue
        # use the handle snapshotted on this solution (stable if the model was
        # re-solved since), not the live one on the shared reaction object
        handle = (solution.rate_constant_params or {}).get(rxn) or rxn.rate_constant_param()
        start, _ = param_slice(handle, model)
        k = rxn.rate_constant_value()
        out[rxn] = (k / r_star) * float(dr_dp[start])

    # Campbell's theorem: the kinetic degrees of rate control of a *reaction rate*
    # sum to 1. A large deviation means the implicit-differentiation sensitivities
    # are ill-conditioned — almost always a saturated coverage evaluated with the
    # default active_tol, where the returned numbers are unreliable (the L3 solver
    # did not raise SensitivityUnavailable but the result is still garbage). Warn
    # rather than return a silently-wrong DRC. Only check the genuine-rate path: a
    # caller-supplied ``rate_expr`` may be any quantity (a selectivity ratio, a
    # faradaic current, ...) whose degrees of control need not sum to 1.
    if rate_expr is None:
        total = sum(out.values())
        if abs(total - 1.0) > 0.1:
            warnings.warn(
                f"degree of rate control sums to {total:.3g}, not ~1 as Campbell's theorem "
                "requires; the result is likely ill-conditioned (often a saturated coverage "
                "with the default active_tol). Retry with a smaller active_tol (below the "
                "smallest coverage) or coordinates='log' with a warm start.",
                stacklevel=2,
            )
    return out


def thermo_rate_control(solution, rate_expr=None, species=None) -> dict:
    """Thermodynamic degree of rate control ``X_TRC,n`` for each species.

    ``X_TRC,n = -(d ln r / d (G_n / (k_B T)))`` (dimensionless), implemented as
    ``X_TRC,n = -(k_B T / r) (dr/d dG_n)`` where ``dG_n`` is the per-species
    free-energy offset parameter that flows into every ``K_eq`` (and hence the
    derived reverse rates). Here ``k_B`` is the gas constant in the model's
    energy units (``mkm.R``).

    Returns
    -------
    dict
        ``{species: X_TRC}`` for every species in the model.
    """
    expr = _resolve_rate_expr(solution, rate_expr, species)
    model, result = solution.dm_model, solution.result
    mkm = solution.mkm

    r_star = evaluate_expression(expr, result, model)
    dr_dp = total_derivative(expr, result, model)
    if dr_dp is None:
        raise SensitivityUnavailable(
            "L3 sensitivities unavailable; thermodynamic rate control is "
            "undefined at this operating point."
        )

    kBT = mkm.R * mkm.T
    out = {}
    for sp in mkm.species:
        # snapshotted free-energy handle (stable across later re-solves)
        handle = (solution.dG_params or {}).get(sp, sp.dG_param)
        if handle is None:
            continue
        start, _ = param_slice(handle, model)
        out[sp] = -(kBT / r_star) * float(dr_dp[start])
    return out
