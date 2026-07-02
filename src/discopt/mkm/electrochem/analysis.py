"""Electrochemical diagnostics: current density and its potential dependence.

These mirror :mod:`discopt.mkm.analysis.apparent` exactly, with the electrode
potential ``U`` playing the role temperature plays for apparent activation energy.
The current is ``j = F * sum_j n_j * r_j`` over the faradaic steps, and its
log-derivative with respect to ``U`` (obtained by automatic differentiation
through the steady state) gives the Tafel slope and the apparent transfer
coefficient.

Sign convention: electrons consumed in the forward direction count positive, so a
reduction (e.g. ORR) carries a positive ``n_electrons`` and a positive faradaic
current. Tafel diagnostics are only meaningful away from the equilibrium
potential (where ``j`` crosses zero); within ``tiny`` of zero current they raise.
"""

from __future__ import annotations

import numpy as np

import discopt.modeling as dm

from discopt.mkm.analysis.drc import SensitivityUnavailable, degree_of_rate_control
from discopt.mkm.analysis.sensitivity import (
    evaluate_expression,
    param_slice,
    total_derivative,
)


def electrochemical_steps(mkm):
    """The faradaic (electron-transfer) steps of a mechanism."""
    return [r for r in mkm.reactions if getattr(r, "is_electrochemical", False)]


def current_expr(solution):
    """discopt expression for the faradaic current ``F * sum n_j r_j``."""
    mkm = solution.mkm
    steps = electrochemical_steps(mkm)
    if not steps:
        raise ValueError("no electrochemical steps (set n_electrons on the faradaic steps)")
    return mkm.F * dm.sum([r.n_electrons * solution.rate_of_progress_expr(r) for r in steps])


def current_density(solution) -> float:
    """Faradaic current at the solved steady state (electrode-area-independent
    per-site current; multiply by site density for a current density)."""
    return evaluate_expression(current_expr(solution), solution.result, solution.dm_model)


def _dlnj_dU(solution, tiny=1e-12):
    """``(d ln|j|/dU, j)`` by AD through the steady state."""
    if solution.U_param is None:
        raise ValueError("model has no potential parameter (no electrochemical steps)")
    model, result = solution.dm_model, solution.result
    expr = current_expr(solution)
    j = evaluate_expression(expr, result, model)
    if abs(j) < tiny:
        raise ValueError(
            f"current is ~0 (j={j:.2e}); the Tafel analysis is undefined near the "
            "equilibrium potential. Evaluate it in the Tafel region (larger |overpotential|).")
    dj = total_derivative(expr, result, model)
    if dj is None:
        raise SensitivityUnavailable(
            "L3 sensitivities unavailable (ill-conditioned KKT, often a saturated "
            "coverage at large overpotential); the potential derivative is undefined here.")
    col = param_slice(solution.U_param, model)[0]
    return float(dj[col]) / j, j


def tafel_slope(solution) -> float:
    """Tafel slope ``dU / d log10|j| = ln(10) / (d ln|j|/dU)`` (volts/decade).

    Negative for a cathodic branch (``|j|`` grows as ``U`` falls), positive for an
    anodic one. For a single one-electron rate-determining step its magnitude is
    ``2.303 R T / (beta F)`` (~118 mV/decade at 298 K, ``beta=0.5``, ``F=1``)."""
    dlnj, _ = _dlnj_dU(solution)
    return float(np.log(10.0) / dlnj)


def apparent_transfer_coefficient(solution) -> float:
    """Apparent transfer coefficient ``-(R T / F) d ln|j|/dU`` (dimensionless).

    Positive on a cathodic branch; equals ``beta`` for a single one-electron
    rate-determining step, and is the electrochemical analog of the apparent
    reaction order / apparent activation energy."""
    dlnj, _ = _dlnj_dU(solution)
    mkm = solution.mkm
    T = float(solution.T_param.value)
    return float(-(mkm.R * T / mkm.F) * dlnj)


def degree_of_current_control(solution) -> dict:
    """Degree of rate control of the *current* ``(k_i/j)(dj/dk_i)`` per step:
    which step limits the current at this potential. Reuses the Campbell DRC on
    the current expression, so it sums to 1."""
    return degree_of_rate_control(solution, rate_expr=current_expr(solution))
