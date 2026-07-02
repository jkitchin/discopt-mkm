"""Adapter isolating the (underscore-prefixed) discopt AD internals.

Everything in this module touches private discopt symbols. Keeping it in one
place means a discopt version bump only needs fixing here. Pin the discopt
version this was written against.
"""

from __future__ import annotations

import jax
import numpy as np

from discopt._jax.differentiable import _compile_parametric_node, _get_param_slice


def evaluate_expression(expr, result, model) -> float:
    """Evaluate a discopt expression at the solved point ``(x*, p)``."""
    fn = _compile_parametric_node(expr, model)
    return float(fn(result._x_star, result._p_flat))


def param_slice(param, model) -> tuple[int, int]:
    """``(start, end)`` indices of ``param`` within the flat parameter vector."""
    return _get_param_slice(param, model)


def total_derivative(expr, result, model):
    """Total derivative ``d expr / d p`` including the implicit ``x*(p)`` term.

    Returns a vector aligned with the flat parameter vector, computed as
    ``(d expr/d x) @ (dx*/dp) + d expr/d p`` (the same identity discopt uses in
    ``DiffSolveResultL3.implicit_gradient``). Returns ``None`` if the L3
    sensitivity matrix is unavailable (ill-conditioned KKT system).
    """
    dx_dp = result.sensitivity_matrix()
    if dx_dp is None:
        return None
    dx_dp = np.asarray(dx_dp)  # (n_vars, n_params)

    fn = _compile_parametric_node(expr, model)
    x_star, p_flat = result._x_star, result._p_flat
    dr_dx = np.asarray(jax.grad(fn, argnums=0)(x_star, p_flat))  # (n_vars,)
    dr_dp_direct = np.asarray(jax.grad(fn, argnums=1)(x_star, p_flat))  # (n_params,)
    return dr_dx @ dx_dp + dr_dp_direct
