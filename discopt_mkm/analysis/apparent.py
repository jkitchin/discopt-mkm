"""Apparent reaction orders and apparent activation energy.

Both are logarithmic sensitivities of the steady-state turnover frequency,
propagated analytically through the steady state by the same implicit
differentiation used for the degree of rate control:

    apparent order in gas i:   n_i     = d ln r / d ln P_i  = (P_i / r)(dr/dP_i)
    apparent activation energy: E_a^app = R T^2 d ln r / dT = (R T^2 / r)(dr/dT)

Because the derivatives are total (they include how the coverages respond), the
apparent quantities are the true local power-law / Arrhenius characterization of
the lumped rate, not the bare elementary-step values.
"""

from __future__ import annotations

from discopt.modeling.core import Parameter

from discopt_mkm.analysis.drc import SensitivityUnavailable
from discopt_mkm.analysis.sensitivity import evaluate_expression, param_slice, total_derivative


def apparent_orders(solution, species, gases=None) -> dict:
    """Apparent reaction order ``d ln r / d ln P_i`` for each gas species.

    ``r`` is the net production rate (turnover frequency) of ``species``. Requires
    a differential reactor, where the gas partial pressures are fixed parameters.

    Returns ``{gas_species: order}``.
    """
    mkm, model, result = solution.mkm, solution.dm_model, solution.result
    rate_expr = solution.production_rate_expr(species)
    r = evaluate_expression(rate_expr, result, model)
    dr = total_derivative(rate_expr, result, model)
    if dr is None:
        raise SensitivityUnavailable("L3 sensitivities unavailable; cannot compute apparent orders")

    out = {}
    for g in gases or mkm.gas_species:
        handle = solution.conc.get(g)
        if not isinstance(handle, Parameter):
            raise ValueError("apparent orders require a differential reactor (fixed gas pressures)")
        P = float(handle.value)
        if P <= 0:
            continue  # order undefined at zero pressure
        start = param_slice(handle, model)[0]
        out[g] = (P / r) * float(dr[start])
    return out


def apparent_activation_energy(solution, species) -> float:
    """Apparent activation energy ``R T^2 d ln r / dT`` (same energy units as ``R``).

    Requires an isothermal solve (temperature is a fixed parameter).
    """
    mkm, model, result = solution.mkm, solution.dm_model, solution.result
    T_param = solution.T_param
    if not isinstance(T_param, Parameter):
        raise ValueError("apparent activation energy requires an isothermal solve (T is a parameter)")
    rate_expr = solution.production_rate_expr(species)
    r = evaluate_expression(rate_expr, result, model)
    dr = total_derivative(rate_expr, result, model)
    if dr is None:
        raise SensitivityUnavailable("L3 sensitivities unavailable; cannot compute apparent Ea")
    start = param_slice(T_param, model)[0]
    T = float(T_param.value)
    return mkm.R * T**2 * float(dr[start]) / r
