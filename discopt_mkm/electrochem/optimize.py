"""Catalyst-descriptor optimization (the volcano peak) with discopt.

Given linear scaling relations that tie each faradaic step's free energy to a
single binding-energy descriptor ``d``, the best catalyst is the ``d`` that makes
the limiting potential as positive as possible (the smallest overpotential). That
is a max-min over the per-step free energies, which discopt solves directly as a
small linear program: maximize an auxiliary ``U_L`` subject to
``U_L <= -ΔG_i(d)/(n_i F)`` for every step.
"""

from __future__ import annotations

import discopt.modeling as dm


def optimize_descriptor(scaling, n_electrons, bounds, F: float = 1.0, U_eq: float | None = None):
    """Find the descriptor that maximizes the limiting potential.

    Parameters
    ----------
    scaling : list of (a_i, b_i)
        Per faradaic step, the linear scaling relation ``ΔG_i(0) = a_i + b_i * d``
        (free energy at ``U=0`` as a function of the descriptor ``d``).
    n_electrons : list of float
        Electrons transferred per step (aligned with ``scaling``).
    bounds : (lo, hi)
        Search range for the descriptor.
    F : float
        Faraday constant in the model's energy units (1.0 for eV/V).
    U_eq : float, optional
        Equilibrium potential; if given, the returned dict includes the
        overpotential ``U_eq - U_L``.

    Returns
    -------
    dict with ``descriptor``, ``limiting_potential`` (and ``overpotential``).
    """
    m = dm.Model("descriptor_volcano")
    d = m.continuous("d", lb=float(bounds[0]), ub=float(bounds[1]))
    U_L = m.continuous("U_L", lb=-50.0, ub=50.0)
    for (a, b), n in zip(scaling, n_electrons):
        # step is exergonic when ΔG_i(d) + n F U <= 0, i.e. U_L <= -(a+b d)/(n F)
        m.subject_to(U_L <= -(a + b * d) / (n * F))
    m.minimize(-U_L)                       # maximize the limiting potential
    res = m.solve(nlp_solver="pounce")
    d_opt, ul = float(res.value(d)), float(res.value(U_L))
    out = {"descriptor": d_opt, "limiting_potential": ul, "status": str(res.status)}
    if U_eq is not None:
        out["overpotential"] = float(U_eq - ul)
    return out
