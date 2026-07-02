"""Computational-hydrogen-electrode (CHE) thermodynamics.

The faradaic free energies already carry the ``n_electrons * F * U`` shift (see
:func:`discopt.mkm.thermo.dG_rxn`), so the CHE free-energy diagram, the limiting
potential, and a descriptor volcano all follow from the numeric reaction free
energies at a chosen potential.
"""

from __future__ import annotations

import numpy as np

from discopt.mkm.numeric import reaction_free_energy


def _faradaic(mkm):
    return [r for r in mkm.reactions if getattr(r, "is_electrochemical", False)]


def che_free_energies(mkm, U: float = 0.0, T: float | None = None):
    """CHE free-energy diagram along the faradaic steps at potential ``U``.

    Returns ``(labels, cumulative, per_step)``: the step names, the cumulative
    reaction free energy after each step (a length ``n+1`` reaction-coordinate
    profile starting at 0), and the per-step ``ΔG_i(U)``. The model's potential is
    restored afterward.
    """
    T = mkm.T if T is None else float(T)
    old = mkm.U
    mkm.U = float(U)
    try:
        steps = _faradaic(mkm)
        dGs = [reaction_free_energy(mkm, r, T) for r in steps]
    finally:
        mkm.U = old
    cumulative = np.concatenate([[0.0], np.cumsum(dGs)])
    return [r.name for r in steps], cumulative, np.asarray(dGs)


def limiting_potential(mkm, T: float | None = None) -> float:
    """Limiting potential: the most positive ``U`` at which *every* faradaic step
    is exergonic, ``U_L = min_i(-ΔG_i(0)/(n_i F))``.

    For a reduction (ORR) the per-step free energy rises with ``U``, so the
    potential-determining step is the least favorable one; the overpotential is
    ``U_eq - U_L`` against the reaction's equilibrium potential.
    """
    T = mkm.T if T is None else float(T)
    old = mkm.U
    mkm.U = 0.0
    try:
        steps = _faradaic(mkm)
        thresholds = [-reaction_free_energy(mkm, r, T) / (r.n_electrons * mkm.F) for r in steps]
    finally:
        mkm.U = old
    return float(min(thresholds))


def che_volcano(mkm, set_descriptor, values, T: float | None = None):
    """Trace the thermodynamic volcano: for each descriptor value, set the binding
    energies via ``set_descriptor(mkm, value)`` and record the limiting potential.

    Returns ``(values, limiting_potentials)`` as arrays. ``set_descriptor`` mutates
    the model's species energies (e.g. through scaling relations).
    """
    out = [limiting_potential((set_descriptor(mkm, v) or mkm), T) for v in values]
    return np.asarray(values, float), np.asarray(out, float)
