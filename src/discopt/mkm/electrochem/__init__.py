"""Electrochemistry for discopt.mkm: potential-dependent kinetics and
thermodynamics, current diagnostics, the computational-hydrogen-electrode (CHE)
free-energy picture, and catalyst-descriptor optimization.

The electrode potential ``U`` is a model operating condition (set ``m.U``, sweep
by re-solving), exactly like temperature. Mark a step faradaic with
``m.step(..., n_electrons=1, beta=0.5)``; the reaction free energy then shifts by
``n_electrons * F * U`` (CHE) and the forward barrier by the Butler-Volmer
fraction ``beta``, with the reverse rate following from detailed balance.

Sign conventions (do not mix — the two current diagnostics use opposite signs):

- :func:`current_density` / :func:`current_expr` (steady-state microkinetic
  current) count electrons consumed in the *forward* direction as positive, so a
  **reduction is positive** (ORR gives ``j > 0``) and an oxidation is negative.
- :func:`cyclic_voltammogram` (transient voltammetry of a soluble couple) follows
  the electrochemistry convention that a **cathodic (reduction) current is
  negative**; its returned current is ``-n F A * (reduction flux)``.

So a reduction that reads positive from ``current_density`` reads negative from
``cyclic_voltammogram``; flip the sign before comparing the two.
"""

from discopt.mkm.electrochem.analysis import (
    apparent_transfer_coefficient,
    current_density,
    current_expr,
    degree_of_current_control,
    electrochemical_steps,
    tafel_slope,
)
from discopt.mkm.electrochem.thermo import (
    che_free_energies,
    che_volcano,
    limiting_potential,
)
from discopt.mkm.electrochem.cv import cyclic_voltammogram
from discopt.mkm.electrochem.optimize import optimize_descriptor
from discopt.mkm.electrochem.examples import (
    ORR_N_ELECTRONS,
    ORR_SCALING,
    ORR_U_EQ,
    orr_4e,
    set_orr_descriptor,
)

__all__ = [
    "current_density",
    "current_expr",
    "tafel_slope",
    "apparent_transfer_coefficient",
    "degree_of_current_control",
    "electrochemical_steps",
    "che_free_energies",
    "limiting_potential",
    "che_volcano",
    "optimize_descriptor",
    "cyclic_voltammogram",
    "orr_4e",
    "set_orr_descriptor",
    "ORR_SCALING",
    "ORR_N_ELECTRONS",
    "ORR_U_EQ",
]
