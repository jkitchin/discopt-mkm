"""Worked electrochemical example: 4-electron oxygen reduction (ORR).

Associative acidic mechanism, with the intermediate binding energies set from a
single descriptor (the OH adsorption free energy ``x = ΔG_OH``) via the standard
linear scaling relations ``ΔG_OOH = x + 3.2`` eV and ``ΔG_O = 2x``. Referencing
O2 as ``2 H2O - 4.92 eV`` (the usual CHE trick to avoid the O2 DFT error) makes
the four step free energies at ``U = 0`` equal to ``[x-1.72, x-3.2, -x, -x]``,
which sum to ``-4.92 eV`` (the 4-electron equilibrium potential 1.23 V).
"""

from __future__ import annotations

import discopt.mkm as mk

# linear scaling of each step's U=0 free energy in the descriptor x = ΔG_OH:
#   ΔG_i(0) = a_i + b_i * x
ORR_SCALING = [(-1.72, 1.0), (-3.20, 1.0), (0.0, -1.0), (0.0, -1.0)]
ORR_N_ELECTRONS = [1, 1, 1, 1]
ORR_U_EQ = 1.23


def orr_4e(descriptor: float = 0.9, T: float = 298.0, A: float = 1e9, Ea: float = 0.40,
           beta: float = 0.5, U: float = 0.80, F: float = 1.0):
    """Build the associative 4-electron ORR model at descriptor ``ΔG_OH = descriptor``.

    Returns ``(model, reactor)`` with O2 / H+ / H2O activities fixed in a
    differential reactor; set ``model.U`` (volts, RHE) and re-solve to trace a
    polarization curve. ``A``/``Ea`` are the (shared) kinetic pre-exponential and
    chemical barrier of the four proton-coupled electron transfers.
    """
    x = float(descriptor)
    m = mk.Model("ORR", T=T, R=8.617e-5, Tref=298.15, U=U, F=F)
    s = m.site("*", density=1.0)
    O2 = m.gas("O2", H=4.92, S=0.0, composition={"O": 2})
    Hp = m.gas("H⁺", H=0.0, S=0.0, composition={"H": 1})
    H2O = m.gas("H2O", H=0.0, S=0.0, composition={"O": 1, "H": 2})
    OOH = m.adsorbate("OOH*", site=s, H=x + 3.2, S=0.0, composition={"O": 2, "H": 1})
    O = m.adsorbate("O*", site=s, H=2.0 * x, S=0.0, composition={"O": 1})
    OH = m.adsorbate("OH*", site=s, H=x, S=0.0, composition={"O": 1, "H": 1})

    k = dict(A=A, Ea=Ea, n_electrons=1, beta=beta)
    m.step(O2 + s + Hp >> OOH,      name="O2 + (H+ + e-) -> OOH*", **k)
    m.step(OOH + Hp >> O + H2O,     name="OOH* + (H+ + e-) -> O* + H2O", **k)
    m.step(O + Hp >> OH,            name="O* + (H+ + e-) -> OH*", **k)
    m.step(OH + Hp >> H2O + s,      name="OH* + (H+ + e-) -> H2O + *", **k)

    reactor = mk.DifferentialReactor({O2: 1.0, Hp: 1.0, H2O: 1.0})
    return m, reactor


def set_orr_descriptor(m, x: float):
    """Reset the ORR intermediate energies to descriptor ``ΔG_OH = x`` (for a
    volcano sweep). Returns the model."""
    x = float(x)
    m._by_name["OOH*"].H = x + 3.2
    m._by_name["O*"].H = 2.0 * x
    m._by_name["OH*"].H = x
    return m
