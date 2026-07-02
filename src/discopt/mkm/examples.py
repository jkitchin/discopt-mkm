"""Worked microkinetic examples."""

from __future__ import annotations

import numpy as np

from discopt.mkm import CSTR, DifferentialReactor, Model


def co_oxidation(T: float = 500.0):
    """CO oxidation on a single Pt site type (eV / eV·K⁻¹ units, R = k_B).

    Mechanism (Langmuir-Hinshelwood)::

        CO  + *   <=> CO*
        O2  + 2*  <=> 2 O*
        CO* + O*  <=> CO2 + 2*

    Returns
    -------
    (model, reactor)
        A :class:`~discopt.mkm.model.MicrokineticModel` and a
        :class:`~discopt.mkm.reactors.DifferentialReactor` at fixed gas
        partial pressures, ready for :func:`~discopt.mkm.solve_steady_state`.
    """
    m = Model("co_oxidation", T=T, R=8.617e-5, Tref=298.15)

    s = m.site("Pt", density=1.0)
    CO = m.gas("CO", H=0.0, S=0.0020)
    O2 = m.gas("O2", H=0.0, S=0.0021)
    CO2 = m.gas("CO2", H=-3.0, S=0.0023)
    COs = m.adsorbate("CO*", site=s, H=-0.8, S=0.0005)
    Os = m.adsorbate("O*", site=s, H=-0.3, S=0.0005)

    m.step(CO + s >> COs, A=1e4, Ea=0.0, name="CO adsorption")
    m.step(O2 + 2 * s >> 2 * Os, A=1e4, Ea=0.0, name="O2 dissociation")
    m.step(COs + Os >> CO2 + 2 * s, A=1e8, Ea=0.7, name="surface reaction")

    reactor = DifferentialReactor({CO: 1.0, O2: 0.5, CO2: 0.0})
    m.infer_composition()  # CO, O2, CO2, CO*, O* parse as formulas
    return m, reactor


def water_gas_shift(T: float = 480.0):
    """Water-gas shift ``CO + H2O <=> CO2 + H2`` on a single site (7 steps).

    Constants (forward rate constant ``kf`` and equilibrium constant ``Keq`` per
    step) are transcribed from the Supporting Information of Yang, Achar &
    Kitchin, *Evaluation of the degree of rate control via automatic
    differentiation*, AIChE J. **68**(6):e17653 (2022) — the case where the net
    rate (~1e-6 s^-1) is a 12-order cancellation of one-way step rates
    (~1e5-1e6 s^-1), so finite differences cannot compute the DRC and the
    sensitivity must be propagated analytically.

    Gas activities are SI partial pressures in bar (P = 1.01325). Returns
    ``(model, reactor)`` for a differential reactor; solve in log coordinates.
    """
    P = 1.01325
    kf = [1.33e8, 2.01e11, 2.64e6, 5.24e1, 2.05e5, 1.48e12, 5.32e2]
    Keq = [2.15e2, 5.93e-5, 6.28e-2, 1.18e-5, 1.03e3, 1.92e5, 4.50e1]

    m = Model("water_gas_shift", T=T, R=8.617e-5, Tref=298.15)
    s = m.site("*", density=1.0)
    CO = m.gas("CO")
    H2O = m.gas("H2O")
    CO2 = m.gas("CO2")
    H2 = m.gas("H2")
    COs = m.adsorbate("CO*", site=s)
    H2Os = m.adsorbate("H2O*", site=s)
    OHs = m.adsorbate("OH*", site=s)
    Hs = m.adsorbate("H*", site=s)
    Os = m.adsorbate("O*", site=s)
    CO2s = m.adsorbate("CO2*", site=s)

    # Step numbers follow the paper's mechanism (SI) ordering. The two
    # rate-controlling steps come out as S4 (~0.88) and S5 (~0.12).
    m.step(CO + s >> COs, kf=kf[0], Keq=Keq[0], name="S1: CO + * -> CO*")
    m.step(H2O + s >> H2Os, kf=kf[1], Keq=Keq[1], name="S2: H2O + * -> H2O*")
    m.step(H2Os + s >> OHs + Hs, kf=kf[2], Keq=Keq[2], name="S3: H2O* + * -> OH* + H*")
    m.step(OHs + s >> Os + Hs, kf=kf[3], Keq=Keq[3], name="S4: OH* + * -> O* + H*")
    m.step(COs + Os >> CO2s + s, kf=kf[4], Keq=Keq[4], name="S5: CO* + O* -> CO2* + *")
    m.step(CO2s >> CO2 + s, kf=kf[5], Keq=Keq[5], name="S6: CO2* -> CO2 + *")
    m.step(2 * Hs >> H2 + 2 * s, kf=kf[6], Keq=Keq[6], name="S7: 2 H* -> H2 + 2*")

    reactor = DifferentialReactor({CO: 0.07 * P, H2O: 0.21 * P, CO2: 0.085 * P, H2: 0.38 * P})
    m.infer_composition()
    return m, reactor


def water_gas_shift_qea(T: float = 480.0):
    """Water-gas shift under the quasi-equilibrium approximation.

    Same constants as :func:`water_gas_shift`, but the five fast steps are marked
    ``equilibrated`` and only the two kinetically relevant steps (S4 OH
    dissociation, S5 CO2* formation) keep a rate law. The fast-step cancellation
    is replaced by equilibrium relations, so this solves in plain *linear*
    coordinates with no warm start and reproduces the full-SSA rate to ~0.01%.
    """
    P = 1.01325
    kf = [1.33e8, 2.01e11, 2.64e6, 5.24e1, 2.05e5, 1.48e12, 5.32e2]
    Keq = [2.15e2, 5.93e-5, 6.28e-2, 1.18e-5, 1.03e3, 1.92e5, 4.50e1]

    m = Model("water_gas_shift_qea", T=T, R=8.617e-5, Tref=298.15)
    s = m.site("*", density=1.0)
    CO = m.gas("CO")
    H2O = m.gas("H2O")
    CO2 = m.gas("CO2")
    H2 = m.gas("H2")
    COs = m.adsorbate("CO*", site=s)
    H2Os = m.adsorbate("H2O*", site=s)
    OHs = m.adsorbate("OH*", site=s)
    Hs = m.adsorbate("H*", site=s)
    Os = m.adsorbate("O*", site=s)
    CO2s = m.adsorbate("CO2*", site=s)

    m.step(CO + s >> COs, Keq=Keq[0], equilibrated=True, name="S1: CO + * -> CO*")
    m.step(H2O + s >> H2Os, Keq=Keq[1], equilibrated=True, name="S2: H2O + * -> H2O*")
    m.step(H2Os + s >> OHs + Hs, Keq=Keq[2], equilibrated=True, name="S3: H2O* + * -> OH* + H*")
    m.step(OHs + s >> Os + Hs, kf=kf[3], Keq=Keq[3], name="S4: OH* + * -> O* + H* (kinetic)")
    m.step(COs + Os >> CO2s + s, kf=kf[4], Keq=Keq[4], name="S5: CO* + O* -> CO2* + * (kinetic)")
    m.step(CO2s >> CO2 + s, Keq=Keq[5], equilibrated=True, name="S6: CO2* -> CO2 + *")
    m.step(2 * Hs >> H2 + 2 * s, Keq=Keq[6], equilibrated=True, name="S7: 2 H* -> H2 + 2*")

    reactor = DifferentialReactor({CO: 0.07 * P, H2O: 0.21 * P, CO2: 0.085 * P, H2: 0.38 * P})
    return m, reactor


def co_oxidation_cstr(T: float = 500.0):
    """CO oxidation in a CSTR with inflow (gas concentrations are unknowns)."""
    m, _ = co_oxidation(T=T)
    CO = m._by_name["CO"]
    O2 = m._by_name["O2"]
    CO2 = m._by_name["CO2"]
    reactor = CSTR(inlet={CO: 1.0, O2: 0.5, CO2: 0.0}, tau=1.0, cat_density=1e-3)
    return m, reactor


def selective_oxidation(T: float = 500.0, P_O2: float = 0.5):
    """A branching ("selectivity") mechanism: one intermediate, two products.

    A reactant ``A`` and oxygen co-adsorb and couple into a common surface
    oxygenate ``AO*``, which then **branches** into a partial-oxidation product
    ``P1`` (desired) or is further oxidized to ``P2`` (over-oxidation)::

        A   + *   <=> A*
        O2  + 2*  <=> 2 O*
        A*  + O*  <=> AO* + *          (form the common intermediate)
        AO*       <=> P1 + *           (selective:  AO* -> P1)
        AO* + O*  <=> P2 + 2*          (unselective: AO* + O* -> P2)

    Selectivity to ``P1`` is ``S = r(P1) / (r(P1) + r(P2)) = k4 / (k4 + k5 θ_O)``,
    so it falls as the oxygen coverage ``θ_O`` rises with ``P_O2`` — the classic
    activity/selectivity tradeoff. Adsorption is rate-limiting and the surface
    branch is faster, so the steady state is well conditioned; the ``θ_O`` that
    sets the branching ratio still tracks ``P_O2``.

    Compositions use abstract elements ``A`` and ``O`` (``P1`` = ``AO``,
    ``P2`` = ``AO2``) so the mechanism is element- and site-balanced.

    Returns ``(model, reactor)`` at fixed gas partial pressures, ready for
    :func:`~discopt.mkm.solve_steady_state` (use ``coordinates="log"`` with a
    warm start for the differentiable degree-of-(selectivity-)control analysis).
    """
    m = Model("selective_oxidation", T=T, R=8.617e-5, Tref=298.15)
    s = m.site("*", density=1.0)
    g = dict(S=0.0005)
    A = m.gas("A", H=0.0, composition={"A": 1}, **g)
    O2 = m.gas("O2", H=0.0, composition={"O": 2}, **g)
    P1 = m.gas("P1", H=-2.0, composition={"A": 1, "O": 1}, **g)
    P2 = m.gas("P2", H=-4.0, composition={"A": 1, "O": 2}, **g)
    As = m.adsorbate("A*", site=s, H=-0.15, S=0.0005, composition={"A": 1})
    Os = m.adsorbate("O*", site=s, H=-0.08, S=0.0005, composition={"O": 1})
    AOs = m.adsorbate("AO*", site=s, H=-0.30, S=0.0005, composition={"A": 1, "O": 1})

    m.step(A + s >> As, A=1e4, Ea=0.0, name="A adsorption")
    m.step(O2 + 2 * s >> 2 * Os, A=1e4, Ea=0.0, name="O2 dissociation")
    m.step(As + Os >> AOs + s, A=1e6, Ea=0.20, name="form AO*")
    m.step(AOs >> P1 + s, A=1e6, Ea=0.40, name="AO* -> P1 (selective)")
    m.step(AOs + Os >> P2 + 2 * s, A=1e6, Ea=0.33, name="AO* -> P2 (over-oxidation)")

    reactor = DifferentialReactor({A: 1.0, O2: P_O2, P1: 0.0, P2: 0.0})
    return m, reactor


def overcomplete_co_oxidation(T: float = 500.0):
    """CO oxidation written as an **over-complete candidate mechanism**.

    The true minimal Langmuir-Hinshelwood mechanism (the three steps of
    :func:`co_oxidation`) plus four decoy steps that a combinatoric generator
    would also propose but that carry negligible flux:

    - an Eley-Rideal step ``CO + O* -> CO2 + *`` (high barrier),
    - a redundant route through a bound ``OCO*`` intermediate (slow formation),
    - water adsorption ``H2O + * <=> H2O*`` (a spectator; ``H2O`` is not fed).

    Used by :mod:`discopt.mkm.select` to demonstrate recovering the minimal
    mechanism. Returns ``(model, reactor)`` like :func:`co_oxidation`; the decoy
    steps are named with a ``"[decoy]"`` prefix so tests can identify them.
    """
    m = Model("co_ox_overcomplete", T=T, R=8.617e-5, Tref=298.15)
    s = m.site("Pt", density=1.0)
    CO = m.gas("CO", H=0.0, S=0.0020)
    O2 = m.gas("O2", H=0.0, S=0.0021)
    CO2 = m.gas("CO2", H=-3.0, S=0.0023)
    H2O = m.gas("H2O", H=-0.5, S=0.0021)
    COs = m.adsorbate("CO*", site=s, H=-0.8, S=0.0005)
    Os = m.adsorbate("O*", site=s, H=-0.3, S=0.0005)
    OCOs = m.adsorbate("OCO*", site=s, H=-0.6, S=0.0005)
    H2Os = m.adsorbate("H2O*", site=s, H=-1.3, S=0.0005)  # bound tightly: tame k_reverse

    # true minimal Langmuir-Hinshelwood mechanism
    m.step(CO + s >> COs, A=1e4, Ea=0.0, name="CO adsorption")
    m.step(O2 + 2 * s >> 2 * Os, A=1e4, Ea=0.0, name="O2 dissociation")
    m.step(COs + Os >> CO2 + 2 * s, A=1e8, Ea=0.7, name="surface reaction")
    # decoys (negligible flux at any condition); irreversible so rate constants stay modest
    m.step(CO + Os >> CO2 + s, A=1e8, Ea=2.0, irreversible=True, name="[decoy] Eley-Rideal")
    m.step(COs + Os >> OCOs + s, A=1e8, Ea=1.8, irreversible=True, name="[decoy] OCO* formation")
    m.step(OCOs >> CO2 + s, A=1e6, Ea=0.5, irreversible=True, name="[decoy] OCO* decomposition")
    m.step(H2O + s >> H2Os, A=1e4, Ea=0.0, name="[decoy] water adsorption")

    m.infer_composition()
    reactor = DifferentialReactor({CO: 1.0, O2: 0.5, CO2: 0.0, H2O: 0.0})
    return m, reactor


def adiabatic_cstr(T_in: float = 500.0):
    """Exothermic ``A -> B`` on a catalyst in an adiabatic CSTR (SI units).

    A simple two-step surface mechanism (``A + * <=> A*``; ``A* <=> B + *``) with
    a net exothermic heat of reaction, so the steady-state temperature rises
    above the feed temperature. Returns ``(model, reactor, energy)`` for use with
    ``solve_steady_state(..., energy=energy)``.
    """
    from discopt.mkm.energy import EnergyBalance

    m = Model("adiabatic_cstr", T=T_in, R=8.314, Tref=298.15)
    s = m.site("cat", density=1.0)
    A = m.gas("A", H=0.0, S=0.0, Cp=30.0)
    B = m.gas("B", H=-12000.0, S=0.0, Cp=30.0)  # -12 kJ/mol overall
    As = m.adsorbate("A*", site=s, H=-6000.0, S=0.0, Cp=30.0)

    # comparable rate constants (no fast-adsorption stiffness)
    m.step(A + s >> As, A=1e2, Ea=0.0, name="adsorption")
    m.step(As >> B + s, A=1e7, Ea=60000.0, name="surface reaction")

    reactor = CSTR(inlet={A: 1.0, B: 0.0}, tau=1.0, cat_density=1.0)
    energy = EnergyBalance(T_in=T_in, Q=0.0)
    return m, reactor, energy
