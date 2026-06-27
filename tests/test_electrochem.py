"""Electrochemistry: Butler-Volmer kinetics, Tafel analysis, CHE thermodynamics,
and the descriptor volcano.

The electrode potential ``U`` enters the same chokepoints as temperature: the
reaction free energy (``+ n F U``) and the forward barrier (``+ beta n F U``).
Because the reverse rate is derived from ``K_eq``, detailed balance holds at every
potential, and the AD machinery gives the Tafel slope and transfer coefficient.
"""

import numpy as np
import pytest

import discopt_mkm as mk
import discopt_mkm.electrochem as ec
from discopt_mkm.numeric import (
    net_rate,
    rate_constants,
    rates_of_progress,
    reaction_free_energy,
    steady_state_numeric,
)

R, T, F = 8.617e-5, 298.0, 1.0


def _her(beta=0.5, A_volmer=1e2):
    """Ideal HER: a slow irreversible electrochemical Volmer step followed by a
    fast chemical Tafel recombination (so the site stays nearly free and the
    current tracks the Volmer rate constant -> a clean Tafel slope)."""
    m = mk.Model("her", T=T, R=R, U=0.0, F=F)
    s = m.site("*", density=1.0)
    Hp = m.gas("Hp", H=0.0, S=0.0, composition={"H": 1})
    H2 = m.gas("H2", H=0.0, S=0.0, composition={"H": 2})
    Hs = m.adsorbate("H*", site=s, H=-0.10, S=0.0, composition={"H": 1})
    volmer = m.step(Hp + s >> Hs, A=A_volmer, Ea=0.30, n_electrons=1, beta=beta, irreversible=True)
    m.step(2 * Hs >> H2 + 2 * s, A=1e10, Ea=0.20)
    return m, Hp, H2, Hs, volmer


def _her_current(m, Hp, H2, U):
    m.U = U
    th, fr = steady_state_numeric(m, {Hp: 1.0, H2: 1e-3}, T, theta0={m._by_name["H*"]: 0.3})
    kf, kr = rate_constants(m, T, th)
    rop = rates_of_progress(m, kf, kr, th, fr, {Hp: 1.0, H2: 1e-3})
    return m.F * sum(r.n_electrons * rop[r] for r in m.reactions if r.is_electrochemical)


def test_detailed_balance_across_potential():
    # a reversible electrochemical step: k_f / k_r must equal exp(-dG(U)/RT) at every U
    m = mk.Model("ec", T=T, R=R, U=0.0, F=F)
    s = m.site("*", density=1.0)
    Hp = m.gas("Hp", H=0.0, S=0.0); m.gas("H2", H=0.0, S=0.0)
    Hs = m.adsorbate("H*", site=s, H=-0.1, S=0.0)
    rxn = m.step(Hp + s >> Hs, A=1e4, Ea=0.3, n_electrons=1, beta=0.5)
    m.step(2 * Hs >> m._by_name["H2"] + 2 * s, A=1e8, Ea=0.4)
    for U in (-0.2, -0.1, 0.0, 0.1, 0.2):
        m.U = U
        kf, kr = rate_constants(m, T, {Hs: 0.3})
        dG = reaction_free_energy(m, rxn, T, {Hs: 0.3})
        assert kf[rxn] / kr[rxn] == pytest.approx(np.exp(-dG / (R * T)), rel=1e-9)


def test_che_free_energy_shifts_linearly_with_potential():
    m, _ = ec.orr_4e(descriptor=0.9)
    _, _, dG0 = ec.che_free_energies(m, U=0.0)
    _, _, dGU = ec.che_free_energies(m, U=0.5)
    # each one-electron step shifts by n*F*U = 0.5 eV
    assert np.allclose(dGU - dG0, 0.5, atol=1e-9)
    assert dG0.sum() == pytest.approx(-4.92, abs=1e-6)   # 4-electron equilibrium 1.23 V


def test_tafel_slope_and_transfer_coefficient():
    for beta in (0.5, 0.3):
        m, Hp, H2, Hs, volmer = _her(beta=beta)
        # finite-difference Tafel slope on the numeric current (deep in the Tafel region)
        Us = np.linspace(-0.45, -0.30, 6)
        j = np.array([_her_current(m, Hp, H2, U) for U in Us])
        fd_slope = 1.0 / np.polyfit(Us, np.log10(j), 1)[0]   # V/decade
        assert fd_slope == pytest.approx(-2.303 * R * T / (beta * F), rel=2e-2)
        # analytic Tafel slope + transfer coefficient through the L3 steady state
        m.U = -0.40
        th, _ = steady_state_numeric(m, {Hp: 1.0, H2: 1e-3}, T, theta0={Hs: 0.3})
        sol = mk.solve_steady_state(m, mk.DifferentialReactor({Hp: 1.0, H2: 1e-3}),
                                    theta0=th, active_tol=1e-13)
        assert ec.tafel_slope(sol) == pytest.approx(-2.303 * R * T / (beta * F), rel=2e-2)
        assert ec.apparent_transfer_coefficient(sol) == pytest.approx(beta, abs=2e-2)


def test_orr_limiting_potential_and_descriptor_volcano():
    m, _ = ec.orr_4e(descriptor=0.9)
    # for x=0.9 the steps are [x-1.72, x-3.2, -x, -x]; U_L = min(1.72-x, 3.2-x, x, x) = 0.82
    assert ec.limiting_potential(m) == pytest.approx(0.82, abs=1e-6)

    # the discopt LP optimum matches a brute-force volcano sweep peak
    xs = np.linspace(0.4, 1.4, 101)
    _, ul = ec.che_volcano(m, ec.set_orr_descriptor, xs)
    sweep_peak_x = xs[ul.argmax()]
    opt = ec.optimize_descriptor(ec.ORR_SCALING, ec.ORR_N_ELECTRONS, bounds=(0.4, 1.4), U_eq=ec.ORR_U_EQ)
    assert opt["descriptor"] == pytest.approx(0.86, abs=1e-2)        # OOH/OH scaling crossing
    assert opt["descriptor"] == pytest.approx(sweep_peak_x, abs=2e-2)
    assert opt["overpotential"] == pytest.approx(0.37, abs=1e-2)


def test_orr_current_is_cathodic_and_grows_below_the_onset():
    m, _ = ec.orr_4e(descriptor=0.9)
    O2, Hp, H2O = (m._by_name[n] for n in ("O2", "H⁺", "H2O"))
    pres = {O2: 1.0, Hp: 1.0, H2O: 1.0}

    def current(U, guess):
        m.U = U
        th, fr = steady_state_numeric(m, pres, m.T, theta0=guess)
        kf, kr = rate_constants(m, m.T, th)
        rop = rates_of_progress(m, kf, kr, th, fr, pres)
        return m.F * sum(r.n_electrons * rop[r] for r in m.reactions if r.is_electrochemical), th

    g = {a: 0.1 for a in m.adsorbates}
    j_hi, g = current(1.0, g)     # above the limiting potential: nearly no current
    j_lo, g = current(0.5, g)     # below it: cathodic current flows
    assert j_lo > 0 and j_hi >= 0
    assert j_lo > 100 * abs(j_hi)


def test_recover_transfer_coefficient_from_tafel_data():
    # fit beta from a noisy Tafel plot (log|j| linear in U with slope -beta F / 2.303 RT)
    m, Hp, H2, Hs, volmer = _her(beta=0.5)
    Us = np.linspace(-0.45, -0.28, 8)
    j = np.array([_her_current(m, Hp, H2, U) for U in Us])
    rng = np.random.default_rng(0)
    logj = np.log10(j) + rng.normal(0, 0.01, j.shape)
    slope = np.polyfit(Us, logj, 1)[0]          # d log10 j / dU
    beta_fit = -slope * 2.303 * R * T / F
    assert beta_fit == pytest.approx(0.5, abs=3e-2)
