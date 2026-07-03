"""Regression tests for the high-severity correctness fixes (REVIEW.md H1-H7,
M5, M6). Each test fails on the pre-fix code and passes after.
"""

import numpy as np
import pytest
import sympy as sp

import discopt.mkm as mk
from discopt.mkm import numeric, select
from discopt.mkm.examples import co_oxidation, co_oxidation_cstr, water_gas_shift_qea
from discopt.mkm.symbolic import lumped_rate_expression

R, T, F = 8.617e-5, 298.0, 1.0


# --- H1: explicit-Keq electrochemical steps carry the n F U shift ------------
def _explicit_ec_model(U):
    """Reversible electrochemical step given as explicit kf/Keq (a DFT/SI table
    style step) plus a fast chemical recombination."""
    m = mk.Model("ec_explicit", T=T, R=R, U=U, F=F)
    s = m.site("*", density=1.0)
    Hp = m.gas("Hp", H=0.0, S=0.0, composition={"H": 1})
    H2 = m.gas("H2", H=0.0, S=0.0, composition={"H": 2})
    Hs = m.adsorbate("H*", site=s, H=-0.1, S=0.0, composition={"H": 1})
    m.step(Hp + s >> Hs, kf=1.0, Keq=1.0, n_electrons=1, beta=0.5)
    m.step(2 * Hs >> H2 + 2 * s, kf=1e6, Keq=1e6)
    return m, Hp, H2, Hs


def test_explicit_keq_electrochemical_detailed_balance():
    # The discopt solve must agree with the (always-correct) numeric path, which
    # applies kr = kf / (Keq * exp(-nFU/RT)). Pre-fix the discopt reverse rate
    # dropped the potential shift, so coverages disagreed by ~exp(nFU/RT).
    U = 0.3
    m, Hp, H2, Hs = _explicit_ec_model(U)
    th, fr = numeric.steady_state_numeric(m, {Hp: 1.0, H2: 1e-3}, T,
                                          theta0={Hs: 0.3})
    sol = mk.solve_steady_state(m, mk.DifferentialReactor({Hp: 1.0, H2: 1e-3}),
                                theta0=th, active_tol=1e-13)
    assert sol.coverage(Hs) == pytest.approx(th[Hs], rel=1e-4)


def test_explicit_keq_equilibrated_electrochemical_responds_to_potential():
    # An equilibrated explicit-Keq faradaic step's coverages must move with U.
    def solve(U):
        m = mk.Model("ec_eq", T=T, R=R, U=U, F=F)
        s = m.site("*", density=1.0)
        Hp = m.gas("Hp", H=0.0, S=0.0, composition={"H": 1})
        H2 = m.gas("H2", H=0.0, S=0.0, composition={"H": 2})
        Hs = m.adsorbate("H*", site=s, H=-0.1, S=0.0, composition={"H": 1})
        m.step(Hp + s >> Hs, equilibrated=True, Keq=1.0, n_electrons=1, beta=0.5)
        m.step(2 * Hs >> H2 + 2 * s, A=1e6, Ea=0.3)
        sol = mk.solve_steady_state(m, mk.DifferentialReactor({Hp: 1.0, H2: 1e-3}))
        return sol.coverage(Hs)
    assert abs(solve(0.2) - solve(-0.2)) > 1e-3


# --- H3: batch reactor has no steady state -----------------------------------
def test_batch_steady_state_raises():
    m, _ = co_oxidation(500.0)
    CO, O2 = m._by_name["CO"], m._by_name["O2"]
    with pytest.raises(ValueError, match="batch"):
        mk.solve_steady_state(m, mk.Batch({CO: 1.0, O2: 0.5}))


# --- H4: fractional stoichiometric coefficients survive symbolic derivation --
def test_symbolic_keeps_fractional_coefficient():
    m = mk.Model("half_o2", T=500.0, R=8.617e-5)
    s = m.site("*", density=1.0)
    O2 = m.gas("O2", H=0.0, S=0.002, composition={"O": 2})
    CO = m.gas("CO", H=0.0, S=0.002, composition={"C": 1, "O": 1})
    CO2 = m.gas("CO2", H=-3.0, S=0.002, composition={"C": 1, "O": 2})
    Os = m.adsorbate("O*", site=s, H=-0.3, S=0.0005, composition={"O": 1})
    m.step(0.5 * O2 + s >> Os, equilibrated=True, Keq=10.0)
    m.step(CO + Os >> CO2 + s, kf=1e3, Keq=1e5)
    m.infer_composition()
    rate, symbols = lumped_rate_expression(m, CO2)
    # the O2 partial-pressure symbol must appear (int(0.5)=0 previously deleted it)
    assert symbols["P"][O2] in rate.free_symbols


# --- H6: numeric paths reject equilibrated steps instead of silently deleting -
def test_numeric_rejects_equilibrated():
    m, reactor = water_gas_shift_qea(T=480.0)
    with pytest.raises(ValueError, match="equilibrated"):
        numeric.steady_state_numeric(m, {}, 480.0)


def test_select_rejects_equilibrated():
    m, reactor = water_gas_shift_qea(T=480.0)
    with pytest.raises(ValueError, match="equilibrated"):
        select.reduce_by_drc(m, [{}], target="H2")


# --- H2 + M6 + T1: log-coordinate solve honors the reactor gas balance -------
def test_log_coordinates_cstr_matches_linear():
    # A CSTR solved in log coordinates must match the linear CSTR solve; pre-fix
    # the log path never constrained the gas concentrations (they drifted to the
    # bound midpoint) and reported "optimal" anyway.
    m, reactor = co_oxidation_cstr(500.0)
    lin = mk.solve_steady_state(m, reactor)
    th = {a: lin.coverage(a) for a in m.adsorbates}
    log = mk.solve_steady_state(m, reactor, coordinates="log", theta0=th)
    for g in m.gas_species:
        assert log.gas_concentration(g) == pytest.approx(lin.gas_concentration(g), rel=1e-3)


# --- H5: one-way flux screen keeps a fast spectator equilibrium step ----------
def test_reduce_by_drc_keeps_spectator_equilibrium_step():
    # Main path A + * <=> A* -> B; plus a reversible dead-end C + * <=> C* whose
    # *net* flux is ~0 at steady state (nothing consumes C*) but whose one-way
    # flux is large and which occupies ~half the sites. A net-flux screen (pre-fix)
    # drops it; the one-way screen keeps it, as it must (removing it frees sites
    # and changes the turnover rate).
    m = mk.Model("spectator", T=500.0, R=8.617e-5, Tref=298.15)
    s = m.site("*", density=1.0)
    A = m.gas("A", H=0.0, S=0.001, composition={"A": 1})
    B = m.gas("B", H=-1.0, S=0.001, composition={"B": 1})
    C = m.gas("C", H=0.0, S=0.001, composition={"C": 1})
    As = m.adsorbate("A*", site=s, H=-0.3, S=0.001, composition={"A": 1})
    Cs = m.adsorbate("C*", site=s, H=-0.3, S=0.001, composition={"C": 1})
    m.step(A + s >> As, A=1e5, Ea=0.0, name="A adsorption")
    m.step(As >> B + s, A=1e5, Ea=0.4, irreversible=True, name="A* -> B")
    m.step(C + s >> Cs, A=1e5, Ea=0.0, name="spectator")
    m.infer_composition()
    res = select.reduce_by_drc(m, [{"A": 1.0, "C": 1.0}], target="B", flux_tol=1e-6)
    assert "spectator" in set(res.kept)
