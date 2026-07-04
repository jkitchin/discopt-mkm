"""Regression tests for the high-severity correctness fixes (REVIEW.md H1-H7,
M5, M6). Each test fails on the pre-fix code and passes after.
"""

import warnings

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


@pytest.mark.slow
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
@pytest.mark.slow
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


# --- H7: the estimator handles explicit-rate steps and rejects equilibrated --
def _explicit_rate_fit_model(temp=500.0):
    """A model mixing an explicit-rate adsorption step (kf/Keq) with an Arrhenius
    surface step. k_forward reads kf_param and k_reverse reads Keq_param, so both
    handles must be wired freshly on each fit's discopt model."""
    m = mk.Model("h7", T=temp, R=8.617e-5, Tref=298.15)
    s = m.site("s", density=1.0)
    A = m.gas("A", H=0.0, S=0.0)
    B = m.gas("B", H=-1.0, S=0.0)
    As = m.adsorbate("A*", site=s, H=-0.3, S=0.0)
    m.step(A + s >> As, kf=5.0, Keq=10.0, name="ads (explicit)")
    surf = m.step(As >> B + s, A=1e3, Ea=0.3, name="surf (arrhenius)")
    return m, A, B, As, surf


@pytest.mark.slow
def test_fit_with_explicit_rate_step_no_stale_handles():
    # Solve the fit model ONCE first (populating handles from one discopt model),
    # then fit (which builds a fresh discopt model). Pre-fix the explicit step's
    # kf_param/Keq_param were never re-wired, so the residuals referenced stale
    # handles from the previous model and the fit crashed.
    TS = [480.0, 500.0, 520.0]
    data = []
    for temp in TS:
        mt, At, Bt, _, _ = _explicit_rate_fit_model(temp)
        sol = mk.solve_steady_state(mt, mk.DifferentialReactor({At: 1.0, Bt: 0.0}))
        data.append(sol.production_rate(Bt))

    m, A, B, As, surf = _explicit_rate_fit_model(500.0)
    mk.solve_steady_state(m, mk.DifferentialReactor({A: 1.0, B: 0.0}))  # populate stale handles

    obs = [mk.Observation(response=B, value=v, T=temp, pressures={A: 1.0, B: 0.0}, sigma=0.01)
           for temp, v in zip(TS, data)]
    fit = [mk.FitParam(surf, "A", lb=1e2, ub=1e4, init=5e2),
           mk.FitParam(surf, "Ea", lb=0.05, ub=0.6, init=0.2)]
    res = mk.fit_kinetics(m, obs, fit)

    A_key = next(k for k in res.parameters if k.startswith("A_"))
    Ea_key = next(k for k in res.parameters if k.startswith("Ea_"))
    assert res.parameters[Ea_key] == pytest.approx(0.3, abs=2e-2)
    assert res.parameters[A_key] == pytest.approx(1e3, rel=0.2)


def test_fit_equilibrated_step_raises():
    # Fitting a mechanism with an equilibrated step must raise clearly rather than
    # silently fit a mechanism with that step deleted (its rate law is dropped).
    m = mk.Model("h7eq", T=500.0, R=8.617e-5)
    s = m.site("s", density=1.0)
    A = m.gas("A", H=0.0, S=0.0)
    B = m.gas("B", H=-1.0, S=0.0)
    As = m.adsorbate("A*", site=s, H=-0.3, S=0.0)
    m.step(A + s >> As, Keq=5.0, equilibrated=True)
    surf = m.step(As >> B + s, A=1e3, Ea=0.3)

    obs = [mk.Observation(response=B, value=1.0, T=500.0, pressures={A: 1.0, B: 0.0})]
    fit = [mk.FitParam(surf, "A", lb=1e2, ub=1e4, init=5e2)]
    with pytest.raises(NotImplementedError, match="equilibrated"):
        mk.fit_kinetics(m, obs, fit)


# --- M5: the numeric warm-start helper reports a spurious (negative) root -----
def test_numeric_warm_start_rejects_spurious_negative_root():
    # From a bad seed CO oxidation's fsolve converges (ier=1) to a non-physical
    # root with O* < 0; the helper must raise rather than return it silently.
    m, _ = co_oxidation(500.0)
    CO, O2 = m._by_name["CO"], m._by_name["O2"]
    with pytest.raises(RuntimeError):
        numeric.steady_state_numeric(m, {CO: 1.0, O2: 0.5}, m.T,
                                     theta0={a: 0.1 for a in m.adsorbates})


# --- M1: re-solving a model must not invalidate an earlier solution ----------
@pytest.mark.slow
def test_earlier_solution_survives_a_resolve():
    # Two solves of the SAME model at different T rewire the shared parameter
    # handles on the Species/Reaction objects. Pre-fix, the first solution rebuilt
    # its rate expressions lazily from those (now-stale) handles and raised
    # "Parameter 'A_2' not found in model". The solution must snapshot what it
    # needs at construction time and stay evaluable.
    from discopt.mkm.analysis import degree_of_rate_control

    m, _ = co_oxidation(500.0)
    CO, O2, CO2, COs = (m._by_name[n] for n in ("CO", "O2", "CO2", "CO*"))
    reactor = mk.DifferentialReactor({CO: 1.0, O2: 0.5, CO2: 0.0})

    sol1 = mk.solve_steady_state(m, reactor)
    r1 = sol1.production_rate(CO2)

    # a fresh single solve at T=500 for the reference value
    mf, _ = co_oxidation(500.0)
    ref = mk.solve_steady_state(
        mf, mk.DifferentialReactor({mf._by_name["CO"]: 1.0, mf._by_name["O2"]: 0.5,
                                    mf._by_name["CO2"]: 0.0})
    ).production_rate(mf._by_name["CO2"])
    assert r1 == pytest.approx(ref, rel=1e-9)

    # re-solve the ORIGINAL model at a different T (rewires the shared handles)
    m.T = 520.0
    mk.solve_steady_state(m, reactor)

    # sol1's accessors must still work and give the original (T=500) answers
    assert sol1.production_rate(CO2) == pytest.approx(ref, rel=1e-9)
    assert 0.0 <= sol1.coverage(COs) <= 1.0
    X = degree_of_rate_control(sol1, species=CO2)
    assert sum(X.values()) == pytest.approx(1.0, abs=1e-3)
    sol1.to_dict()  # must not raise


# --- M2: the reaction parser must not split a trailing '+' in a species name -
def test_parser_preserves_ionic_species():
    from discopt.mkm.spec import parse_equation

    r, p, irr = parse_equation("O2 + H+ + * -> OOH*")
    assert "H+" in r and "H" not in r
    assert r == {"O2": 1.0, "H+": 1.0, "*": 1.0}
    assert p == {"OOH*": 1.0} and irr is True
    # an existing spaced equation still parses correctly
    r2, p2, irr2 = parse_equation("CO + 2 * <=> 2 CO*")
    assert r2 == {"CO": 1.0, "*": 2.0} and p2 == {"CO*": 2.0} and irr2 is False


# --- M3: energy-diagram TS carries the electrochemical barrier shift ----------
def test_energy_diagram_electrochemical_ts_shift():
    from discopt.mkm import viz

    U = 0.5
    m = mk.Model("ec_diagram", T=298.0, R=R, U=U, F=F)
    s = m.site("*", density=1.0)
    Hp = m.gas("H+", H=0.0, S=0.0, composition={"H": 1})
    Hs = m.adsorbate("H*", site=s, H=-0.3, S=0.0, composition={"H": 1})
    rxn = m.step(Hp + s >> Hs, A=1e9, Ea=0.2, n_electrons=1, beta=0.5)

    states, ts = viz.energy_profile(m)
    expected = states[0] + rxn.Ea + rxn.beta * rxn.n_electrons * m.F * m.U
    assert ts[0] == pytest.approx(expected)
    # the shifted barrier sits above the (potential-shifted) product state
    assert ts[0] > states[1]

    ax = viz.energy_diagram(m)  # exercises the drawing path
    import matplotlib.pyplot as plt

    plt.close(ax.figure)


# --- M4: energy balance uses the thermo-model Cp (not a zero fallback) --------
def _nasa7_const_cp(H, S, Cp=30.0, Rg=8.314, Tref=298.15):
    a1 = Cp / Rg
    a6 = (H - Cp * Tref) / Rg
    a7 = (S - Cp * np.log(Tref)) / Rg
    return mk.NASA7([a1, 0.0, 0.0, 0.0, 0.0, a6, a7])


def _nasa7_energy_cstr(T=500.0):
    from discopt.mkm.energy import EnergyBalance

    m = mk.Model("nasa_energy", T=T, R=8.314, Tref=298.15)
    s = m.site("cat", density=1.0)
    A = m.gas("A", thermo=_nasa7_const_cp(0.0, 0.0))
    B = m.gas("B", thermo=_nasa7_const_cp(-12000.0, 0.0))
    m.adsorbate("A*", site=s, H=-6000.0, S=0.0, Cp=30.0)
    m.step(A + s >> m._by_name["A*"], A=1e2, Ea=0.0, name="ads")
    m.step(m._by_name["A*"] >> B + s, A=1e7, Ea=60000.0, name="surf")
    reactor = mk.CSTR(inlet={A: 1.0, B: 0.0}, tau=1.0, cat_density=1.0)
    return m, reactor, EnergyBalance(T_in=T), A


@pytest.mark.slow
def test_energy_balance_uses_thermo_cp():
    # The gas heat capacity lives in a NASA7 model (Cp_param is None). Pre-fix the
    # inlet-stream cp_in fell back to the constant g.Cp (0), degenerating the
    # balance to q + Q = 0 (unphysical T). It must take Cp from the thermo model.
    m, reactor, energy, A = _nasa7_energy_cstr(500.0)
    # the thermo Cp itself is nonzero (pre-fix the energy path used the constant
    # g.Cp == 0 here) and the inlet-stream mixture heat capacity is positive
    assert A.thermo.Cp(500.0, m.R) == pytest.approx(30.0, rel=1e-6)
    cp_in = sum(g.thermo.Cp(500.0, m.R) for g in m.gas_species if g.thermo is not None)
    assert cp_in > 0.0

    sol = mk.solve_steady_state(m, reactor, energy=energy)
    T = sol.temperature()
    conv = 1.0 - sol.gas_concentration(A)
    assert T > 500.0  # exothermic: temperature rises (not the degenerate result)

    # self-consistent: an isothermal solve at the solved T gives the same conversion
    m2, r2, _, A2 = _nasa7_energy_cstr(T)
    conv2 = 1.0 - mk.solve_steady_state(m2, r2).gas_concentration(A2)
    assert conv == pytest.approx(conv2, abs=2e-3)


def test_energy_balance_rejects_equilibrated_step():
    # heat_release_rate needs each step's rate of progress; an equilibrated step's
    # rate is an unknown extent, so the energy path must raise clearly.
    from discopt.mkm.energy import EnergyBalance

    m = mk.Model("eq_energy", T=500.0, R=8.314, Tref=298.15)
    s = m.site("cat", density=1.0)
    A = m.gas("A", H=0.0, S=0.0, Cp=30.0)
    B = m.gas("B", H=-12000.0, S=0.0, Cp=30.0)
    As = m.adsorbate("A*", site=s, H=-6000.0, S=0.0, Cp=30.0)
    m.step(A + s >> As, Keq=5.0, equilibrated=True)
    m.step(As >> B + s, A=1e7, Ea=60000.0)
    reactor = mk.CSTR(inlet={A: 1.0, B: 0.0}, tau=1.0, cat_density=1.0)
    with pytest.raises(NotImplementedError, match="equilibrated"):
        mk.solve_steady_state(m, reactor, energy=EnergyBalance(T_in=500.0))


# --- M7: the MILP capacity bound is pressure-aware ---------------------------
def _eley_rideal_model(T=500.0):
    m = mk.Model("er", T=T, R=8.617e-5, Tref=298.15)
    s = m.site("*", density=1.0)
    A = m.gas("A", H=0.0, S=0.001, composition={"A": 1})
    P = m.gas("P", H=-1.0, S=0.001, composition={"A": 2})
    As = m.adsorbate("A*", site=s, H=-0.3, S=0.001, composition={"A": 1})
    m.step(A + s >> As, A=1e5, Ea=0.0, name="ads")
    # Eley-Rideal step consumes gas A: its steady-state flux at P_A=10 is ~10x its
    # bare kf, so a cap of max(kf, kr) (pre-fix) makes the true mechanism infeasible.
    m.step(A + As >> P + s, A=2.0, Ea=0.0, irreversible=True, name="eley-rideal")
    m.infer_composition()
    return m, A, P


@pytest.mark.slow
def test_milp_selection_pressure_aware_cap():
    m, A, P = _eley_rideal_model(500.0)
    obs = mk.solve_steady_state(m, mk.DifferentialReactor({A: 10.0, P: 0.0})).production_rate(P)
    assert obs > 2.0  # exceeds the bare kf of the Eley-Rideal step
    res = mk.select_subgraph(m, [{"A": 10.0}], "P", [obs], engine="milp")
    assert res.status == "optimal"
    assert set(res.selected) == {"ads", "eley-rideal"}
    assert res.misfit < 1e-6


# --- M8: _drc_table returns None (not 0.0) when no solve yields sensitivities -
def test_drc_table_none_when_all_solves_fail(monkeypatch):
    from discopt.mkm import select

    m = mk.Model("drc_none", T=500.0, R=8.617e-5, Tref=298.15)
    s = m.site("*", density=1.0)
    A = m.gas("A", H=0.0, S=0.001, composition={"A": 1})
    B = m.gas("B", H=-1.0, S=0.001, composition={"B": 1})
    As = m.adsorbate("A*", site=s, H=-0.3, S=0.001, composition={"A": 1})
    m.step(A + s >> As, A=1e5, Ea=0.0, name="ads")
    m.step(As >> B + s, A=1e5, Ea=0.4, irreversible=True, name="surf")
    m.infer_composition()

    def boom(*a, **k):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(select, "solve_steady_state", boom)
    table = select._drc_table(m, [{"A": 1.0}], "B", 500.0)
    assert all(v is None for v in table.values())  # None, not a misleading 0.0


# --- M9: spec validation rejects unknown keys and cross-field reactor misuse --
def test_spec_rejects_unknown_keys():
    from pydantic import ValidationError

    from discopt.mkm.spec import ModelSpec, ReactionSpec

    with pytest.raises(ValidationError):
        ModelSpec(**{"reactons": []})           # typo for 'reactions'
    with pytest.raises(ValidationError):
        ReactionSpec(equation="A -> B", n_electron=1)  # typo for 'n_electrons'


def test_reactor_cross_field_misuse_raises():
    from discopt.mkm.spec import from_spec

    base = {
        "name": "x", "sites": [{"name": "*", "density": 1.0}],
        "gas": [{"name": "A"}], "adsorbates": [{"name": "A*", "site": "*"}],
        "reactions": [{"equation": "A + * <=> A*", "A": 1e3, "Ea": 0.0}],
    }
    # a CSTR given 'pressures' (which belongs to a differential reactor) is a
    # silent all-zero feed pre-fix; it must raise a clear error now.
    with pytest.raises(ValueError, match="cstr"):
        from_spec({**base, "reactor": {"type": "cstr", "pressures": {"A": 1.0}}})
    # a differential 'pressures' key naming an adsorbate rather than a gas
    with pytest.raises(ValueError, match="gas"):
        from_spec({**base, "reactor": {"type": "differential", "pressures": {"A*": 1.0}}})


# --- M10: to_spec round-trips thermo models and the reactor type -------------
def test_to_spec_roundtrips_thermo_and_reactor():
    from discopt.mkm.spec import from_spec, to_spec

    m = mk.Model("rt", T=450.0, R=8.314, Tref=298.15)
    s = m.site("cat", density=1.0)
    A = m.gas("A", thermo=_nasa7_const_cp(-5000.0, 10.0))
    B = m.gas("B", H=-1.0, S=0.0)
    As = m.adsorbate("A*", site=s, H=-0.3, S=0.0)
    m.step(A + s >> As, A=1e3, Ea=0.0)
    m.step(As >> B + s, A=1e3, Ea=0.3)
    reactor = mk.CSTR(inlet={A: 1.0, B: 0.0}, tau=2.0, cat_density=0.5)

    spec = to_spec(m, reactor)
    assert spec["reactor"]["type"] == "cstr"
    assert spec["gas"][0]["thermo"]["type"] == "nasa7"

    m2, r2 = from_spec(spec)
    from discopt.mkm.reactors import CSTR
    from discopt.mkm.thermo_models import NASA7

    assert isinstance(r2, CSTR) and r2.tau == 2.0
    assert isinstance(m2._by_name["A"].thermo, NASA7)


# --- M12: the non-isothermal steady solve starts at a physical temperature ---
@pytest.mark.slow
def test_nonisothermal_solve_temperature_is_physical():
    # With R = 8.314 and Ea = 60 kJ/mol the Arrhenius factor underflows at the
    # ~10 K start discopt's clip would otherwise pick; the scaled T variable starts
    # near the inlet temperature so the solve lands on a physical steady state.
    from discopt.mkm.examples import adiabatic_cstr

    m, reactor, energy = adiabatic_cstr(T_in=500.0)
    sol = mk.solve_steady_state(m, reactor, energy=energy)
    assert sol.status == "optimal"
    T = sol.temperature()
    assert 500.0 < T < 2000.0  # rose above the feed, stayed physical (not ~10 K)


# ==========================================================================
# Completeness-gap fixes (REVIEW.md C1-C12; C5 is a documented limitation).
# ==========================================================================

_HER_SPEC = {
    "name": "her", "T": 298.0, "R": R, "U": -0.30, "F": F,
    "sites": [{"name": "*", "density": 1.0}],
    "gas": [{"name": "Hp", "H": 0.0, "S": 0.0, "composition": {"H": 1}},
            {"name": "H2", "H": 0.0, "S": 0.0, "composition": {"H": 2}}],
    "adsorbates": [{"name": "H*", "site": "*", "H": -0.10, "S": 0.0, "composition": {"H": 1}}],
    "reactions": [
        {"equation": "Hp + * -> H*", "A": 1e2, "Ea": 0.30, "n_electrons": 1, "beta": 0.5,
         "irreversible": True},
        {"equation": "2 H* -> H2 + 2 *", "A": 1e10, "Ea": 0.20}],
    "reactor": {"type": "differential", "pressures": {"Hp": 1.0, "H2": 1e-3}},
}


def _orr_spec():
    from discopt.mkm.electrochem import orr_4e
    from discopt.mkm.spec import to_spec

    m, reactor = orr_4e(descriptor=0.9)
    return to_spec(m, reactor)


# --- C1: electrochemistry is reachable from the agent / MCP surface ----------
@pytest.mark.slow
def test_agent_current_returns_finite_number():
    from discopt.mkm import agent

    out = agent.current(_HER_SPEC)
    assert np.isfinite(out["current"]) and out["current"] > 0.0
    assert out["U"] == pytest.approx(-0.30)
    assert out["status"] == "optimal"


def test_agent_che_diagram_structure_and_limiting_potential():
    from discopt.mkm import agent

    che = agent.che_diagram(_orr_spec())
    # 4 faradaic steps -> 4 labels/ΔG and an (n+1)-point cumulative profile from 0
    assert len(che["steps"]) == 4
    assert len(che["delta_g"]) == 4
    assert len(che["cumulative"]) == 5
    assert che["cumulative"][0] == 0.0
    assert all(isinstance(x, float) for x in che["delta_g"])
    import json

    json.dumps(che)  # JSON-serializable
    assert agent.limiting_potential(_orr_spec())["limiting_potential"] == pytest.approx(0.82, abs=1e-6)


def test_agent_electrochem_requires_faradaic_steps():
    from discopt.mkm import agent

    from discopt.mkm.examples import co_oxidation
    from discopt.mkm.spec import to_spec

    m, _ = co_oxidation(500.0)
    spec = to_spec(m, mk.DifferentialReactor({m._by_name["CO"]: 1.0, m._by_name["O2"]: 0.5}))
    with pytest.raises(ValueError, match="electrochemical"):
        agent.current(spec)


def test_mcp_registers_electrochemistry_tools():
    import asyncio

    from discopt.mkm import mcp_server

    tools = asyncio.new_event_loop().run_until_complete(mcp_server.mcp.list_tools())
    names = {t.name for t in tools}
    assert {"current", "tafel_slope", "che_diagram", "limiting_potential"} <= names


# --- C2: MCP tools mirror the agent-function arguments ------------------------
def test_mcp_tools_expose_method_and_coordinates():
    import inspect

    from discopt.mkm import agent, mcp_server

    # agent functions have the params the tools must forward
    assert "method" in inspect.signature(agent.solve).parameters
    assert "coordinates" in inspect.signature(agent.report).parameters
    # and the MCP tools now expose them too
    assert "method" in inspect.signature(mcp_server.solve).parameters
    assert "coordinates" in inspect.signature(mcp_server.report).parameters


# --- C3: analyze records why apparent kinetics were skipped (CSTR) -----------
@pytest.mark.slow
def test_analyze_cstr_records_apparent_kinetics_note():
    from discopt.mkm import agent

    spec = {
        "name": "cstr", "T": 500.0, "R": R,
        "sites": [{"name": "*", "density": 1.0}],
        "gas": [{"name": "A", "H": 0.0, "S": 0.001, "composition": {"A": 1}},
                {"name": "B", "H": -1.0, "S": 0.001, "composition": {"B": 1}}],
        "adsorbates": [{"name": "A*", "site": "*", "H": -0.3, "S": 0.001, "composition": {"A": 1}}],
        "reactions": [{"equation": "A + * <=> A*", "A": 1e5, "Ea": 0.0},
                      {"equation": "A* -> B + *", "A": 1e5, "Ea": 0.4, "irreversible": True}],
        "reactor": {"type": "cstr", "inlet": {"A": 1.0, "B": 0.0}, "tau": 1.0},
    }
    out = agent.analyze(spec, target="B")
    # apparent kinetics are undefined for a CSTR; a note explains it rather than
    # the keys silently vanishing.
    assert "apparent_orders" not in out
    assert "apparent_kinetics_note" in out and out["apparent_kinetics_note"]


# --- C4: alpha with no interactions warns (it would otherwise be inert) -------
@pytest.mark.slow
def test_alpha_without_interactions_warns():
    m = mk.Model("bep", T=500.0, R=R, Tref=298.15)
    s = m.site("*", density=1.0)
    A = m.gas("A", H=0.0, S=0.001)
    As = m.adsorbate("A*", site=s, H=-0.3, S=0.001)
    m.step(A + s >> As, A=1e5, Ea=0.2, alpha=0.5)
    with pytest.warns(UserWarning, match="alpha"):
        mk.solve_steady_state(m, mk.DifferentialReactor({A: 1.0}))


# --- C6: fit warm-start knobs are wired; log lb<=0 is a clear error -----------
def _explicit_fit_setup():
    def model(temp=500.0):
        m = mk.Model("c6", T=temp, R=R, Tref=298.15)
        s = m.site("s", density=1.0)
        A = m.gas("A", H=0.0, S=0.0)
        B = m.gas("B", H=-1.0, S=0.0)
        As = m.adsorbate("A*", site=s, H=-0.3, S=0.0)
        m.step(A + s >> As, kf=5.0, Keq=10.0, name="ads")
        surf = m.step(As >> B + s, A=1e3, Ea=0.3, name="surf")
        return m, A, B, As, surf

    TS = [480.0, 500.0, 520.0]
    data = []
    for temp in TS:
        mt, At, Bt, _, _ = model(temp)
        data.append(mk.solve_steady_state(mt, mk.DifferentialReactor({At: 1.0, Bt: 0.0})).production_rate(Bt))
    m, A, B, As, surf = model(500.0)
    obs = [mk.Observation(response=B, value=v, T=temp, pressures={A: 1.0, B: 0.0}, sigma=0.01,
                          theta0={As: 0.2}) for temp, v in zip(TS, data)]
    return m, A, B, As, surf, obs


@pytest.mark.slow
def test_fit_theta0_warm_start_is_wired():
    # Observation.theta0 warm-starts each condition's coverage variables (adsorbate
    # + free-site, completed from the site balance) via the estimation solve's
    # initial_solution — previously it was read by nothing.
    from discopt.mkm.estimate import _MKMExperiment

    m, A, B, As, surf, obs = _explicit_fit_setup()  # each obs carries theta0={As: 0.2}
    fit = [mk.FitParam(surf, "A", lb=1e2, ub=1e4, init=5e2),
           mk.FitParam(surf, "Ea", lb=0.05, ub=0.6, init=0.2)]
    exp = _MKMExperiment(m, obs, fit)
    exp.create_model()
    # one warm-started coverage var + one free-site var per observation
    assert len(exp._initial_solution) == len(obs) * 2
    assert init_val_for(exp, f"th0_{As.name.replace('*', '_')}") == pytest.approx(0.2)
    res = mk.fit_kinetics(m, obs, fit)
    Ea_key = next(k for k in res.parameters if k.startswith("Ea_"))
    assert res.parameters[Ea_key] == pytest.approx(0.3, abs=2e-2)


def init_val_for(exp, varname):
    for v, val in exp._initial_solution.items():
        if v.name == varname:
            return val
    raise KeyError(varname)


def test_fit_log_param_nonpositive_lb_raises():
    m, A, B, As, surf, obs = _explicit_fit_setup()
    fit = [mk.FitParam(surf, "A", lb=0.0, ub=1e4)]  # A is fit in log space by default
    with pytest.raises(ValueError, match="log"):
        mk.fit_kinetics(m, obs, fit)


# --- C7: limiting potential / volcano raise for an oxidation mechanism --------
def test_limiting_potential_rejects_oxidation():
    from discopt.mkm.electrochem import limiting_potential, optimize_descriptor

    m = mk.Model("ox", T=298.0, R=R, U=0.0, F=F)
    s = m.site("*", density=1.0)
    X = m.gas("X", H=0.0, S=0.0)
    Xs = m.adsorbate("X*", site=s, H=-0.2, S=0.0)
    m.step(Xs >> X + s, A=1e4, Ea=0.2, n_electrons=-1, beta=0.5)  # oxidation, n<0
    with pytest.raises(ValueError, match="oxidation"):
        limiting_potential(m)
    with pytest.raises(ValueError, match="oxidation"):
        optimize_descriptor([(-1.0, 1.0)], [-1], bounds=(0.0, 1.0))


# --- C8: the electrochem package documents its two sign conventions ----------
def test_electrochem_documents_sign_conventions():
    import discopt.mkm.electrochem as ec

    doc = ec.__doc__.lower()
    assert "sign convention" in doc
    assert "reduction is positive" in doc and "cathodic" in doc


# --- C9: HTML render shows n_electrons / beta and the potential U ------------
def test_mechanism_html_shows_electrochemistry():
    from discopt.mkm.electrochem import orr_4e

    m, _ = orr_4e(descriptor=0.9)
    html = m.to_html()
    assert "n<sub>e</sub>" in html      # electron count on the faradaic steps
    assert "&beta;" in html             # transfer coefficient
    assert f"U={m.U:g}" in html         # potential in the title


# --- C10: half-integer routes are reported as fractions, not int-rounded -----
def test_route_stoichiometric_numbers_not_int_rounded():
    from discopt.mkm import agent

    # X* is produced 3-per-turnover by step 1 and consumed 2-per-turnover by step 2,
    # so the route stoichiometric numbers are in ratio 2:3 -> rationalized [1, 1.5].
    spec = {
        "name": "half", "T": 500.0, "R": R,
        "sites": [{"name": "*", "density": 1.0}],
        "gas": [{"name": "A", "composition": {"A": 1}}, {"name": "B", "composition": {"B": 1}}],
        "adsorbates": [{"name": "X*", "site": "*", "composition": {"A": 1}}],
        "reactions": [{"equation": "A + 3 * -> 3 X*", "A": 1e4, "Ea": 0.0},
                      {"equation": "2 X* -> B + 2 *", "A": 1e4, "Ea": 0.0}],
        "infer_composition": False,
    }
    routes = agent.structure(spec)["routes"]
    assert len(routes) == 1
    sigma = routes[0]["stoichiometric_numbers"]
    assert sigma[0] == pytest.approx(1.0) and sigma[1] == pytest.approx(1.5)
    assert sigma != [1, 2]  # int(round) would have corrupted the half-integer route


# --- C11: a large CSTR feed is not clipped by a hard gas upper bound ----------
@pytest.mark.slow
def test_cstr_large_inlet_not_clipped():
    m = mk.Model("big", T=500.0, R=R, Tref=298.15)
    s = m.site("*", density=1.0)
    Inert = m.gas("Inert", H=0.0, S=0.0)          # inert: steady C == inlet
    A = m.gas("A", H=0.0, S=0.001)
    As = m.adsorbate("A*", site=s, H=-0.3, S=0.001)
    m.step(A + s >> As, A=1.0, Ea=0.0)
    # inlet 2e6 exceeds the old hard-coded ub=1e6; the adaptive bound must not clip it.
    sol = mk.solve_steady_state(m, mk.CSTR(inlet={Inert: 2e6, A: 1.0}, tau=1.0, cat_density=1e-6))
    assert sol.gas_concentration(Inert) == pytest.approx(2e6, rel=1e-6)


# --- C12: numeric callable-H path mirrors the symbolic contract (raises) ------
def test_numeric_callable_H_without_theta_raises():
    m = mk.Model("cbH", T=500.0, R=R)
    s = m.site("*", density=1.0)
    A = m.gas("A", H=0.0, S=0.0)
    m.adsorbate("A*", site=s, H=lambda theta: -0.3, S=0.0)  # coverage-dependent H
    m.step(A + s >> m._by_name["A*"], A=1e5, Ea=0.0)
    with pytest.raises(ValueError, match="coverage-dependent H"):
        numeric.rate_constants(m, 500.0)  # theta=None


# --- Round 4: degree of rate control flags an ill-conditioned (non-unit) sum ---
def _saturated_coox():
    """CO-oxidation conditions (from the round-4 differential fuzz) that saturate
    the surface with CO*, where the default-active_tol DRC is ill-conditioned."""
    from discopt.mkm.model import MicrokineticModel

    m = MicrokineticModel("sat", T=401.5, R=R, Tref=298.15)
    s = m.site("*", density=1.0)
    CO = m.gas("CO", H=0.0, S=0.002, composition={"C": 1, "O": 1})
    O2 = m.gas("O2", H=0.0, S=0.002, composition={"O": 2})
    CO2 = m.gas("CO2", H=-3.0, S=0.002, composition={"C": 1, "O": 2})
    m.adsorbate("CO*", site=s, H=-1.190, S=0.0005, composition={"C": 1, "O": 1})
    m.adsorbate("O*", site=s, H=-0.168, S=0.0005, composition={"O": 1})
    m.step(CO + s >> m._by_name["CO*"], A=21119.66, Ea=0.0, name="CO ads")
    m.step(O2 + 2 * s >> 2 * m._by_name["O*"], A=9166.76, Ea=0.0, name="O2 diss")
    m.step(m._by_name["CO*"] + m._by_name["O*"] >> CO2 + 2 * s,
           A=3.0528e8, Ea=0.5114, name="surf")
    m.infer_composition()
    return m, mk.DifferentialReactor({CO: 1.1191, O2: 1.4480, CO2: 0.0}), CO2


@pytest.mark.slow
def test_drc_flags_ill_conditioned_sum():
    from discopt.mkm.analysis import degree_of_rate_control
    from discopt.mkm.analysis.drc import SensitivityUnavailable

    m, reactor, CO2 = _saturated_coox()
    sol = mk.solve_steady_state(m, reactor)  # default active_tol -> saturated/ill-conditioned
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            X = degree_of_rate_control(sol, species=CO2)
        except SensitivityUnavailable:
            return  # acceptable: flagged by raising instead of returning garbage
        # if it returned, it must NOT silently hand back a sum far from 1
        assert any("sums to" in str(rec.message) for rec in w), (
            f"saturated DRC returned sum={sum(X.values()):.3f} with no warning"
        )


@pytest.mark.slow
def test_drc_wellconditioned_does_not_warn():
    from discopt.mkm.analysis import degree_of_rate_control

    m, reactor = co_oxidation(500.0)
    sol = mk.solve_steady_state(m, reactor)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        X = degree_of_rate_control(sol, species=m._by_name["CO2"])
    assert abs(sum(X.values()) - 1.0) < 1e-2
    assert not any("sums to" in str(rec.message) for rec in w)
