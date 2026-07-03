"""Temperature-dependent thermo models: NASA7, Shomate, general callable."""

import numpy as np
import pytest

import discopt.mkm as mk

# Whole file: NASA7/Shomate thermo models each drive a full solve (slow).
pytestmark = pytest.mark.slow


def _adsorption(thermo=None, H=0.0, S=0.0, Cp=0.0, T=400.0):
    """A + * <=> A* with A* carrying the given thermo; returns solved coverage."""
    m = mk.Model("t", T=T, R=8.314, Tref=298.15)
    s = m.site("s", density=1.0)
    A = m.gas("A")
    As = m.adsorbate("A*", site=s, H=H, S=S, Cp=Cp, thermo=thermo)
    m.step(A + s >> As, A=1e3, Ea=0.0)  # reversible; reverse from thermo
    sol = mk.solve_steady_state(m, mk.DifferentialReactor({A: 1.0}))
    return sol, As


def test_nasa7_reduces_to_constant_cp():
    """A NASA-7 polynomial encoding a constant Cp matches the constant-Cp model."""
    H, S, Cp, R, Tref = -5000.0, 10.0, 30.0, 8.314, 298.15
    # constant-Cp species: H(T)=H+Cp(T-Tref), S(T)=S+Cp ln(T/Tref)
    sol_const, As_c = _adsorption(H=H, S=S, Cp=Cp)
    theta_const = sol_const.coverage(As_c)

    # equivalent NASA-7: a1=Cp/R, a6=(H-Cp*Tref)/R, a7=(S-Cp*ln Tref)/R, rest 0
    a1 = Cp / R
    a6 = (H - Cp * Tref) / R
    a7 = (S - Cp * np.log(Tref)) / R
    coeffs = [a1, 0.0, 0.0, 0.0, 0.0, a6, a7]
    sol_nasa, As_n = _adsorption(thermo=mk.NASA7(coeffs))
    theta_nasa = sol_nasa.coverage(As_n)

    assert theta_nasa == pytest.approx(theta_const, rel=1e-6)


def test_general_thermo_matches_constant():
    """A GeneralThermo with explicit H(T), S(T) reproduces the constant-Cp model."""
    H, S, Cp, Tref = -5000.0, 10.0, 30.0, 298.15
    sol_const, As_c = _adsorption(H=H, S=S, Cp=Cp)

    gen = mk.GeneralThermo(
        h=lambda T, log: H + Cp * (T - Tref),
        s=lambda T, log: S + Cp * log(T / Tref),
    )
    sol_gen, As_g = _adsorption(thermo=gen)
    assert sol_gen.coverage(As_g) == pytest.approx(sol_const.coverage(As_c), rel=1e-6)


def test_shomate_solves_and_is_temperature_dependent():
    """A Shomate species solves, is thermodynamically consistent, and depends on T."""
    # CO gas Shomate coefficients (NIST, 298-1300 K), illustrative use
    co = mk.Shomate(25.56759, 6.096130, 4.054656, -2.671201, 0.131021,
                    -118.0089, 227.3665, -110.5271)

    def tof(T):
        m = mk.Model("sh", T=T, R=8.314, Tref=298.15)
        s = m.site("s", density=1.0)
        A = m.gas("A", thermo=co)
        As = m.adsorbate("A*", site=s, H=-30000.0, S=-50.0)
        m.step(A + s >> As, A=1e3, Ea=0.0)
        sol = mk.solve_steady_state(m, mk.DifferentialReactor({A: 1.0}))
        # thermodynamic consistency holds with a thermo model in the loop
        from discopt.mkm.analysis.sensitivity import evaluate_expression
        from discopt.mkm.kinetics import k_forward, k_reverse
        r = m.reactions[0]
        kf = evaluate_expression(k_forward(r, sol.T_param, m.R), sol.result, sol.dm_model)
        kr = evaluate_expression(k_reverse(r, sol.T_param, m.R, m.Tref), sol.result, sol.dm_model)
        from discopt.mkm.thermo import K_eq
        keq = evaluate_expression(K_eq(r, sol.T_param, m.R, m.Tref), sol.result, sol.dm_model)
        assert kf / kr == pytest.approx(keq, rel=1e-8)
        return sol.coverage(As)

    assert tof(400.0) != pytest.approx(tof(600.0), rel=1e-3)
