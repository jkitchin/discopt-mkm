"""Optional temperature-dependent per-species thermodynamic models.

A thermo model emits the standard molar enthalpy ``H(T)`` and entropy ``S(T)``
(and hence ``G(T) = H - T S``) as functions of the temperature *expression*.
Because temperature is a discopt expression (a parameter or, in a non-isothermal
solve, a variable), the polynomials compile and differentiate exactly like the
rest of the model.

The methods take a ``log`` callable so the same code works for the discopt
expression backend (``log = dm.log``) and the pure-NumPy backend used for warm
starts (``log = np.log``); all other operations (``+ - * / **``) are shared.

Units are the caller's responsibility and must match the model's gas constant
``R``: NASA-7 coefficients are dimensionless (scaled by ``R``), so use SI
(``R = 8.314``); Shomate uses the NIST convention (``Cp`` in J/mol/K, ``H`` in
kJ/mol — returned here converted to J/mol — entropy in J/mol/K).
"""

from __future__ import annotations

from typing import Callable


class ThermoModel:
    """Base class: subclasses implement ``H`` and ``S``; ``g`` follows."""

    def select(self, T_nominal: float) -> None:
        """Hook for piecewise models to pick a range from the nominal temperature."""

    def H(self, T, R: float, log: Callable):  # noqa: N802
        raise NotImplementedError

    def S(self, T, R: float, log: Callable):  # noqa: N802
        raise NotImplementedError

    def Cp(self, T, R: float):  # noqa: N802
        """Molar heat capacity ``Cp(T)`` (same energy units as ``H``). Used by the
        non-isothermal energy balance for thermo-carrying species."""
        raise NotImplementedError

    def g(self, T, R: float, Tref: float, log: Callable):
        return self.H(T, R, log) - T * self.S(T, R, log)


class NASA7(ThermoModel):
    """NASA 7-coefficient polynomial thermo (two temperature ranges).

    ``Cp/R = a1 + a2 T + a3 T^2 + a4 T^3 + a5 T^4``;
    ``H/(R T) = a1 + a2 T/2 + a3 T^2/3 + a4 T^3/4 + a5 T^4/5 + a6/T``;
    ``S/R = a1 ln T + a2 T + a3 T^2/2 + a4 T^3/3 + a5 T^4/4 + a7``.

    ``low``/``high`` are the 7 coefficients below/above ``Tmid``. The range is
    selected once from the model's nominal temperature; for a non-isothermal
    solve whose temperature crosses ``Tmid`` this single-range choice is an
    approximation.
    """

    def __init__(self, low, high=None, Tmid: float = 1000.0):
        self.low = [float(c) for c in low]
        self.high = [float(c) for c in (high if high is not None else low)]
        self.Tmid = float(Tmid)
        self._a = self.low

    def select(self, T_nominal: float) -> None:
        self._a = self.high if float(T_nominal) >= self.Tmid else self.low

    def H(self, T, R, log=None):  # noqa: N802
        a = self._a
        return R * (a[0] * T + a[1] * T**2 / 2 + a[2] * T**3 / 3 + a[3] * T**4 / 4 + a[4] * T**5 / 5 + a[5])

    def S(self, T, R, log):  # noqa: N802
        a = self._a
        return R * (a[0] * log(T) + a[1] * T + a[2] * T**2 / 2 + a[3] * T**3 / 3 + a[4] * T**4 / 4 + a[6])

    def Cp(self, T, R):  # noqa: N802
        a = self._a
        return R * (a[0] + a[1] * T + a[2] * T**2 + a[3] * T**3 + a[4] * T**4)


class Shomate(ThermoModel):
    """NIST Shomate equation thermo.

    With ``t = T / 1000``:
    ``Cp = A + B t + C t^2 + D t^3 + E/t^2`` (J/mol/K);
    ``H(T) - H(298.15) = A t + B t^2/2 + C t^3/3 + D t^4/4 - E/t + F - H`` (kJ/mol,
    returned here as J/mol); ``S = A ln t + B t + C t^2/2 + D t^3/3 - E/(2 t^2) + G``
    (J/mol/K). Use ``R = 8.314``.
    """

    def __init__(self, A, B, C, D, E, F, G, H, Tscale: float = 1000.0):
        self.coef = (float(A), float(B), float(C), float(D), float(E), float(F), float(G), float(H))
        self.Tscale = float(Tscale)

    def H(self, T, R, log=None):  # noqa: N802
        A, B, C, D, E, F, G, H = self.coef
        t = T / self.Tscale
        return 1000.0 * (A * t + B * t**2 / 2 + C * t**3 / 3 + D * t**4 / 4 - E / t + F - H)

    def S(self, T, R, log):  # noqa: N802
        A, B, C, D, E, F, G, H = self.coef
        t = T / self.Tscale
        return A * log(t) + B * t + C * t**2 / 2 + D * t**3 / 3 - E / (2 * t**2) + G

    def Cp(self, T, R):  # noqa: N802
        A, B, C, D, E, F, G, H = self.coef
        t = T / self.Tscale
        return A + B * t + C * t**2 + D * t**3 + E / t**2


class GeneralThermo(ThermoModel):
    """Arbitrary ``H(T)`` / ``S(T)`` from user callables.

    Each callable takes ``(T, log)`` and returns an expression, e.g.::

        GeneralThermo(h=lambda T, log: H0 + Cp*(T - 298.15),
                      s=lambda T, log: S0 + Cp*log(T/298.15))
    """

    def __init__(self, h: Callable, s: Callable):
        self.h_fn = h
        self.s_fn = s

    def H(self, T, R, log):  # noqa: N802
        return self.h_fn(T, log)

    def S(self, T, R, log):  # noqa: N802
        return self.s_fn(T, log)
