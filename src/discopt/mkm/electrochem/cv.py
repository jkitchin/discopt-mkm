"""Cyclic voltammetry of a soluble redox couple (the classic "duck").

A linear-sweep / cyclic voltammogram of an outer-sphere couple ``O + n e- <=> R``
is governed by semi-infinite diffusion of O and R to a planar electrode with a
Butler-Volmer (or, in the fast limit, Nernst) surface boundary condition. With
equal diffusivities and ``C_O + C_R`` conserved, the two-species problem collapses
to a single diffusing species with a Robin boundary condition, which we integrate
by an unconditionally stable implicit (backward-Euler) finite-difference scheme.

This is a self-contained transient simulation (unlike the rest of the package's
steady-state solves); it shares the Butler-Volmer rate-constant form so the same
``k0``, ``alpha``, ``n`` mean the same thing as elsewhere. The current rises,
peaks as the surface depletes, and decays with a Cottrell tail, giving the
duck-shaped trace; on the return sweep the accumulated product is reconverted.

Default constants are SI-ish (``F = 96485`` C/mol, ``R = 8.314`` J/mol/K) so a
reversible one-electron couple shows the textbook ~59 mV peak separation and the
Randles-Sevcik ``i_p ∝ sqrt(scan_rate)`` scaling.
"""

from __future__ import annotations

import numpy as np


def _thomas(a, b, c, d):
    """Solve a tridiagonal system (sub ``a``, diag ``b``, super ``c``, rhs ``d``)."""
    n = len(d)
    cp = np.empty(n); dp = np.empty(n)
    cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def cyclic_voltammogram(U0=0.0, k0=1.0, alpha=0.5, n_electrons=1,
                        U_start=0.3, U_vertex=-0.3, scan_rate=0.05,
                        C_bulk=1.0, D=1e-5, area=1.0,
                        T=298.0, R=8.314, F=96485.0,
                        nx=240, nt_per_sweep=2000):
    """Simulate a cyclic voltammogram of ``O + n e- <=> R``.

    Triangular sweep from ``U_start`` down to ``U_vertex`` and back at
    ``scan_rate`` (V/s). ``k0`` is the standard heterogeneous rate constant (large
    ``k0`` -> reversible/Nernstian; small -> irreversible), ``alpha`` the transfer
    coefficient, ``U0`` the formal potential. ``D`` is the (shared) diffusivity,
    ``C_bulk`` the bulk concentration of O (R starts at 0).

    Returns ``(U, i)``: the potential and current arrays over the full cycle.
    Cathodic (reduction) current is negative.
    """
    f = n_electrons * F / (R * T)

    # triangular potential program
    span = abs(U_start - U_vertex)
    nt = int(nt_per_sweep)
    t_half = span / scan_rate
    dt = t_half / nt
    down = np.linspace(U_start, U_vertex, nt + 1)
    up = np.linspace(U_vertex, U_start, nt + 1)[1:]
    U = np.concatenate([down, up])

    # spatial grid: semi-infinite, far boundary beyond the diffusion length
    L = 6.0 * np.sqrt(D * 2.0 * t_half)
    dx = L / (nx - 1)
    lam = D * dt / dx ** 2

    C = np.full(nx, C_bulk)          # C_O profile; C_R = C_bulk - C_O
    DoverDx = D / dx
    current = np.empty(U.size)

    # backward-Euler tridiagonal coefficients (interior rows are time-invariant)
    a = np.full(nx, -lam); b = np.full(nx, 1.0 + 2.0 * lam); c = np.full(nx, -lam)
    b[-1] = 1.0; a[-1] = 0.0; c[-1] = 0.0          # bulk Dirichlet C = C_bulk

    for k, Uk in enumerate(U):
        kf = k0 * np.exp(-alpha * f * (Uk - U0))    # reduction (cathodic)
        kb = k0 * np.exp((1.0 - alpha) * f * (Uk - U0))  # oxidation (anodic)
        d = C.copy()
        # surface Robin BC (algebraic flux balance, no accumulation):
        #   D (C_1 - C_0)/dx = (kf+kb) C_0 - kb C_bulk
        b[0] = -(DoverDx + kf + kb); c[0] = DoverDx; a[0] = 0.0
        d[0] = -kb * C_bulk
        d[-1] = C_bulk
        C = _thomas(a, b, c, d)
        J = (kf + kb) * C[0] - kb * C_bulk          # net reduction flux at surface
        current[k] = -n_electrons * F * area * J     # cathodic negative
    return U, current
