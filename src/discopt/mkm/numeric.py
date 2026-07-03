"""Pure-NumPy rate evaluation and steady-state root-find.

Independent of discopt. Used to (a) provide a warm start for the stiff
log-coverage steady-state solve, and (b) validate the discopt solution. Mirrors
the rate laws in :mod:`discopt.mkm.kinetics` exactly, but with plain floats.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import fsolve

from discopt.mkm.model import MicrokineticModel
from discopt.mkm.species import Adsorbate, GasSpecies, Site

_EQUILIBRATED_MSG = (
    "mechanism has equilibrated (quasi-equilibrium) steps; the numeric evaluator "
    "cannot model them"
)


def _reject_equilibrated(mkm):
    """Guard the numeric paths against ``equilibrated`` steps.

    ``rate_constants`` sets ``kf = kr = 0`` for an equilibrated step and imposes
    no equilibrium relation, so a numeric solve/integration would silently be of
    a *different* mechanism (those steps deleted). Solving the quasi-equilibrium
    system requires the extent/equilibrium reformulation in the discopt
    linear-coordinate path, not this evaluator.
    """
    if any(r.equilibrated for r in mkm.reactions):
        raise ValueError(_EQUILIBRATED_MSG)


def _interaction_map(mkm):
    """Numeric lateral-interaction map ``species -> [(partner, eps), ...]``."""
    inter = {}
    for a, b, eps in getattr(mkm, "_interactions", []):
        inter.setdefault(a, []).append((b, eps))
        if a is not b:
            inter.setdefault(b, []).append((a, eps))
    return inter


def rate_constants(mkm: MicrokineticModel, T: float, theta: dict | None = None):
    """Return ``(kf, kr)`` dicts ``{reaction: value}`` at temperature ``T``.

    With lateral interactions the constants depend on the coverages, so pass the
    current ``theta`` (otherwise interactions are ignored).
    """
    R, Tref = mkm.R, mkm.Tref
    inter = _interaction_map(mkm)

    def dH(sp):
        if theta is None:
            return 0.0
        return sum(eps * float(theta.get(p, 0.0)) for p, eps in inter.get(sp, []))

    def baseH(sp):
        if callable(sp.H):
            return float(sp.H(theta)) if theta is not None else 0.0
        return sp.H

    kf, kr = {}, {}
    for rxn in mkm.reactions:
        if rxn.equilibrated:
            # quasi-equilibrated steps have no rate law; this numeric evaluator
            # uses full kinetics, so it does not model their (extent) flux.
            kf[rxn] = kr[rxn] = 0.0
            continue
        if rxn.explicit_rate:
            kf_j = rxn.kf
        else:
            Ea = rxn.Ea
            if inter and theta is not None and rxn.alpha:
                Ea = Ea + rxn.alpha * sum(nu * dH(s) for s, nu in rxn.net_stoich().items())
            kf_j = rxn.A * np.exp(-Ea / (R * T))
        # electrochemical Butler-Volmer factor on the forward rate constant
        ec = rxn.n_electrons * mkm.F * mkm.U if getattr(rxn, "is_electrochemical", False) else 0.0
        if ec:
            kf_j = kf_j * np.exp(-rxn.beta * ec / (R * T))
        if rxn.irreversible:
            kr_j = 0.0
        elif rxn.explicit_rate:
            kr_j = kf_j / (rxn.Keq * np.exp(-ec / (R * T)))
        else:
            dG = ec
            for s, nu in rxn.net_stoich().items():
                if getattr(s, "thermo", None) is not None:
                    s.thermo.select(T)
                    G = s.thermo.g(T, R, Tref, np.log) + dH(s)
                else:
                    H0 = baseH(s) + dH(s)
                    G = H0 - T * s.S
                    if s.Cp != 0.0:
                        G = (H0 + s.Cp * (T - Tref)) - T * (s.S + s.Cp * np.log(T / Tref))
                dG += nu * G
            kr_j = kf_j / np.exp(-dG / (R * T))
        kf[rxn] = kf_j
        kr[rxn] = kr_j
    return kf, kr


def _activity(sp, theta, free, conc):
    if isinstance(sp, GasSpecies):
        return conc[sp]
    if isinstance(sp, Adsorbate):
        return theta[sp]
    if isinstance(sp, Site):
        return free[sp]
    raise TypeError(f"no activity for {sp!r}")


def _prod(stoich, theta, free, conc):
    out = 1.0
    for s, c in stoich.items():
        out = out * _activity(s, theta, free, conc) ** c
    return out


def rates_of_progress(mkm, kf, kr, theta, free, conc) -> dict:
    return {
        rxn: kf[rxn] * _prod(rxn.reactants, theta, free, conc)
        - kr[rxn] * _prod(rxn.products, theta, free, conc)
        for rxn in mkm.reactions
    }


def net_rate(mkm, sp, rops) -> float:
    return sum(rxn.net_stoich().get(sp, 0.0) * rops[rxn] for rxn in mkm.reactions)


def steady_state_numeric(
    mkm: MicrokineticModel, pressures: dict, T: float, theta0: dict | None = None, xtol: float = 1e-13
):
    """Solve the steady-state coverages numerically with ``scipy.fsolve``.

    Free-site coverage on each site is determined by the site balance. Returns
    ``(theta, free)`` dicts. Provide ``theta0`` (adsorbate -> coverage) to seed
    the physical root for stiff/poisoning-prone mechanisms.

    This is a *warm-start helper*: it seeds the differentiable log-coordinate
    solve and validates it, so it tolerates a slightly imperfect root. It raises
    :class:`RuntimeError` only when ``fsolve`` fails to converge *and* the
    residual is large relative to the one-way fluxes (suggesting the seed landed
    in a bad basin — try a different ``theta0``), or when a coverage comes back
    meaningfully negative.
    """
    _reject_equilibrated(mkm)
    ads = mkm.adsorbates
    conc = {g: float(pressures.get(g, 0.0)) for g in mkm.gas_species}

    def unpack(x):
        theta = {a: x[i] for i, a in enumerate(ads)}
        free = {s: 1.0 - sum(theta[a] for a in mkm.adsorbates_on(s)) for s in mkm.sites}
        return theta, free

    def resid(x):
        theta, free = unpack(x)
        kf, kr = rate_constants(mkm, T, theta)  # coverage-dependent when interactions on
        rops = rates_of_progress(mkm, kf, kr, theta, free, conc)
        return [net_rate(mkm, a, rops) for a in ads]

    n = len(ads)
    if theta0 is None:
        x0 = np.full(n, 0.5 / max(n, 1))
    else:
        x0 = np.array([float(theta0.get(a, 1e-3)) for a in ads])
    # a warm start only needs to be approximately right; ignore fsolve's
    # slow-progress warning on stiff (near-equilibrium) systems.
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, infodict, ier, msg = fsolve(resid, x0, xtol=xtol, full_output=True)

    # verify the solve instead of trusting a returned iterate. The steady-state
    # residuals are *net* rates; scale the tolerance by the largest one-way flux
    # so a genuine near-equilibrium cancellation (net ~ 1e-6 of a ~1e6 flux) is
    # not flagged, but a non-converged basin is.
    theta, free = unpack(x)
    kf, kr = rate_constants(mkm, T, theta)
    fluxes = [abs(kf[r] * _prod(r.reactants, theta, free, conc)) for r in mkm.reactions]
    fluxes += [abs(kr[r] * _prod(r.products, theta, free, conc)) for r in mkm.reactions]
    scale = max(fluxes + [1e-30])
    resnorm = float(np.max(np.abs(np.asarray(resid(x), dtype=float)))) if n else 0.0
    if ier != 1 and resnorm > 1e-6 * scale:
        raise RuntimeError(
            f"steady_state_numeric (warm-start helper) did not converge: fsolve "
            f"ier={ier} ({str(msg).strip()}); residual {resnorm:.3e} vs one-way flux "
            f"scale {scale:.3e}. Try a different theta0 seed."
        )

    # coverages must be physical: a *meaningfully* negative coverage means a
    # spurious root. Tolerance is loose (1e-4) because this is only a warm start
    # for stiff systems where fsolve returns ~1e-6 negative noise on genuinely
    # near-zero coverages (e.g. water-gas shift); a real poisoned root is O(0.1)
    # negative. Smaller negatives are clamped to 0.
    negatives = [a.name for a in ads if theta[a] < -1e-4]
    if negatives:
        raise RuntimeError(
            f"steady_state_numeric (warm-start helper) returned negative coverage(s) "
            f"{negatives}; the seed landed on a non-physical root. Try a different theta0."
        )
    x = np.clip(np.asarray(x, dtype=float), 0.0, None)  # clamp tiny negatives to 0
    return unpack(x)


def species_free_energy(mkm, sp, T, theta=None) -> float:
    """Numeric standard free energy ``G(T)`` of a species (thermo model or constant)."""
    R, Tref = mkm.R, mkm.Tref
    inter = _interaction_map(mkm)
    dH = sum(eps * float((theta or {}).get(p, 0.0)) for p, eps in inter.get(sp, [])) if theta else 0.0
    if getattr(sp, "thermo", None) is not None:
        sp.thermo.select(T)
        return float(sp.thermo.g(T, R, Tref, np.log)) + dH
    base = float(sp.H(theta)) if callable(sp.H) else sp.H
    H0 = base + dH
    if sp.Cp != 0.0:
        return (H0 + sp.Cp * (T - Tref)) - T * (sp.S + sp.Cp * np.log(T / Tref))
    return H0 - T * sp.S


def reaction_free_energy(mkm, rxn, T, theta=None) -> float:
    """Numeric reaction free energy ``sum_i nu_i G_i(T)`` (plus the
    electrochemical ``n_electrons * F * U`` shift for a faradaic step)."""
    dG = sum(nu * species_free_energy(mkm, s, T, theta) for s, nu in rxn.net_stoich().items())
    if getattr(rxn, "is_electrochemical", False):
        dG += rxn.n_electrons * mkm.F * mkm.U
    return dG


def _solve_ivp():
    """The implicit Radau IVP solver: POUNCE (stiff + DAE) if available, else SciPy."""
    try:
        from pounce import solve_ivp

        return solve_ivp, True
    except ImportError:
        from scipy.integrate import solve_ivp

        return solve_ivp, False


def integrate_coverages(mkm, conc, T, t_eval, theta0=None, dae=False, rtol=1e-8, atol=1e-12):
    """Integrate ``dtheta_i/dt = net_rate(adsorbate_i)`` over time (implicit Radau).

    Uses POUNCE's stiff ``solve_ivp`` (3-stage Radau IIA, L-stable) when available,
    falling back to SciPy. ``conc`` is the gas concentration as a constant
    ``{gas: value}`` dict or a callable ``conc(t) -> {gas: value}`` (e.g. a
    time-varying input). Free-site coverages follow from the site balance.

    Parameters
    ----------
    dae : bool
        If True, solve the index-1 DAE form ``M y' = f`` with explicit free-site
        algebraic variables and the site balance imposed as an algebraic equation
        (needs the POUNCE solver's mass-matrix support), instead of substituting
        ``theta_free = 1 - sum(theta)``.

    Returns the ``OdeResult`` (``sol.t``; ``sol.y`` rows = adsorbate coverages, in
    ``mkm.adsorbates`` order). For piecewise-constant inputs (e.g. a PRBS) integrate
    one constant segment per call and carry the final coverages forward as the next
    ``theta0``.
    """
    _reject_equilibrated(mkm)
    solve_ivp, is_pounce = _solve_ivp()
    ads, sites = mkm.adsorbates, mkm.sites
    conc_fn = conc if callable(conc) else (lambda t: conc)
    n = len(ads)
    theta0 = theta0 or {}
    span = (float(t_eval[0]), float(t_eval[-1]))

    def _gas(t):
        return {g: float(conc_fn(t).get(g, 0.0)) for g in mkm.gas_species}

    if not dae:
        def rhs(t, x):
            theta = {a: x[i] for i, a in enumerate(ads)}
            free = {s: max(1.0 - sum(theta[a] for a in mkm.adsorbates_on(s)), 0.0) for s in sites}
            kf, kr = rate_constants(mkm, T, theta)
            rops = rates_of_progress(mkm, kf, kr, theta, free, _gas(t))
            return [net_rate(mkm, a, rops) for a in ads]

        x0 = np.array([float(theta0.get(a, 0.0)) for a in ads])
        return solve_ivp(rhs, span, x0, method="Radau", t_eval=t_eval, rtol=rtol, atol=atol)

    if not is_pounce:
        raise RuntimeError("dae=True needs the POUNCE solver (mass-matrix support); it is unavailable")

    # index-1 DAE: free-site coverages are algebraic unknowns; M y' = f
    site_pos = {s: n + i for i, s in enumerate(sites)}
    mass = np.diag([1.0] * n + [0.0] * len(sites))

    def rhs(t, x):
        theta = {a: x[i] for i, a in enumerate(ads)}
        free = {s: x[site_pos[s]] for s in sites}
        kf, kr = rate_constants(mkm, T, theta)
        rops = rates_of_progress(mkm, kf, kr, theta, free, _gas(t))
        d = [net_rate(mkm, a, rops) for a in ads]
        d += [1.0 - sum(theta[a] for a in mkm.adsorbates_on(s)) - free[s] for s in sites]
        return d

    free0 = [max(1.0 - sum(theta0.get(a, 0.0) for a in mkm.adsorbates_on(s)), 0.0) for s in sites]
    x0 = np.array([float(theta0.get(a, 0.0)) for a in ads] + free0)
    return solve_ivp(rhs, span, x0, method="Radau", t_eval=t_eval, mass=mass, rtol=rtol, atol=atol)


def turnover_frequency(mkm, species, theta, free, pressures, T) -> float:
    """Net production rate of ``species`` at the given coverages/conditions."""
    _reject_equilibrated(mkm)
    kf, kr = rate_constants(mkm, T, theta)
    conc = {g: float(pressures.get(g, 0.0)) for g in mkm.gas_species}
    rops = rates_of_progress(mkm, kf, kr, theta, free, conc)
    return net_rate(mkm, species, rops)
