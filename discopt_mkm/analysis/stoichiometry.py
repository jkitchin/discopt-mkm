"""Stoichiometric structure: independence, reaction routes, conservation laws.

Pure linear algebra on the stoichiometric matrix ``nu`` (species x reactions),
built from each reaction's ``net_stoich``. No solve is required.

Two notions of "redundant" live here:

- **Stoichiometric** — the structure of ``nu`` itself: how many reactions are
  linearly independent, which combinations cancel the surface intermediates
  (Horiuti-Temkin reaction routes), and which species combinations are conserved
  (e.g. the site balance). That is what this module computes.
- **Kinetic** — whether a step actually controls or whether its rate constants
  are identifiable. That comes from the degree of rate control
  (:mod:`discopt_mkm.analysis.drc`) and the estimation Fisher information.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import null_space, qr

from discopt_mkm.species import GasSpecies


def stoichiometric_matrix(mkm, species=None):
    """Return ``(nu, species, reactions)`` with ``nu[i, j] = net coeff of species i in reaction j``."""
    species = list(species) if species is not None else list(mkm.species)
    reactions = list(mkm.reactions)
    nu = np.zeros((len(species), len(reactions)))
    index = {sp: i for i, sp in enumerate(species)}
    for j, rxn in enumerate(reactions):
        for sp, c in rxn.net_stoich().items():
            nu[index[sp], j] = c
    return nu, species, reactions


def _rank(A, tol=1e-9) -> int:
    if A.size == 0:
        return 0
    return int(np.linalg.matrix_rank(A, tol=tol))


def _rationalize(v, tol=1e-9):
    """Scale a null-space vector to small integers when possible (for display)."""
    v = np.asarray(v, dtype=float)
    nz = np.abs(v[np.abs(v) > tol])
    if nz.size:
        v = v / nz.min()
    rounded = np.round(v)
    if np.allclose(v, rounded, atol=1e-6):
        v = rounded
    # canonical sign: first significant entry positive
    for x in v:
        if abs(x) > tol:
            if x < 0:
                v = -v
            break
    return v


def n_independent_reactions(mkm) -> int:
    """Number of stoichiometrically independent reactions = ``rank(nu)``."""
    nu, _, _ = stoichiometric_matrix(mkm)
    return _rank(nu)


def independent_reactions(mkm):
    """A maximal linearly independent subset of reactions (column-pivoted QR)."""
    nu, _, reactions = stoichiometric_matrix(mkm)
    if nu.shape[1] == 0:
        return []
    r = _rank(nu)
    _, _, piv = qr(nu, pivoting=True, mode="economic")
    keep = sorted(int(j) for j in piv[:r])
    return [reactions[j] for j in keep]


def is_redundant(mkm, rxn) -> bool:
    """True if ``rxn``'s net stoichiometry is a linear combination of the others.

    (Stoichiometric dependence, not a statement that the step is mechanistically
    removable.)
    """
    nu, _, reactions = stoichiometric_matrix(mkm)
    j = reactions.index(rxn)
    return _rank(np.delete(nu, j, axis=1)) == _rank(nu)


def reaction_routes(mkm, tol=1e-9):
    """Horiuti-Temkin reaction routes.

    A route is a combination of steps whose stoichiometric numbers cancel every
    surface intermediate (adsorbates and sites); what remains is a net reaction
    over the gas-phase species. Returns a list of ``(sigma, overall)`` where
    ``sigma`` is the per-reaction stoichiometric-number vector and ``overall`` is
    ``{gas_species: coefficient}`` of the resulting overall reaction (empty for a
    closed cycle).
    """
    nu, species, reactions = stoichiometric_matrix(mkm)
    if not reactions:
        return []
    inter_idx = [i for i, sp in enumerate(species) if not isinstance(sp, GasSpecies)]
    gas_idx = [i for i, sp in enumerate(species) if isinstance(sp, GasSpecies)]
    nu_int = nu[inter_idx, :] if inter_idx else np.zeros((0, len(reactions)))
    basis = null_space(nu_int, rcond=tol) if nu_int.size else np.eye(len(reactions))

    routes = []
    for k in range(basis.shape[1]):
        sigma = _rationalize(basis[:, k], tol)
        overall = {}
        for i in gas_idx:
            coeff = float(nu[i, :] @ sigma)
            if abs(coeff) > 1e-6:
                overall[species[i]] = coeff
        routes.append((sigma, overall))
    return routes


def site_conservation_laws(mkm):
    """Clean structural site balances, one per site type.

    For each site, ``{site: 1, adsorbate: 1, ...}`` over the adsorbates on it —
    the invariant the solver enforces as the site balance. Always a true
    conservation law (``nu^T v = 0``); this is the interpretable recovery of the
    site balance from the stoichiometry.
    """
    laws = []
    for site in mkm.sites:
        law = {site: 1.0}
        for a in mkm.adsorbates_on(site):
            law[a] = 1.0
        laws.append(law)
    return laws


def element_conservation_laws(mkm, tol=1e-9):
    """Conserved element balances from species composition.

    For each element appearing in any species' ``composition``, the atom-count
    vector is a conservation law iff every reaction balances that element. Returns
    ``{element: {species: count}}`` for the elements that are conserved (skipping
    any that are not — see :func:`check_element_balance`). Requires compositions
    to be set (explicitly or via ``Model.infer_composition``).
    """
    nu, species, _ = stoichiometric_matrix(mkm)
    elements = sorted({e for sp in species for e in sp.composition})
    laws = {}
    for e in elements:
        vec = np.array([float(sp.composition.get(e, 0)) for sp in species])
        if np.allclose(nu.T @ vec, 0.0, atol=tol):
            laws[e] = {species[i]: vec[i] for i in range(len(species)) if vec[i] != 0}
    return laws


def check_element_balance(mkm, tol=1e-9):
    """Verify every reaction conserves every element. Returns a list of
    ``(reaction, element, residual)`` for any imbalance (empty if all balanced)."""
    elements = {e for sp in mkm.species for e in sp.composition}
    violations = []
    for rxn in mkm.reactions:
        for e in elements:
            residual = sum(nu * sp.composition.get(e, 0) for sp, nu in rxn.net_stoich().items())
            if abs(residual) > tol:
                violations.append((rxn, e, residual))
    return violations


def check_site_balance(mkm, tol=1e-9):
    """Verify every reaction conserves each site type. Returns a list of
    ``(reaction, site, residual)`` for any imbalance (sites created/destroyed)."""
    violations = []
    for site in mkm.sites:
        occ = {a: 1.0 for a in mkm.adsorbates_on(site)}
        occ[site] = 1.0
        for rxn in mkm.reactions:
            residual = sum(nu * occ.get(sp, 0.0) for sp, nu in rxn.net_stoich().items())
            if abs(residual) > tol:
                violations.append((rxn, site, residual))
    return violations


def conserved_quantities(mkm):
    """Labeled conservation laws: site balances + element balances.

    Returns ``{"site:<name>": {...}, "element:<symbol>": {...}}`` — the clean,
    interpretable basis (in contrast to :func:`conservation_laws`, which returns an
    arbitrary numerical basis). With composition set, the number of these should
    equal :func:`n_conservation_laws`.
    """
    out = {}
    for site, law in zip(mkm.sites, site_conservation_laws(mkm)):
        out[f"site:{site.name}"] = law
    for e, vec in element_conservation_laws(mkm).items():
        out[f"element:{e}"] = vec
    return out


def n_conservation_laws(mkm) -> int:
    """Number of independent conserved quantities = ``n_species - rank(nu)``.

    For a heterogeneous mechanism this is the number of site types plus the
    number of conserved elements.
    """
    nu, species, _ = stoichiometric_matrix(mkm)
    return len(species) - _rank(nu)


def conservation_laws(mkm, tol=1e-9):
    """Conserved species combinations = left null space of ``nu``.

    Returns a list of ``{species: coefficient}`` dicts, each a combination every
    reaction leaves unchanged. NOTE: this is an arbitrary orthonormal basis of
    the invariant space — the *count* is meaningful (see :func:`n_conservation_laws`)
    and the site balances lie in its span, but the individual vectors are not
    cleanly separated into "site" vs "element" unless species carry atomic
    composition. Use :func:`site_conservation_laws` for the interpretable site
    balances.
    """
    nu, species, _ = stoichiometric_matrix(mkm)
    if nu.size == 0:
        return []
    basis = null_space(nu.T, rcond=tol)
    laws = []
    for k in range(basis.shape[1]):
        v = _rationalize(basis[:, k], tol)
        laws.append({species[i]: v[i] for i in range(len(species)) if abs(v[i]) > 1e-6})
    return laws


def summary(mkm) -> dict:
    """A compact dict summary of the stoichiometric structure."""
    nu, species, reactions = stoichiometric_matrix(mkm)
    return {
        "n_species": len(species),
        "n_reactions": len(reactions),
        "rank": _rank(nu),
        "n_independent_reactions": _rank(nu),
        "n_routes": len(reaction_routes(mkm)),
        "n_conservation_laws": len(conservation_laws(mkm)),
    }
