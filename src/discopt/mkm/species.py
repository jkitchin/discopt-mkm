"""Species algebra for the discopt microkinetic-modeling plugin.

Species are *not* discopt ``Expression`` objects. They are lightweight domain
objects that overload ``+``, ``*`` (for stoichiometric coefficients) and ``>>``
(to close a reaction) so that a reaction network can be written as::

    m.step(CO + s >> COs, A=1e8, Ea=0.0)
    m.step(COs + Os >> CO2 + 2 * s, A=1e13, Ea=1.0)

The operators build ``StoichTerm`` / ``ReactionSide`` collections and finally a
:class:`~discopt.mkm.reaction.Reaction` with reactant/product stoichiometry
dicts. Discopt expressions are only produced later, at model-assembly time, by
:mod:`discopt.mkm.kinetics`.
"""

from __future__ import annotations

from typing import Union


class Species:
    """A chemical species carrying constant thermodynamic data.

    Parameters
    ----------
    name : str
        Unique species label (e.g. ``"CO"`` or ``"CO*"``).
    H : float
        Standard enthalpy (energy units consistent with ``Ea`` and ``R``).
    S : float
        Standard entropy (same energy units per K).
    Cp : float, optional
        Constant heat capacity for the temperature correction of ``H`` and
        ``S``. Default ``0`` (no correction).
    """

    phase: str = "abstract"

    def __init__(self, name: str, H=0.0, S: float = 0.0, Cp: float = 0.0, thermo=None, composition=None):
        self.name = name
        # elemental composition {element: count}; used for element conservation
        # laws / mass-balance checks. Empty for bare sites.
        self.composition = dict(composition) if composition else {}
        # H may be a constant or a callable H(theta) -> expression for
        # coverage-dependent (lateral-interaction) energetics.
        self.H = H if callable(H) else float(H)
        self.S = float(S)
        self.Cp = float(Cp)
        # optional temperature-dependent thermo model (NASA7 / Shomate / general);
        # when set it supplies G(T) and overrides the constant H/S/Cp path.
        self.thermo = thermo
        # filled in when registered with a model:
        self.model = None
        self.H_param = None
        self.S_param = None
        self.Cp_param = None
        self.dG_param = None  # free-energy offset handle, used only for TRC
        self._interaction_params = []  # list of (partner_species, eps_param)

    # -- operator algebra -------------------------------------------------
    def __mul__(self, coeff: float) -> "StoichTerm":
        return StoichTerm(self, float(coeff))

    __rmul__ = __mul__  # support both ``2 * s`` and ``s * 2``

    def __add__(self, other) -> "ReactionSide":
        return ReactionSide([StoichTerm(self, 1.0)]) + other

    def __rshift__(self, other) -> "object":
        return ReactionSide([StoichTerm(self, 1.0)]) >> other

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class GasSpecies(Species):
    """A gas-phase species. Its activity is its concentration / partial pressure."""

    phase = "gas"


class Site(Species):
    """A catalyst site type.

    Parameters
    ----------
    name : str
        Site label (e.g. ``"Pt"``).
    density : float
        Areal site density (sites per unit area). **Currently metadata only** — it
        is stored for reference but not yet wired into the rate / ODE / steady-state
        equations (coverages are fractional and the site balance is normalized to
        1). The active scaling knob coupling surface rate to the gas balance is the
        reactor-level ``cat_density`` (see :mod:`discopt.mkm.reactors`).
    """

    phase = "site"

    def __init__(self, name: str, density: float, H: float = 0.0, S: float = 0.0, Cp: float = 0.0, thermo=None):
        super().__init__(name, H=H, S=S, Cp=Cp, thermo=thermo)  # sites carry no atoms
        self.density = float(density)


class Adsorbate(Species):
    """A surface-bound species occupying a :class:`Site`.

    Its activity is its fractional coverage ``theta``.
    """

    phase = "surface"

    def __init__(self, name: str, site: Site, H: float = 0.0, S: float = 0.0, Cp: float = 0.0, thermo=None, composition=None):
        super().__init__(name, H=H, S=S, Cp=Cp, thermo=thermo, composition=composition)
        if not isinstance(site, Site):
            raise TypeError(f"adsorbate {name!r} site must be a Site, got {type(site).__name__}")
        self.site = site


class StoichTerm:
    """A ``(species, coefficient)`` pair produced by ``coeff * species``."""

    def __init__(self, species: Species, coeff: float = 1.0):
        self.species = species
        self.coeff = float(coeff)

    def __add__(self, other) -> "ReactionSide":
        return ReactionSide([self]) + other

    def __rshift__(self, other):
        return ReactionSide([self]) >> other

    def __repr__(self) -> str:
        return f"{self.coeff:g} {self.species.name}"


def _as_terms(obj) -> list[StoichTerm]:
    """Normalize a Species / StoichTerm / ReactionSide into a list of terms."""
    if isinstance(obj, ReactionSide):
        return list(obj.terms)
    if isinstance(obj, StoichTerm):
        return [obj]
    if isinstance(obj, Species):
        return [StoichTerm(obj, 1.0)]
    raise TypeError(f"cannot interpret {obj!r} as part of a reaction side")


class ReactionSide:
    """An additive collection of :class:`StoichTerm` (one side of a reaction)."""

    def __init__(self, terms: list[StoichTerm]):
        self.terms = list(terms)

    def __add__(self, other) -> "ReactionSide":
        return ReactionSide(self.terms + _as_terms(other))

    def __rshift__(self, products):
        # local import to avoid a circular import at module load
        from discopt.mkm.reaction import Reaction

        return Reaction(reactants=self._stoich(), products=ReactionSide(_as_terms(products))._stoich())

    def _stoich(self) -> dict:
        """Merge terms into a ``{species: coefficient}`` dict, summing duplicates."""
        stoich: dict = {}
        for term in self.terms:
            stoich[term.species] = stoich.get(term.species, 0.0) + term.coeff
        return stoich

    def __repr__(self) -> str:
        return " + ".join(repr(t) for t in self.terms)


SpeciesLike = Union[Species, StoichTerm, ReactionSide]
