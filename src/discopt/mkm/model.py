"""The user-facing microkinetic model: species/site/reaction registration."""

from __future__ import annotations

import re

from discopt.mkm.reaction import Reaction
from discopt.mkm.species import Adsorbate, GasSpecies, Site


def _safe(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", name)


class MicrokineticModel:
    """A heterogeneous-catalysis microkinetic model.

    Parameters
    ----------
    name : str
        Model name.
    T : float
        Temperature (default 500). Drives kinetics and thermodynamics.
    R : float
        Gas constant in units consistent with the species/reaction energies.
        Default ``8.617e-5`` (eV/K). Use ``8.314`` for J/mol/K.
    Tref : float
        Reference temperature for heat-capacity corrections (default 298.15).
    """

    def __init__(self, name: str, T: float = 500.0, R: float = 8.617e-5, Tref: float = 298.15,
                 U: float = 0.0, F: float = 1.0):
        self.name = name
        self.T = float(T)
        self.R = float(R)
        self.Tref = float(Tref)
        # Electrode potential (V) and Faraday constant in the model's energy
        # units. F = 1.0 means energies in eV and U in volts (1 eV per electron
        # per volt); use F = 96485 for J/mol. U is an operating condition, swept
        # by re-solving, exactly like T. Only used by electrochemical steps.
        self.U = float(U)
        self.F = float(F)
        self._U_param = None
        self.gas_species: list[GasSpecies] = []
        self.sites: list[Site] = []
        self.adsorbates: list[Adsorbate] = []
        self.reactions: list[Reaction] = []
        self._interactions: list[tuple] = []  # (a, b, eps) lateral interactions
        self._by_name: dict[str, object] = {}

    # -- registration -----------------------------------------------------
    def _register(self, sp):
        if sp.name in self._by_name:
            raise ValueError(f"species {sp.name!r} already defined")
        self._by_name[sp.name] = sp
        sp.model = self
        return sp

    def gas(self, name: str, H: float = 0.0, S: float = 0.0, Cp: float = 0.0, thermo=None, composition=None) -> GasSpecies:
        sp = self._register(GasSpecies(name, H=H, S=S, Cp=Cp, thermo=thermo, composition=composition))
        self.gas_species.append(sp)
        return sp

    def site(self, name: str, density: float, H: float = 0.0, S: float = 0.0, thermo=None) -> Site:
        sp = self._register(Site(name, density=density, H=H, S=S, thermo=thermo))
        self.sites.append(sp)
        return sp

    def adsorbate(
        self, name: str, site: Site, H: float = 0.0, S: float = 0.0, Cp: float = 0.0, thermo=None, composition=None
    ) -> Adsorbate:
        sp = self._register(Adsorbate(name, site=site, H=H, S=S, Cp=Cp, thermo=thermo, composition=composition))
        self.adsorbates.append(sp)
        return sp

    def infer_composition(self) -> list:
        """Best-effort: fill empty species ``composition`` by parsing names as formulas.

        Parses each gas/adsorbate name (the surface-site ``*`` stripped); sites are
        left empty (they carry no atoms). Returns the list of species whose names
        do not look like chemical formulas (left untouched for you to set
        explicitly with ``composition=``).
        """
        from discopt.mkm.formula import looks_like_formula, parse_formula

        unparsed = []
        for sp in [*self.gas_species, *self.adsorbates]:
            if sp.composition:
                continue
            if looks_like_formula(sp.name):
                sp.composition = parse_formula(sp.name)
            else:
                unparsed.append(sp)
        return unparsed

    def interaction(self, a: Adsorbate, b: Adsorbate, eps: float) -> None:
        """Register a symmetric lateral interaction between two adsorbates.

        The formation enthalpy of each adsorbate ``i`` becomes coverage-dependent,
        ``H_i(theta) = H_i^0 + sum_j eps_ij theta_j``. ``interaction(a, b, eps)``
        sets ``eps_ab = eps_ba = eps`` (use ``a is b`` for a self-interaction;
        positive ``eps`` is repulsive — it destabilizes as coverage rises). The
        ``eps`` value becomes a differentiable parameter.
        """
        if not isinstance(a, Adsorbate) or not isinstance(b, Adsorbate):
            raise TypeError("interactions are defined between adsorbates")
        self._interactions.append((a, b, float(eps)))

    def step(
        self,
        reaction: Reaction,
        A: float | None = None,
        Ea: float | None = None,
        kf: float | None = None,
        Keq: float | None = None,
        irreversible: bool = False,
        equilibrated: bool = False,
        alpha: float = 0.0,
        n_electrons: float | None = None,
        beta: float = 0.5,
        name: str | None = None,
    ) -> Reaction:
        """Attach kinetics to a reaction and register it.

        Two mutually exclusive modes:

        - **Arrhenius** (default): pass ``A`` and ``Ea``. The forward rate is
          ``A exp(-Ea/RT)`` and the reverse rate is derived from species
          thermodynamics (``k_f / K_eq``, ``K_eq = exp(-dG_rxn/RT)``).
        - **Explicit rate**: pass ``kf`` and ``Keq`` directly (e.g. transcribed
          from a DFT/SI table). The reverse rate is ``kf / Keq``. No species
          thermodynamics are needed for this step.

        Pass ``irreversible=True`` to set the reverse rate to exactly zero. No
        ``K_eq`` / reverse-rate expression is built (so an Arrhenius irreversible
        step needs no product thermodynamics), and an explicit irreversible step
        needs only ``kf`` (``Keq`` is not required).

        Pass ``equilibrated=True`` for the quasi-equilibrium approximation: the
        step's rate law is dropped and an equilibrium-quotient constraint
        ``Q = K_eq`` is imposed instead (its rate of progress becomes an unknown
        extent). Only ``K_eq`` is needed — give it explicitly (``Keq=...``) or
        leave it to come from species thermodynamics. Rate constants are not
        used. Supported in steady-state solves.
        """
        reaction.irreversible = bool(irreversible)
        reaction.equilibrated = bool(equilibrated)
        reaction.alpha = float(alpha)
        reaction.n_electrons = None if n_electrons is None else float(n_electrons)
        reaction.beta = float(beta)
        if equilibrated:
            # only K_eq matters: explicit if given, else from species thermo
            if Keq is not None:
                reaction.Keq = float(Keq)
                reaction.explicit_rate = True
            reaction.name = name or reaction.equation()
            self.reactions.append(reaction)
            return reaction
        if kf is not None or Keq is not None:
            if kf is None:
                raise ValueError("explicit-rate steps need kf")
            if Keq is None and not irreversible:
                raise ValueError("reversible explicit-rate steps need Keq")
            reaction.kf = float(kf)
            reaction.Keq = float(Keq) if Keq is not None else None
            reaction.explicit_rate = True
        else:
            if A is None or Ea is None:
                raise ValueError("Arrhenius steps need both A and Ea")
            reaction.A = float(A)
            reaction.Ea = float(Ea)
        reaction.name = name or reaction.equation()
        self.reactions.append(reaction)
        return reaction

    # -- helpers ----------------------------------------------------------
    @property
    def species(self) -> list:
        return [*self.gas_species, *self.sites, *self.adsorbates]

    def adsorbates_on(self, site: Site) -> list[Adsorbate]:
        return [a for a in self.adsorbates if a.site is site]

    # -- parameter wiring -------------------------------------------------
    def wire_parameters(self, m):
        """Attach one scalar ``dm.parameter`` per differentiation handle.

        Creates handles on a *fresh* discopt model ``m`` for the temperature,
        every species ``H``/``S``/``Cp``/``dG`` offset, and every reaction
        ``A``/``Ea``. Returns the temperature parameter. Scalar-per-handle keeps
        the sensitivity-matrix column bookkeeping unambiguous for DRC/TRC.
        """
        T_param = m.parameter("T", self.T)
        U_param = m.parameter("U", self.U)
        self._U_param = U_param
        for i, sp in enumerate(self.species):
            tag = f"{i}_{_safe(sp.name)}"
            sp._interaction_params = []  # reset (a fresh model each solve)
            if sp.thermo is not None:
                # temperature-dependent model supplies G(T); pick its range from
                # the nominal temperature and skip the constant H/S/Cp parameters
                sp.thermo.select(self.T)
                sp.H_param = sp.S_param = sp.Cp_param = None
            else:
                # callable H is evaluated as H(theta); only constant H gets a parameter
                sp.H_param = None if callable(sp.H) else m.parameter(f"H_{tag}", sp.H)
                sp.S_param = m.parameter(f"S_{tag}", sp.S)
                sp.Cp_param = m.parameter(f"Cp_{tag}", sp.Cp) if sp.Cp != 0.0 else None
            sp.dG_param = m.parameter(f"dG_{tag}", 0.0)
        # lateral interactions: one differentiable eps parameter per pair, shared
        # symmetrically between the two adsorbates.
        for k, (a, b, eps) in enumerate(self._interactions):
            eps_param = m.parameter(f"eps_{k}", eps)
            a._interaction_params.append((b, eps_param))
            if a is not b:
                b._interaction_params.append((a, eps_param))
        has_interactions = bool(self._interactions)
        for j, rxn in enumerate(self.reactions):
            rxn.alpha_param = None
            rxn.beta_param = None
            # electrochemical handles: the global potential and a per-step
            # transfer coefficient drive the CHE / Butler-Volmer shifts.
            if rxn.is_electrochemical:
                rxn._U_param = U_param
                rxn._F = self.F
                rxn.beta_param = m.parameter(f"beta_{j}", rxn.beta)
            if rxn.equilibrated:
                # only K_eq is used; wire it as a parameter when given explicitly
                # (otherwise it comes from the species thermo parameters above)
                if rxn.Keq is not None:
                    rxn.Keq_param = m.parameter(f"Keq_{j}", rxn.Keq)
            elif rxn.explicit_rate:
                rxn.kf_param = m.parameter(f"kf_{j}", rxn.kf)
                if not rxn.irreversible:
                    rxn.Keq_param = m.parameter(f"Keq_{j}", rxn.Keq)
            else:
                rxn.A_param = m.parameter(f"A_{j}", rxn.A)
                rxn.Ea_param = m.parameter(f"Ea_{j}", rxn.Ea)
                # BEP coverage dependence of the barrier (only with interactions)
                if has_interactions:
                    rxn.alpha_param = m.parameter(f"alpha_{j}", rxn.alpha)
        return T_param

    def __repr__(self) -> str:
        return (
            f"MicrokineticModel({self.name!r}, T={self.T}, "
            f"{len(self.species)} species, {len(self.reactions)} reactions)"
        )

    # -- rendering --------------------------------------------------------
    def to_latex(self) -> str:
        """An ``align`` block of all elementary steps (LaTeX source)."""
        from discopt.mkm import render

        return render.mechanism_latex(self)

    def to_html(self) -> str:
        """A mechanism table (step, reaction, kinetics, type)."""
        from discopt.mkm import render

        return render.mechanism_html(self)

    def _repr_html_(self) -> str:
        return self.to_html()
