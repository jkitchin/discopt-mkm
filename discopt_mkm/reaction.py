"""Reaction objects produced by the species ``>>`` operator."""

from __future__ import annotations

from collections import defaultdict

from discopt_mkm.species import Species


class Reaction:
    """An elementary reversible step with reactant/product stoichiometry.

    A reaction is created by the ``>>`` operator on species
    (``reactants >> products``); kinetic data (``A``, ``Ea``) is attached later
    by :meth:`~discopt_mkm.model.MicrokineticModel.step`. The reverse rate is
    never stored: it is *derived* from the forward rate and the equilibrium
    constant so that thermodynamic consistency is structural.
    """

    def __init__(self, reactants: dict, products: dict):
        self.reactants: dict[Species, float] = dict(reactants)
        self.products: dict[Species, float] = dict(products)
        self.name: str | None = None
        # Arrhenius mode (default): forward k = A exp(-Ea/RT), reverse derived
        # from species thermodynamics.
        self.A: float | None = None
        self.Ea: float | None = None
        # Explicit-rate mode: forward k and equilibrium constant given directly
        # (e.g. transcribed from DFT / SI tables). Reverse is k_f / K_eq.
        self.kf: float | None = None
        self.Keq: float | None = None
        self.explicit_rate: bool = False
        # Irreversible: reverse rate is exactly zero; no K_eq / reverse-rate
        # expression is built and no product thermodynamics are needed.
        self.irreversible: bool = False
        # Quasi-equilibrated: the step is assumed at equilibrium. Its rate law is
        # NOT used; instead its rate of progress is an unknown extent and an
        # equilibrium-quotient constraint (Q = K_eq) is imposed. Only K_eq is
        # needed (from thermo or given), not the rate constants.
        self.equilibrated: bool = False
        # Brønsted-Evans-Polanyi transfer coefficient: fraction of the
        # coverage-induced reaction-energy change that shifts the forward barrier.
        self.alpha: float = 0.0
        self.alpha_param = None
        # Electrochemical step: transfers ``n_electrons`` electrons (consumed in
        # the forward / reduction direction; negative for an oxidation written
        # forward). The reaction free energy shifts by ``n_electrons * F * U``
        # (computational hydrogen electrode) and the forward barrier by
        # ``beta * n_electrons * F * U`` (Butler-Volmer, transfer coefficient
        # beta). ``None``/0 electrons leaves the step purely chemical.
        self.n_electrons: float | None = None
        self.beta: float = 0.5
        self.beta_param = None
        self._U_param = None   # global potential handle, set at assembly
        self._F = 1.0          # Faraday constant in the model's energy units
        # discopt Parameter (or Variable) handles, set at model assembly:
        self.A_param = None
        self.Ea_param = None
        self.kf_param = None
        self.Keq_param = None

    @property
    def is_electrochemical(self) -> bool:
        return self.n_electrons not in (None, 0, 0.0)

    def rate_constant_param(self):
        """The forward-rate-constant handle to perturb for degree of rate control."""
        return self.kf_param if self.explicit_rate else self.A_param

    def rate_constant_value(self) -> float:
        """Numeric value of the forward-rate-constant handle."""
        return float(self.kf if self.explicit_rate else self.A)

    def net_stoich(self) -> dict[Species, float]:
        """Net stoichiometric coefficients ``nu_i`` (products positive)."""
        nu: dict[Species, float] = defaultdict(float)
        for s, c in self.reactants.items():
            nu[s] -= c
        for s, c in self.products.items():
            nu[s] += c
        # drop species that cancel out
        return {s: c for s, c in nu.items() if abs(c) > 1e-12}

    def species(self) -> set[Species]:
        """All species participating in the reaction."""
        return set(self.reactants) | set(self.products)

    def _format_side(self, stoich: dict[Species, float]) -> str:
        parts = []
        for s, c in stoich.items():
            parts.append(s.name if abs(c - 1.0) < 1e-12 else f"{c:g} {s.name}")
        return " + ".join(parts) if parts else "0"

    def equation(self) -> str:
        arrow = " -> " if self.irreversible else " <=> "
        return f"{self._format_side(self.reactants)}{arrow}{self._format_side(self.products)}"

    def __repr__(self) -> str:
        return f"Reaction({self.equation()!r})"

    # -- rendering --------------------------------------------------------
    def to_latex(self, inline: bool = True) -> str:
        from discopt_mkm import render

        return render.reaction_latex(self, inline=inline)

    def to_html(self) -> str:
        from discopt_mkm import render

        return render.reaction_html(self)

    def _repr_latex_(self) -> str:
        return self.to_latex(inline=True)

    def _repr_html_(self) -> str:
        return self.to_html()
