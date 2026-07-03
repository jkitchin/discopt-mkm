"""Parameter estimation bridge to ``discopt.estimate``.

Fits kinetic/thermodynamic constants of a microkinetic model to measured rate
(or coverage) data across one or more operating conditions. Uses discopt's
all-at-once weighted-least-squares estimator: the unknown constants and every
condition's steady-state coverages are solved *simultaneously* subject to the
steady-state constraints, and discopt returns the Fisher information matrix,
covariance, and confidence intervals.

Design
------
- Each fitted constant becomes a discopt **Variable** (not a Parameter), since
  ``discopt.estimate`` differentiates the objective w.r.t. Variables. Pre-
  exponentials, which span many orders of magnitude, are fit in log space by
  default (``A = exp(u)``) for good conditioning; results are reported back in
  physical units via the delta method.
- Every other constant is a fixed ``Parameter``, shared across conditions.
- Each condition contributes its own coverage variables, steady-state
  constraints, and a scalar response expression (e.g. a turnover frequency).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import discopt.modeling as dm
from discopt.estimate import Experiment, ExperimentModel, estimate_parameters

from discopt.mkm.assemble import site_balance_residual
from discopt.mkm.kinetics import net_rate
from discopt.mkm.model import MicrokineticModel, _safe
from discopt.mkm.reaction import Reaction
from discopt.mkm.species import GasSpecies, Species

# attr -> the parameter-handle attribute the kinetics code reads
_ATTR_TO_HANDLE = {"A": "A_param", "Ea": "Ea_param", "kf": "kf_param", "Keq": "Keq_param",
                   "H": "H_param", "S": "S_param", "Cp": "Cp_param", "beta": "beta_param"}


@dataclass
class FitParam:
    """A constant to estimate.

    Parameters
    ----------
    target : Reaction or Species
        The reaction (for ``"A"``/``"Ea"``) or species (for ``"H"``/``"S"``/
        ``"Cp"``) whose constant is fit.
    attr : {"A", "Ea", "H", "S", "Cp"}
        Which constant.
    lb, ub : float
        Bounds in physical units.
    init : float, optional
        Starting value in physical units (defaults to the target's current
        value). Used to center the search box.
    log : bool, optional
        Fit in log space (``value = exp(u)``). Defaults to ``True`` for ``"A"``
        (pre-exponentials), ``False`` otherwise.
    name : str, optional
        Result key (defaults to ``f"{attr}_{target label}"``).
    """

    target: object
    attr: str
    lb: float
    ub: float
    init: float | None = None
    log: bool | None = None
    name: str | None = None

    def resolved_name(self) -> str:
        if self.name:
            return self.name
        label = self.target.name if isinstance(self.target, Species) else (
            self.target.name or "rxn"
        )
        return f"{self.attr}_{_safe(label)}"

    def is_log(self) -> bool:
        return self.attr == "A" if self.log is None else bool(self.log)

    def current_value(self) -> float:
        if self.init is not None:
            return float(self.init)
        if isinstance(self.target, Reaction):
            return float(getattr(self.target, self.attr))
        return float(getattr(self.target, self.attr))


@dataclass
class Observation:
    """A single measured data point at one operating condition.

    Parameters
    ----------
    response : Species
        The species whose net production rate (turnover frequency) was measured.
    value : float
        The measured rate.
    T : float
        Temperature for this condition.
    pressures : dict
        Fixed gas activities ``{gas_species: partial pressure}`` (differential
        reactor: gas held constant).
    sigma : float
        Measurement standard deviation (default 1.0).
    label : str, optional
        Unique key for this observation (defaults to ``obs{index}``).
    """

    response: Species
    value: float
    T: float
    pressures: dict
    sigma: float = 1.0
    label: str | None = None
    theta0: dict = field(default_factory=dict)
    U: float = 0.0   # electrode potential for this condition (electrochemical fits)


@dataclass
class MKMEstimationResult:
    """Estimation result in physical units, wrapping discopt's result."""

    parameters: dict
    std_errors: dict
    confidence_intervals: dict
    objective: float
    n_observations: int
    raw: object  # discopt EstimationResult

    def __repr__(self) -> str:
        lines = ["MKMEstimationResult:"]
        for k in self.parameters:
            lo, hi = self.confidence_intervals[k]
            lines.append(f"  {k} = {self.parameters[k]:.6g}  (95% CI [{lo:.4g}, {hi:.4g}])")
        lines.append(f"  objective={self.objective:.6g}, n_obs={self.n_observations}")
        return "\n".join(lines)


class _MKMExperiment(Experiment):
    """discopt Experiment that builds a multi-condition simultaneous MKM fit."""

    def __init__(self, mkm: MicrokineticModel, observations: list[Observation], fit: list[FitParam]):
        self.mkm = mkm
        self.observations = observations
        self.fit = fit

    def create_model(self, **kwargs) -> ExperimentModel:
        mkm = self.mkm
        if any(r.equilibrated for r in mkm.reactions):
            # equilibrated steps have no rate law: they need an unknown extent
            # variable plus an equilibrium-quotient constraint per observation
            # (as the steady-state builder does). The estimation residuals here
            # (net_rate without extents) cannot represent that, so refuse rather
            # than silently fit a mechanism with those steps deleted.
            raise NotImplementedError("fitting models with equilibrated steps is not supported")
        m = dm.Model(f"{mkm.name}_fit")

        # 1) fitted constants -> shared Variables; attach effective expression to the handle
        unknown: dict[str, dm.Variable] = {}
        log_names: set[str] = set()
        for fp in self.fit:
            name = fp.resolved_name()
            v0 = float(kwargs.get(name, fp.current_value()))
            if fp.is_log():
                var = m.continuous(name, lb=float(np.log(fp.lb)), ub=float(np.log(fp.ub)))
                setattr(fp.target, _ATTR_TO_HANDLE[fp.attr], dm.exp(var))
                log_names.add(name)
            else:
                var = m.continuous(name, lb=fp.lb, ub=fp.ub)
                setattr(fp.target, _ATTR_TO_HANDLE[fp.attr], var)
            unknown[name] = var

        # 2) every non-fitted constant -> shared fixed Parameter. Reset the stale
        # handles left on the shared Species/Reaction objects by any earlier solve
        # (each fit builds a fresh discopt model), exactly as
        # ``model.wire_parameters`` does — otherwise a previously-solved model
        # mixes parameter handles from two different discopt models.
        fitted = {(id(fp.target), fp.attr) for fp in self.fit}
        for sp in mkm.species:
            sp._interaction_params = []  # reset lateral-interaction handles
            if (id(sp), "H") not in fitted:
                sp.H_param = m.parameter(f"H_{_safe(sp.name)}", sp.H)
            if (id(sp), "S") not in fitted:
                sp.S_param = m.parameter(f"S_{_safe(sp.name)}", sp.S)
            sp.Cp_param = (
                m.parameter(f"Cp_{_safe(sp.name)}", sp.Cp)
                if (sp.Cp != 0.0 and (id(sp), "Cp") not in fitted)
                else getattr(sp, "Cp_param", None)
            )
            sp.dG_param = None  # not used in estimation
        # lateral interactions: one shared eps parameter per pair
        has_interactions = bool(getattr(mkm, "_interactions", []))
        for k, (a, b, eps) in enumerate(getattr(mkm, "_interactions", [])):
            eps_param = m.parameter(f"eps_{k}", eps)
            a._interaction_params.append((b, eps_param))
            if a is not b:
                b._interaction_params.append((a, eps_param))
        for j, rxn in enumerate(mkm.reactions):
            # reset per-reaction handles (fresh model each fit); keep any handle
            # that step 1 already wired as a fitted Variable.
            rxn.alpha_param = None
            if (id(rxn), "beta") not in fitted:
                rxn.beta_param = m.parameter(f"beta_{j}", rxn.beta) if rxn.is_electrochemical else None
            if rxn.explicit_rate:
                # explicit kf/Keq step (e.g. from a DFT/SI table): k_forward reads
                # kf_param and k_reverse reads Keq_param, so both must be wired.
                if (id(rxn), "kf") not in fitted:
                    rxn.kf_param = m.parameter(f"kf_{j}", rxn.kf)
                if not rxn.irreversible and (id(rxn), "Keq") not in fitted:
                    rxn.Keq_param = m.parameter(f"Keq_{j}", rxn.Keq)
            else:
                if (id(rxn), "A") not in fitted:
                    rxn.A_param = m.parameter(f"A_{j}", rxn.A)
                if (id(rxn), "Ea") not in fitted:
                    rxn.Ea_param = m.parameter(f"Ea_{j}", rxn.Ea)
                # BEP coverage dependence of the barrier (only with interactions)
                if has_interactions:
                    rxn.alpha_param = m.parameter(f"alpha_{j}", rxn.alpha)

        # 3) one steady-state block + response per observation
        responses: dict = {}
        measurement_error: dict = {}
        for i, obs in enumerate(self.observations):
            tag = obs.label or f"obs{i}"
            T_i = m.parameter(f"T_{i}", obs.T)
            # per-observation electrode potential: point the faradaic steps at
            # this condition's U before their rate expressions are built below.
            U_i = m.parameter(f"U_{i}", obs.U)
            for rxn in mkm.reactions:
                if rxn.is_electrochemical:
                    rxn._U_param = U_i
                    rxn._F = mkm.F
            conc = {
                g: m.parameter(f"P{i}_{_safe(g.name)}", float(obs.pressures.get(g, 0.0)))
                for g in mkm.gas_species
            }
            theta = {a: m.continuous(f"th{i}_{_safe(a.name)}", lb=0.0, ub=1.0) for a in mkm.adsorbates}
            free = {s: m.continuous(f"fr{i}_{_safe(s.name)}", lb=0.0, ub=1.0) for s in mkm.sites}

            for a in mkm.adsorbates:
                r = net_rate(a, mkm.reactions, conc, theta, free, T_i, mkm.R, mkm.Tref)
                m.subject_to(r == 0.0, name=f"ss{i}_{_safe(a.name)}")
            for s in mkm.sites:
                m.subject_to(site_balance_residual(mkm, s, theta, free) == 0.0, name=f"site{i}_{_safe(s.name)}")

            responses[tag] = net_rate(
                obs.response, mkm.reactions, conc, theta, free, T_i, mkm.R, mkm.Tref
            )
            measurement_error[tag] = float(obs.sigma)

        em = ExperimentModel(
            model=m,
            unknown_parameters=unknown,
            design_inputs={},
            responses=responses,
            measurement_error=measurement_error,
        )
        em._log_names = log_names  # carried for back-transform
        return em


def fit_kinetics(
    mkm: MicrokineticModel,
    observations: list[Observation],
    fit: list[FitParam],
    solver_options: dict | None = None,
) -> MKMEstimationResult:
    """Estimate kinetic/thermodynamic constants from rate data across conditions.

    Parameters
    ----------
    mkm : MicrokineticModel
        The mechanism (species + reactions). Constants not in ``fit`` are held
        at their declared values.
    observations : list of Observation
        Measured turnover frequencies at distinct operating conditions.
    fit : list of FitParam
        Which constants to estimate, with bounds.
    solver_options : dict, optional
        Forwarded to discopt's solve (e.g. ``{"nlp_solver": "pounce"}``).

    Returns
    -------
    MKMEstimationResult
        Estimates, standard errors and 95% confidence intervals in physical
        units, plus the raw discopt ``EstimationResult``.
    """
    experiment = _MKMExperiment(mkm, observations, fit)
    data = {(o.label or f"obs{i}"): float(o.value) for i, o in enumerate(observations)}

    raw = estimate_parameters(experiment, data, solver_options=solver_options)

    # back-transform log-fit parameters to physical units (delta method)
    log_names = {fp.resolved_name() for fp in fit if fp.is_log()}
    raw_se = raw.standard_errors
    raw_ci = raw.confidence_intervals
    params, ses, cis = {}, {}, {}
    for name in raw.parameter_names:
        u = raw.parameters[name]
        if name in log_names:
            val = float(np.exp(u))
            params[name] = val
            ses[name] = val * float(raw_se[name])  # se(exp(u)) ~ exp(u) se(u)
            lo, hi = raw_ci[name]
            cis[name] = (float(np.exp(lo)), float(np.exp(hi)))
        else:
            params[name] = float(u)
            ses[name] = float(raw_se[name])
            cis[name] = tuple(float(x) for x in raw_ci[name])

    return MKMEstimationResult(
        parameters=params,
        std_errors=ses,
        confidence_intervals=cis,
        objective=float(raw.objective),
        n_observations=int(raw.n_observations),
        raw=raw,
    )
