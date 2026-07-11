# discopt-mkm

[![tests](https://github.com/jkitchin/discopt-mkm/actions/workflows/tests.yml/badge.svg)](https://github.com/jkitchin/discopt-mkm/actions/workflows/tests.yml)

A microkinetic modeling (MKM) plugin for the
[discopt](https://pypi.org/project/discopt/) modeling language. Declare
heterogeneous-catalysis species and reactions with an operator syntax; the
plugin assembles the mole balances and rate laws automatically and solves them
at steady state or transiently. Temperature drives both the kinetics (Arrhenius)
and the thermodynamics (equilibrium constants) from a single consistent source,
and Campbell degree of rate control is computed by automatic differentiation.

## Concepts

- **Species** carry thermodynamics: by default a constant enthalpy `H`, entropy
  `S`, and optional heat capacity `Cp` (giving `H(T) = H + Cp(T−Tref)`,
  `S(T) = S + Cp·ln(T/Tref)`). For wider temperature ranges pass a
  `thermo=` model instead — `NASA7(...)`, `Shomate(...)`, or
  `GeneralThermo(h=..., s=...)` — which supplies `G(T)` directly and flows into
  `K_eq(T)`, the reverse rates, and the energy balance. `H` may also be a
  callable `H(theta)` for coverage-dependent (lateral-interaction) energetics.
  Gas species use concentration / partial pressure as their activity; adsorbates
  use coverage; bare sites use free-site coverage.
- **Reactions** are written with operators: `reactants >> products`, with
  stoichiometric coefficients via `2 * site`. Two ways to give kinetics:
  - **Arrhenius** (default): `m.step(rxn, A=..., Ea=...)`. The forward rate is
    `k_f = A * exp(-Ea / (R T))` and the **reverse rate is derived** as
    `k_r = k_f / K_eq`, `K_eq = exp(-dG_rxn / (R T))` from the species
    thermodynamics — thermodynamic consistency is structural.
  - **Explicit rate**: `m.step(rxn, kf=..., Keq=...)`, for constants transcribed
    from a DFT/SI table. The reverse rate is `kf / Keq`. No species thermo
    needed for that step.
  - **Irreversible**: add `irreversible=True` to set the reverse rate to exactly
    zero. No `K_eq`/reverse expression is built — an Arrhenius irreversible step
    needs no product thermodynamics, and an explicit one needs only `kf`. This is
    cleaner (and, for Arrhenius, numerically safer) than faking irreversibility
    with a huge `K_eq`, which can overflow `exp(-dG/RT)` in the AD Jacobian.
- **Reactors** decide the gas treatment: `DifferentialReactor` (fixed gas
  partial pressures, the natural setting for steady-state degree of rate
  control), `CSTR` (gas balance with in/outflow), and `Batch` (transient only).

## Units

Units are the caller's responsibility. The energies (`H`, `S`, `Cp`, `Ea`) and
the gas constant `R` must be mutually consistent. The default `R = 8.617e-5`
matches energies in **eV** and entropies in **eV/K**; pass `R = 8.314` for
J/mol and J/(mol·K).

## Example

```python
import discopt.mkm as mk

m   = mk.Model("co_ox", T=500)                 # R defaults to eV/K
s   = m.site("Pt", density=1.0)
CO  = m.gas("CO",  H=0.0,  S=0.0020)
O2  = m.gas("O2",  H=0.0,  S=0.0021)
CO2 = m.gas("CO2", H=-3.0, S=0.0023)
COs = m.adsorbate("CO*", site=s, H=-0.8, S=0.0005)
Os  = m.adsorbate("O*",  site=s, H=-0.3, S=0.0005)

m.step(CO + s >> COs,            A=1e4, Ea=0.0)
m.step(O2 + 2 * s >> 2 * Os,     A=1e4, Ea=0.0)
m.step(COs + Os >> CO2 + 2 * s,  A=1e8, Ea=0.7)

reactor = mk.DifferentialReactor({CO: 1.0, O2: 0.5, CO2: 0.0})

# steady state
sol = mk.solve_steady_state(m, reactor)
print(sol.coverage(COs), sol.coverage(Os), sol.free_coverage(s))
print("TOF:", sol.production_rate(CO2))

# degree of rate control (Campbell) for the CO2 production rate
X = mk.degree_of_rate_control(sol, species=CO2)   # {reaction: X_RC}, sums to ~1

# thermodynamic rate control w.r.t. each species free energy
Xt = mk.thermo_rate_control(sol, species=CO2)

# transient
tr = mk.solve_transient(m, reactor, t_span=(0.0, 5.0),
                        theta0={COs: 0.0, Os: 0.0}, nfe=30, ncp=3)
print(tr.final_coverage(COs))
```

## Parameter estimation

Fit kinetic/thermodynamic constants to measured turnover frequencies across
operating conditions. Unknown constants and every condition's coverages are
solved simultaneously (all-at-once weighted least squares), and discopt returns
the Fisher information, covariance, and confidence intervals. Pre-exponentials
are fit in log space by default and reported back in physical units.

```python
obs = [
    mk.Observation(response=CO2, value=tof, T=T, pressures={CO: 1.0, O2: 0.5, CO2: 0.0}, sigma=0.02)
    for T, tof in zip(temperatures, measured_tofs)
]
fit = [
    mk.FitParam(surface_step, "A",  lb=1e6, ub=1e10),   # log-fit pre-exponential
    mk.FitParam(surface_step, "Ea", lb=0.3, ub=1.2),
]
result = mk.fit_kinetics(m, obs, fit)
print(result.parameters, result.confidence_intervals)
```

### Transient (time-series) fits

`fit_kinetics_transient` is the transient counterpart: fit constants to
measured *time series* (e.g. a product rate under a PRBS/pulse-modulated feed).
Each run's coverage ODEs are transcribed to collocation constraints
(`discopt.dae`), with element boundaries aligned to the switching times of the
piecewise-constant inputs, and one NLP solves for the shared constants and all
coverage trajectories simultaneously: the all-at-once alternative to the
integrate-inside-`curve_fit` shooting pattern of `10_prbs_fitting.ipynb`. The
model response is interpolated to the exact measurement times (irregular
sampling needs no grid alignment), trajectories are warm-started from a numeric
implicit-Radau integration at the nominal constants, and multi-run fits at
different temperatures share `A`/`Ea` for Arrhenius-consistent estimates.

```python
runs = [
    mk.TransientRun(
        response=CO2, t=t_meas, y=rate_meas, T=T,          # measured r_CO2(t)
        pressures={CO: (t_switch, P_co_values),            # piecewise-constant PRBS input
                   O2: 0.5, CO2: 0.0},
        sigma=0.01, t_span=(0.0, 2.0),
    )
    for T, t_meas, rate_meas in experiments
]
result = mk.fit_kinetics_transient(m, runs, fit, nfe=40)
result.predictions["run0"]           # fitted response at the measurement times
result.trajectories["run0"]["CO*"]   # fitted coverage trajectory
```

A `GasSpecies` response means its net production rate was measured; an
`Adsorbate` response means its coverage was. Confidence intervals use the full
trajectory sensitivity `dr/du = ∂r/∂u + (∂r/∂x)(dx/du)` (implicit function
theorem on the collocation system), so constants that act on the response only
through the coverage dynamics, the typical transient-fit case, are covered.

## Quasi-equilibrium approximation

Mark the fast steps `equilibrated=True` to apply the quasi-equilibrium
approximation: those steps' rate laws are dropped and replaced by an
equilibrium-quotient constraint `Q = K_eq` (their rate of progress becomes an
unknown extent). Only `K_eq` is needed — give it explicitly or let it come from
thermodynamics; the fast rate constants drop out.

```python
m.step(CO + s >> COs, Keq=215, equilibrated=True)     # quasi-equilibrated
m.step(COs + Os >> CO2s + s, kf=2.05e5, Keq=1.03e3)   # the kinetic step
```

This is both a modeling simplification (assume an RDS, equilibrate the rest) and
a *numerical* one: writing `Q = K_eq` instead of `kf·Π − kr·Π = 0` removes the
near-equilibrium cancellation, so a stiff network like water-gas shift solves in
plain linear coordinates with no warm start. The
`examples.water_gas_shift_qea` mechanism reproduces the full-SSA rate to ~0.01%.

Two notes: a quasi-equilibrated step reports a kinetic degree of rate control of
exactly 0 (the QEA asserts it is not rate-controlling), and for DRC of systems
with extremely small coverages (e.g. θ ~ 1e-12) pass a small `active_tol`
(below the smallest coverage) so the L3 sensitivity does not mistake a tiny
coverage for a bound-active one.

## Stiff mechanisms and log coverages

Near-equilibrium mechanisms (e.g. water-gas shift) have coverages spanning many
orders of magnitude and a net rate that is a huge cancellation of one-way step
rates — finite differences over a re-solved steady state cannot compute the DRC
at all. Solve such systems in **log coordinates** with a numeric warm start:

```python
from discopt.mkm import numeric
theta0, _ = numeric.steady_state_numeric(m, reactor.pressures, T, theta0=seed)
sol = mk.solve_steady_state(m, reactor, coordinates="log", theta0=theta0, log_box=8.0)
X = mk.degree_of_rate_control(sol, species=H2)   # analytic sensitivity, sums to 1
```

`coordinates="log"` solves for `z = ln(theta)` (so coverages stay positive and
well scaled) inside a box centered on the warm start, with a regularizer that
selects the physical root. `discopt.mkm.numeric` is a pure-NumPy/scipy rate
evaluator and steady-state root-find used for the warm start and as an
independent check.

## Agent / declarative interface

For LLM agents (and anyone who prefers data over the operator DSL), a mechanism
can be a **spec** — a dict/YAML with string reaction equations — and results come
back as **JSON**. See [`AGENTS.md`](AGENTS.md) for the full guide.

```python
from discopt.mkm import agent
spec = {
  "name": "co_ox", "T": 500, "R": 8.617e-5,
  "sites": [{"name": "*", "density": 1.0}],
  "gas": [{"name": "CO", "H": 0, "S": 0.0020}, {"name": "CO2", "H": -3.0, "S": 0.0023}, ...],
  "adsorbates": [{"name": "CO*", "site": "*", "H": -0.8, "S": 0.0005}, ...],
  "reactions": [
    {"equation": "CO + * <=> CO*", "A": 1e4, "Ea": 0.0},
    {"equation": "CO* + O* -> CO2 + 2 *", "A": 1e8, "Ea": 0.7}],   # '->' = irreversible
  "reactor": {"type": "differential", "pressures": {"CO": 1.0, "O2": 0.5, "CO2": 0.0}},
}
agent.validate(spec)              # parse + element/site mass balance + structure (call first)
agent.analyze(spec, "CO2")        # structure + steady state + DRC + apparent kinetics, all JSON
```

The same functions are exposed as an **MCP server** (`mcp.ModelSpec` JSON schema,
structured returns): `uv sync --extra mcp`, then run `discopt-mkm-mcp`. Tools:
`validate`, `structure`, `solve`, `degree_of_rate_control`, `apparent_kinetics`,
`analyze`, `report`, `spec_schema`. `mk.from_spec` / `mk.from_yaml` / `mk.to_spec`
convert between specs and models.

## Reactors

The reactor decides how the gas phase is treated; the same species/reaction
core is reused throughout.

- `DifferentialReactor` — gas at fixed partial pressures (the natural setting for
  steady-state DRC).
- `CSTR` — well-mixed gas balance with in/outflow (steady or transient).
- `Batch` — closed, gas evolves only by reaction (transient).
- `solve_pfr` — plug-flow reactor as a **spatial DAE**: gas concentrations are
  axial states, surface coverages are quasi-steady-state algebraic variables.

Any of these can be run **non-isothermal** by passing an `EnergyBalance`:
temperature becomes an unknown (a variable for a steady CSTR, a spatial/time
state for a PFR/batch) coupled to the kinetics and thermodynamics through an
adiabatic (or heated) energy balance.

```python
from discopt.mkm.energy import EnergyBalance
sol = mk.solve_steady_state(m, cstr, energy=EnergyBalance(T_in=500.0))   # adiabatic CSTR
print(sol.temperature())
pfr = mk.solve_pfr(m, feed, length=1.0, velocity=1.0, cat_density=0.02,
                   energy=EnergyBalance(T_in=500.0))                      # adiabatic PFR
```

## Worked examples

Executed Jupyter notebooks in `examples/`:

- `01_co_oxidation_drc.ipynb` — steady state, degree of rate control, thermodynamic RC.
- `02_transient.ipynb` — collocation transient converging to the steady state.
- `03_estimation.ipynb` — fit `A`/`Ea` to synthetic TOF data with confidence intervals.
- `04_water_gas_shift_drc.ipynb` — the WGS DRC from Yang, Achar & Kitchin
  (AIChE J. 2022), reproduced via explicit-rate steps and a log-coverage solve.
- `05_pfr.ipynb` — plug-flow reactor axial profiles and conversion.
- `06_nonisothermal.ipynb` — adiabatic CSTR and PFR with the energy balance.
- `07_rendering_and_structure.ipynb` — HTML/LaTeX rendering, stoichiometric
  matrix, reaction routes, conservation laws.
- `08_apparent_kinetics.ipynb` — apparent orders, apparent barrier, and a
  symbolic Langmuir-Hinshelwood lumped rate.
- `09_figures_and_report.ipynb` — free-energy diagram, reaction-network graph,
  and the one-call HTML mechanism report.
- `10_prbs_fitting.ipynb` — fit kinetic constants to a **transient** experiment:
  *independent* pseudo-random binary-sequence (PRBS) pulses of CO and O2 at
  *irregular* sample times. Forward model uses POUNCE's implicit Radau integrator
  (`numeric.integrate_coverages`); the fit uses `pounce.curve_fit` (confidence
  intervals + parameter correlation), with `pounce.jax.odeint` for analytic
  trajectory sensitivities. Identifies the adsorption rate constant that
  steady-state data cannot.
- `11_mechanism_discovery.ipynb` — map a potential energy surface with POUNCE's
  `reaction_network` and turn the states + barriers into an mkm spec
  (`mk.discover_mechanism`), then render its free-energy diagram.
- `12_multitemperature_fit.ipynb` — separate `A` from `Ea` by fitting transient
  data at several temperatures with a JAX model and `pounce.curve_fit(jac="jax")`
  (analytic Jacobian, confidence intervals); single-temperature data leaves `Ea`
  undetermined.
- `13_selectivity.ipynb` — a branching mechanism (one intermediate, two
  product-forming steps): compute selectivity, show the activity/selectivity
  tradeoff as oxygen pressure rises, and find the **degree of selectivity
  control** `∂ln S/∂ln k` from the same AD machinery as the degree of rate
  control (`mk.examples.selective_oxidation`).
- `14_mechanism_selection.ipynb` — recover a minimal mechanism from an
  over-complete candidate set four ways (DRC flux-screen, structural search, the
  flux **MILP**, and a symbolic rate law); see Mechanism selection below.
- `15_tpd.ipynb` — temperature-programmed desorption: predict a first-order
  spectrum (coverage-independent peak) and fit `ν` and `Eₐ` to second-order data
  at several initial coverages (the peak shifts with coverage) via
  `pounce.curve_fit` with an analytic Jacobian.
- `16_orr_electrochemistry.ipynb` — 4-electron oxygen reduction: CHE free-energy
  diagram vs potential, polarization curve, Tafel slope and transfer coefficient
  by automatic differentiation, and the discopt-optimized activity volcano (see
  Electrochemistry below).
- `17_rde_outer_sphere.ipynb` — a rotating disk electrode: the outer-sphere couple
  `Fe²⁺/Fe³⁺` (no adsorbed intermediate) with a Levich diffusion layer, showing
  the kinetic-to-transport crossover, the Levich `ω¹ᐟ²` limiting current, and a
  Koutecký-Levich plot that separates kinetics from mass transport.
- `18_cyclic_voltammetry.ipynb` — the classic diffusion-controlled "duck": a
  cyclic voltammogram from transient semi-infinite diffusion with a Butler-Volmer
  boundary condition, showing reversible/quasi-reversible/irreversible shapes, the
  peak separation, and Randles-Sevcik `i_p ∝ √v` scaling.
- `19_transient_fitting.ipynb` — the PRBS experiment of notebook 10 refit with
  the packaged **simultaneous** API `fit_kinetics_transient`: one collocation
  NLP for the constants and the coverage trajectories, physical-unit confidence
  intervals from the full trajectory sensitivity, and a side-by-side comparison
  (estimates and wall time) with the shooting fit on the same data.

## Mechanism selection

Combinatoric generators (RMG and similar) produce mechanisms that are too large
and full of irrelevant steps, because relevance is decided *after* enumeration by
rate-rule thresholds. `discopt.mkm.select` decides relevance *during* selection,
using exact sensitivities and a parsimony objective. Given an over-complete
candidate mechanism and turnover data, three complementary methods recover the
minimal mechanism (`mk.examples.overcomplete_co_oxidation` adds Eley-Rideal,
redundant-route, and spectator decoys to the three-step CO oxidation):

```python
conds = [{"CO": 1.0, "O2": 0.5}, {"CO": 2.0, "O2": 0.5}, {"CO": 0.5, "O2": 1.0}]

mk.reduce_by_drc(m, conds, "CO2")              # flux screen + degree of rate control
mk.select_subgraph(m, conds, "CO2", data)      # smallest sub-mechanism fitting the data
mk.pareto_subgraph(m, conds, "CO2", data)      # accuracy-vs-size front (knee = minimal)
mk.fit_rate_law(conds, data, ["CO", "O2"])     # compact closed-form rate law
```

`reduce_by_drc` removes steps that carry negligible flux (provably safe: a
zero-flux step cannot change the steady state) and ranks the survivors by the
exact Campbell degree of rate control. `select_subgraph` finds the fewest steps
that reproduce the data; its `engine="milp"` option poses this as a true MILP for
discopt (sub-second) by working in **flux space** rather than coverage space: the
steady-state balances on the rates of progress are linear, a binary gates each
step, and a capacity bound (a flux cannot exceed its rate constant) rejects
kinetically incapable shortcut routes. The default `engine="greedy"` searches
over structure and scores each candidate with an exact numerical steady-state
solve; `engine="minlp"` expresses the full nonlinear formulation but is
intractable beyond a few steps, which is exactly why the flux MILP exists.
`fit_rate_law` regresses a Langmuir-Hinshelwood or power-law rate from a template
library, scored by AIC.

## Electrochemistry

The `discopt.mkm.electrochem` subpackage adds electrocatalysis by treating the
electrode potential `U` as a second global driving variable, the exact analog of
temperature. Mark a step faradaic with `n_electrons` (and a transfer coefficient
`beta`); its reaction free energy then shifts by `n·F·U` (computational hydrogen
electrode) and its forward barrier by the Butler-Volmer fraction `beta·n·F·U`.
Because the reverse rate is derived from `K_eq`, detailed balance `k_f/k_r = K_eq`
holds at every potential. Default units are eV with `U` in volts and `F = 1`.

```python
import discopt.mkm.electrochem as ec

m, reactor = ec.orr_4e(descriptor=0.9)          # 4-electron ORR from one descriptor
m.U = 0.7                                        # set the electrode potential and re-solve
sol = mk.solve_steady_state(m, reactor, theta0=warm)

ec.current_density(sol)                          # j = F Σ n_j r_j
ec.tafel_slope(sol)                              # dU/d log10|j|  (~ -118 mV/dec)
ec.apparent_transfer_coefficient(sol)            # (RT/F) d ln|j|/dU  (= beta)
ec.degree_of_current_control(sol)                # which step limits the current

ec.che_free_energies(m, U=1.23)                  # CHE free-energy diagram
ec.limiting_potential(m)                         # thermodynamic onset / overpotential
ec.optimize_descriptor(ec.ORR_SCALING, ec.ORR_N_ELECTRONS, bounds=(0.3, 1.5), U_eq=1.23)
```

The Tafel slope and transfer coefficient come from the same automatic-
differentiation sensitivity that gives the apparent activation energy (the
potential plays temperature's role); `optimize_descriptor` finds the activity-
volcano peak as a small linear program. A potential is set on the model (`m.U`)
and swept by re-solving, like temperature; `Observation(..., U=...)` carries the
potential for fitting a polarization curve with `fit_kinetics`.

**Mass transport.** A `RotatingDiskElectrode` reactor (or the general
`MassTransferReactor`) couples a solution species' surface activity to the bulk
through a Levich mass-transfer coefficient `k_m = 0.62 D^(2/3) ν^(-1/6) ω^(1/2)`,
making the surface activity an unknown set by `k_m (C_bulk − C_surface) + r = 0`.
The steady state then captures the kinetic-to-mass-transport crossover (the
Koutecký-Levich relation `1/j = 1/j_kinetic + 1/j_Levich`) with no new solver.
**Outer-sphere** electron transfers carry no adsorbed intermediate, so the model
is just the solution species and one faradaic step (`Fe²⁺ → Fe³⁺ + e⁻`); with the
RDE reactor this gives the full mixed kinetic/transport polarization curve of a
soluble redox couple. For the *transient* counterpart,
`electrochem.cyclic_voltammogram` simulates a cyclic voltammogram (semi-infinite
diffusion with a Butler-Volmer boundary condition), the classic diffusion-
controlled "duck" with Randles-Sevcik `i_p ∝ √v` scaling.

## How it works

Steady state is posed as a feasibility NLP (`minimize 0` subject to
`net_rate(adsorbate) == 0`, site conservation, and any reactor gas balance) and
solved with discopt's default POUNCE backend (a pure-Rust Ipopt port, no cyipopt
needed) through `differentiable_solve_l3`. That same solve yields the implicit sensitivity
`dx*/dp` of the steady state with respect to every kinetic/thermodynamic
parameter. Degree of rate control is the chain-rule total derivative of an
output rate, `dr/dp = (dr/dx)(dx*/dp) + dr/dp`, with the perturbation taken on
the forward rate constant of each step (`A_i` for Arrhenius steps, `kf_i` for
explicit-rate steps). That scales `k_f` and the derived `k_r` together at fixed
`K_eq`, exactly satisfying Campbell's definition.

Transient solves use discopt's orthogonal-collocation DAE builder
(`discopt.dae`) and reuse the identical rate core.

## Caveats

- Degree of rate control needs the L3 implicit sensitivities. When a coverage
  saturates (θ → 0 or 1) the KKT system can be singular and discopt returns no
  sensitivity matrix; `degree_of_rate_control` then raises
  `SensitivityUnavailable`. Use `coordinates="log"` (and a warm start) rather
  than forcing a linear solve.
- `fit_kinetics` returns discopt's Fisher-information-based covariance, which
  uses the explicit response Jacobian; the point estimates are exact (from the
  simultaneous least-squares solve) while the reported standard errors are this
  FIM approximation. `fit_kinetics_transient` improves on this with the full
  (implicit-function-theorem) trajectory sensitivity, since a transient
  response is mostly sensitive to the constants through the coverage
  trajectory; a singular-FIM warning there indicates a constant that is
  genuinely not identifiable from the data.
- `fit_kinetics_transient` solve time is set by the number of *input switches*,
  not the number of measurements: every switch is a mandatory element boundary,
  and discopt's Hessian-sparsity detection currently records nonlinear coupling
  at whole-array granularity for vectorized DAE constraints, so the NLP
  iteration cost grows superlinearly with the element count. Runs with up to
  ~10 switches solve in seconds; a long PRBS train (30+ switches, ~120
  elements) takes minutes. Split long sequences into several runs (they share
  the fitted constants) until that upstream limitation is lifted.
- A strongly exothermic, high-activation-energy adiabatic PFR develops a sharp
  ignition front that orthogonal collocation resolves poorly; keep the per-pass
  temperature rise modest (lower catalyst loading / shorter reactor) or refine
  the mesh. `solve_pfr` warm-starts from a numeric inlet steady state, so the
  mechanism should not be so stiff that the inlet root-find itself stalls.
- The analysis layer (`src/discopt/mkm/analysis/sensitivity.py`) calls
  underscore-prefixed discopt internals. Pin the discopt version.

## Apparent orders, apparent barriers, lumped rates

Experimental kinetics is usually reported as apparent reaction orders and an
apparent activation energy — the local power-law/Arrhenius fit of the overall
rate. Both are logarithmic sensitivities of the steady-state turnover frequency,
computed analytically by the same implicit differentiation as the DRC (so they
propagate through the coverages — they are the true lumped descriptors, not the
elementary values):

```python
from discopt.mkm.analysis import apparent_orders, apparent_activation_energy
sol = mk.solve_steady_state(m, mk.DifferentialReactor({CO: 1.0, O2: 0.5, CO2: 0.0}))
apparent_orders(sol, CO2)              # {CO: -0.17, O2: +0.27}  (d ln r / d ln P_i)
apparent_activation_energy(sol, CO2)   # 0.68 eV  (R T^2 d ln r / dT)
```

For a quasi-equilibrium mechanism the coverages solve in closed form, so the
**overall (lumped) rate** can be derived symbolically (SymPy) — the classic
Langmuir-Hinshelwood result:

```python
rate, syms = mk.lumped_rate_expression(lh_model, B)   # -> kf*K*P_A / (1 + K*P_A)
```

For a general network there is no closed form; the numerical model *is* the
lumped rate and the apparent orders/barrier characterize it locally.

## Rendering & stoichiometric structure

Model, `Reaction`, and `SteadyStateSolution` objects render in Jupyter
(`_repr_html_`/`_repr_latex_`) and export with `to_latex()` / `to_html()`
(mechanism table, `align` block, mhchem-style equations). The stoichiometric
structure is exposed in `discopt.mkm.analysis.stoichiometry`:

```python
from discopt.mkm.analysis import stoichiometry as st
st.reaction_routes(m)          # Horiuti-Temkin routes -> the overall reaction
                               #   WGS: CO + H2O -> CO2 + H2;  CO-ox: 2CO + O2 -> 2CO2
st.independent_reactions(m)    # a maximal linearly independent subset
st.site_conservation_laws(m)   # recovers the site balance, e.g. Pt + CO* + O* = const
st.n_conservation_laws(m)      # number of invariants (sites + conserved elements)
```

Give species an elemental `composition` (explicitly, or via
`Model.infer_composition()` which parses formula-like names) and the conservation
laws separate cleanly into site and per-element balances, for *any* elements:

```python
m.infer_composition()                  # CO -> {C:1, O:1}, H2O -> {H:2, O:1}, ...
st.conserved_quantities(m)             # {"site:Pt": ..., "element:C": ..., "element:O": ...}
st.check_element_balance(m)            # [] if every reaction balances every element
                                       #   (otherwise (reaction, element, residual) — a typo catcher)
```

For figures and a one-call document there are `energy_diagram` (the free-energy
landscape along the mechanism, with smooth barrier connectors), `network_graph`
(static species/reaction bipartite graph, flux-weighted at a solution),
`interactive_network` (a draggable vis-network widget with hover tooltips and
flux-weighted edge thickness), `to_dot` (Graphviz source), and a self-contained
HTML report:

```python
mk.energy_diagram(m)                  # matplotlib Axes
mk.network_graph(m, solution=sol)
mk.interactive_network(m, solution=sol)         # inline in Jupyter; .save("net.html")
mk.report_html(m, solution=sol, target=CO2)     # mechanism + structure + figures
mk.write_report(m, "co_ox.html", solution=sol, target=CO2)  # + steady state + DRC
```

## Related work

Microkinetic modeling has a mature software ecosystem, and `discopt-mkm` is a
small, focused entry in it rather than a replacement. The closest relatives are
[CatMAP](https://doi.org/10.1007/s10562-015-1495-6) (Medford et al., 2015),
which pioneered descriptor-based microkinetic *mapping* of catalytic trends;
[MKMCXX](https://doi.org/10.1002/anie.201406521) (Filot et al., 2014), a fast
C++ engine that integrates the rate equations and reports reaction orders and
degrees of rate control; [OpenMKM](https://doi.org/10.1021/acs.jcim.3c00088)
(Medasani et al., 2023), a Cantera-based C++ multiscale simulator; and
[pMuTT](https://doi.org/10.1016/j.cpc.2019.106864) (Lym et al., 2020), the
Vlachos-group toolbox that turns ab-initio data into thermochemistry and kinetic
inputs. The general chemical-kinetics toolkit
[Cantera](https://doi.org/10.5281/zenodo.4527812) (Goodwin et al., 2021)
underlies several of these.

What `discopt-mkm` adds is a different *primitive*: the model is assembled
in-process on the [discopt](https://github.com/jkitchin/discopt) modeling layer,
and the **degree of rate control is computed by implicit automatic
differentiation through the steady state** (via JAX and discopt's sIPOPT
sensitivity) rather than by finite differences — following Yang, Achar & Kitchin
([2022](https://doi.org/10.1002/aic.17653)), which is the WGS example reproduced
here. The degree-of-rate-control concept itself is due to Stegelmann, Andreasen
& Campbell ([2009](https://doi.org/10.1021/ja9000097)) and Campbell
([2017](https://doi.org/10.1021/acscatal.7b00115)).

Full, DOI-verified BibTeX entries are in [`references.bib`](references.bib).

## Install (uv)

The project is managed with [uv](https://docs.astral.sh/uv/). `discopt-mkm` is a
`discopt` namespace plugin: it installs under the `discopt` package (as
`discopt.mkm`) and depends on `discopt>=0.5`, which is published to PyPI with
prebuilt wheels — no Rust toolchain needed for a plain install:

```bash
uv sync                  # create .venv, install discopt (from PyPI) + everything
uv sync --extra mcp      # ...also install the MCP server dependency
uv run pytest            # run the test suite
```

PyPI ships discopt wheels for CPython 3.10–3.13 on Linux and CPython 3.12 on
macOS, so use a 3.12 interpreter on macOS (`uv sync --python 3.12`) to avoid
building the Rust extension from source. To develop against a local `discopt`
checkout instead, add a `[tool.uv.sources]` override pointing `discopt` at your
editable clone of `github.com/jkitchin/discopt`.

`uv.lock` pins the full dependency set for reproducibility. Add packages with
`uv add <pkg>` (or `uv add --dev <pkg>` for tooling).

### Use it from an LLM agent (skill + MCP)

After `uv sync` the package installs two console scripts. Install the Claude skill
(auto-discovered usage guide) and register the MCP server in one go:

```bash
discopt-mkm-install-skill            # -> ~/.claude/skills/discopt-mkm/SKILL.md (user-wide)
discopt-mkm-install-skill --project  # -> ./.claude/skills/  (versioned with this repo)
discopt-mkm-install-skill --mcp      # also runs `claude mcp add discopt-mkm -- discopt-mkm-mcp`
```

Flags: `--user` (default) / `--project` choose the scope, `--force` overwrites,
`--mcp` also registers the MCP server. To register the server by hand:

```bash
claude mcp add discopt-mkm -- discopt-mkm-mcp
```

The skill teaches an agent the spec format and tool workflow (see
[`AGENTS.md`](AGENTS.md)); the MCP server exposes `validate`, `solve`, `analyze`,
`degree_of_rate_control`, `apparent_kinetics`, `structure`, `report`, and
`spec_schema` as callable tools.

## Tests

```bash
uv run pytest tests/
```

# Potential future work

It probably would be straightforward to develop an electrochemical version of this package. Please reach out if it is something you would be interested in collaborating on.