# discopt-mkm for agents

A microkinetic-modeling toolkit. Describe a catalytic mechanism as **data** and
get **structured** results: steady-state coverages, turnover frequency, Campbell
degree of rate control, apparent orders/activation energy, stoichiometric
structure, and an HTML report.

Two ways to drive it:
- **Python**: `from discopt.mkm import agent` — each function takes a spec dict, returns JSON-able data.
- **MCP**: run `discopt-mkm-mcp` (needs `uv sync --extra mcp`); tools mirror the `agent` functions.

Do **not** hand-write the operator DSL (`m.step(CO + s >> COs)`); always pass a spec.

## The spec

```yaml
name: co_oxidation
T: 500                 # temperature
R: 8.617e-5            # gas constant — MUST match your energy units (see Units)
Tref: 298.15
sites:
  - {name: "*", density: 1.0}
gas:
  - {name: CO,  H: 0.0,  S: 0.0020}     # H, S, optional Cp (constant-Cp thermo)
  - {name: O2,  H: 0.0,  S: 0.0021}
  - {name: CO2, H: -3.0, S: 0.0023}
adsorbates:
  - {name: "CO*", site: "*", H: -0.8, S: 0.0005}
  - {name: "O*",  site: "*", H: -0.3, S: 0.0005}
reactions:
  - {equation: "CO + * <=> CO*",        A: 1e4, Ea: 0.0}    # Arrhenius; reverse from thermo
  - {equation: "O2 + 2 * <=> 2 O*",     A: 1e4, Ea: 0.0}
  - {equation: "CO* + O* -> CO2 + 2 *", A: 1e8, Ea: 0.7}    # '->' = irreversible
reactor: {type: differential, pressures: {CO: 1.0, O2: 0.5, CO2: 0.0}}
```

Reaction strings: `<=>` reversible, `->` irreversible; terms are `coeff species`
(coefficient optional), and the site/adsorbate `*` is allowed.

Per-reaction kinetics, choose one:
- `A`, `Ea` — Arrhenius; reverse rate derived from species thermodynamics.
- `kf`, `Keq` — explicit constants (e.g. from a DFT/SI table); reverse = `kf/Keq`.
- `equilibrated: true` — quasi-equilibrium (give `Keq` or rely on thermo); for fast steps.
- add `irreversible: true` to force `kr=0`; `alpha: <0..1>` for a BEP coverage-dependent barrier.
  `alpha` only has an effect when lateral `interactions` are registered (that is the only source of the
  coverage-dependent reaction energy the BEP shift acts on); set on a step with no interactions it is
  inert and the model build emits a warning.
- electrochemical step: `n_electrons: <int>` (electrons consumed forward; reduction positive) and
  `beta: <0..1>` (transfer coefficient). The free energy shifts by `n_electrons*F*U` and the forward
  barrier by `beta*n_electrons*F*U`; set model-level `U` (volts) and `F` (1.0 for eV/V, 96485 for J/mol).
  The electrochemical observables are exposed as agent/MCP tools (see below); `discopt.mkm.electrochem`
  has the underlying object-level functions plus the descriptor volcano and cyclic voltammetry.

Optional: `interactions: [{a: "CO*", b: "O*", eps: 0.1}]` (lateral interactions),
per-species `composition: {C: 1, O: 1}` (else inferred from the name),
per-species `thermo: {type: nasa7, low: [...], high: [...]}` or `{type: shomate, coeffs: [A..H]}`.
Reactor types: `differential` (fixed `pressures`), `cstr` (`inlet`, `tau`, `cat_density`), `batch`
(`initial`). The reactor fields are type-specific and validated: a `differential` reactor takes
`pressures` (gas species only), a `cstr` takes `inlet`, a `batch` takes `initial` — mixing them
(e.g. a `cstr` with `pressures`) is a clear error, not a silent all-zero feed.

Specs are strict: an unknown key anywhere (a typo like `reactons:` or `n_electron:`, or a field on
the wrong object) is rejected with a validation error rather than silently dropped — `validate(spec)`
surfaces these.

## Tools / functions

- `validate(spec)` → `{ok, errors, warnings, info}`. **Call first**: parses, checks element + site
  mass balance, reports the overall reaction. Fix `errors` before solving.
- `structure(spec)` → overall reaction, independent reactions, conservation laws (no solve).
- `solve(spec, coordinates="linear")` → coverages, gas, rates, status.
- `degree_of_rate_control(spec, target)` → `{step: X_RC}`, sums to ~1 (which steps control the rate).
- `apparent_kinetics(spec, target)` → apparent reaction orders + apparent activation energy.
- `analyze(spec, target)` → all of the above in one call. **Prefer this.**
- `report(spec, target, path=...)` → self-contained HTML report.

Electrochemistry (specs with faradaic steps; set model-level `U`, `F`):
- `current(spec)` → `{U, current, status}`, the faradaic current `j = F*sum(n_j*r_j)` per active site
  (reduction positive). Solves the steady state first.
- `tafel_slope(spec)` → `{U, tafel_slope, transfer_coefficient, status}`; evaluate in the Tafel region
  (away from the equilibrium potential where `j` crosses zero, else it raises).
- `che_diagram(spec, U=None)` → `{U, steps, delta_g, cumulative}`, the CHE free-energy diagram along the
  faradaic steps (no solve needed).
- `limiting_potential(spec)` → `{limiting_potential}` (reduction mechanisms only; raises for oxidation).

`target` is a species name (a gas product); it defaults to a net-produced gas species.

## Decision guide

- **Reactor**: rate/DRC at fixed conditions → `differential`. Conversion with flow → `cstr`. Time
  evolution → `batch` (use `solve_transient`, not these tools; a steady-state solve on a `batch`
  reactor raises — a closed batch has no nontrivial steady state).
- **coordinates**: default `"linear"`. Use `"log"` for stiff near-equilibrium mechanisms where
  coverages span many orders of magnitude (e.g. water-gas shift) — `solve`/`analyze` auto-warm-start
  it (reactor-aware: the warm start uses the fixed pressures / CSTR inlet / bulk as appropriate, and
  the flow-reactor gas balance is imposed, so `"log"` works with `cstr` too).
- **Fast steps you want to assume equilibrated** → `equilibrated: true` (removes stiffness; only `Keq`
  needed). Those steps then report degree of rate control = 0 by construction.
- If `degree_of_rate_control` returns `drc: null`, a coverage is pinned near 0/1 — often a tiny
  coverage falling below the default `active_tol` (1e-3) in linear coordinates, which then reads as
  bound-active. Retry with `coordinates="log"` (coverages are `exp(z)`, never near the linear
  0-bound), which is the robust fix for tiny-coverage DRC.

## Units (important)

The package is unit-agnostic; **you** must keep `H`, `S`, `Cp`, `Ea` and `R` consistent:
- eV and eV/K → `R = 8.617e-5` (typical for DFT energies)
- J/mol and J/(mol·K) → `R = 8.314`
Apparent `Ea` comes back in the same energy units as `R`.

## Example (Python)

```python
from discopt.mkm import agent
result = agent.analyze(spec, target="CO2")
# -> {structure, steady_state, tof, drc, apparent_orders, apparent_Ea}
```
