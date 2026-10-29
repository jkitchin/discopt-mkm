---
name: discopt-mkm
description: >-
  Build and analyze heterogeneous-catalysis microkinetic models — steady-state
  coverages, turnover frequency, Campbell degree of rate control, apparent
  reaction orders / activation energy, reaction routes and the overall reaction,
  lumped (Langmuir-Hinshelwood) rate laws, and HTML reports. Use whenever the
  user describes a catalytic reaction mechanism, surface kinetics, adsorbates and
  sites, coverages, rate-determining or rate-controlling steps, or asks to solve
  or analyze a microkinetic model.
license: MIT
---

# discopt-mkm

Describe a catalytic mechanism as **data** (a spec) and get **structured** results.
Use `from discopt.mkm import agent` (each function takes a spec dict, returns
JSON-able data) or the MCP tools of the same names. Do **not** hand-write the
operator DSL (`m.step(CO + s >> COs)`); always pass a spec.

Requires the `discopt-mkm` package (Python >= 3.12; pulls in `discopt[jax]>=0.11`
and `pounce-solver>=0.14`). If `import discopt.mkm` fails, install it with
`pip install git+https://github.com/jkitchin/discopt-mkm` (or
`pip install "discopt-mkm[mcp] @ git+https://github.com/jkitchin/discopt-mkm"` for
the MCP server, `discopt-mkm-mcp`).

## Spec

```yaml
name: co_oxidation
T: 500                 # temperature
R: 8.617e-5            # gas constant — MUST match your energy units (see Units)
Tref: 298.15
sites:      [{name: "*", density: 1.0}]
gas:        [{name: CO, H: 0.0, S: 0.0020}, {name: O2, H: 0.0, S: 0.0021}, {name: CO2, H: -3.0, S: 0.0023}]
adsorbates: [{name: "CO*", site: "*", H: -0.8, S: 0.0005}, {name: "O*", site: "*", H: -0.3, S: 0.0005}]
reactions:
  - {equation: "CO + * <=> CO*",        A: 1e4, Ea: 0.0}    # Arrhenius; reverse from thermo
  - {equation: "O2 + 2 * <=> 2 O*",     A: 1e4, Ea: 0.0}
  - {equation: "CO* + O* -> CO2 + 2 *", A: 1e8, Ea: 0.7}    # '->' = irreversible
reactor: {type: differential, pressures: {CO: 1.0, O2: 0.5, CO2: 0.0}}
```

Equations: `<=>` reversible, `->` irreversible; terms are `coeff species`
(coefficient optional), site/adsorbate `*` allowed.

Per-reaction kinetics (choose one): `A`+`Ea` (Arrhenius, reverse from species
thermo) · `kf`+`Keq` (explicit constants) · `equilibrated: true` (quasi-
equilibrium for fast steps; give `Keq` or rely on thermo). Add `irreversible:
true` to force `kr=0`, `alpha: <0..1>` for a BEP coverage-dependent barrier
(only active when lateral `interactions` are registered; otherwise inert, with a
build warning). Electrochemical step: `n_electrons: <int>` (consumed forward;
reduction positive) and `beta: <0..1>`; set model-level `U` (volts) and `F` (1.0
for eV/V, 96485 for J/mol).

Optional: `interactions: [{a: "CO*", b: "O*", eps: 0.1}]`, per-species
`composition: {C: 1, O: 1}` (else inferred from the name), per-species `thermo:
{type: nasa7, low: [...], high: [...]}` or `{type: shomate, coeffs: [A..H]}`.
Reactor types: `differential` (fixed `pressures`), `cstr` (`inlet`, `tau`,
`cat_density`), `batch` (`initial`). Reactor fields are type-specific: mixing them
(e.g. a `cstr` with `pressures`) is an error. Specs are strict — an unknown key
anywhere (a typo like `reactons:`) is a validation error, not silently dropped.

## Tools / functions (spec in, JSON out)

- `validate(spec)` → `{ok, errors, warnings, info}`. **Call first**: parses and
  checks element + site mass balance; fix `errors` before solving.
- `structure(spec)` → overall reaction, independent reactions, conservation laws.
- `solve(spec, coordinates="linear")` → coverages, gas, rates, status.
- `degree_of_rate_control(spec, target)` → `{step: X_RC}` (sums to ~1).
- `apparent_kinetics(spec, target)` → apparent orders + apparent activation energy.
- `analyze(spec, target)` → all of the above in one call. **Prefer this.**
- `report(spec, target, path=...)` → self-contained HTML report.

Electrochemistry (faradaic steps; needs model-level `U`, `F`):
- `current(spec)` → `{U, current, status}`, `j = F*sum(n_j*r_j)` per site.
- `tafel_slope(spec)` → `{U, tafel_slope, transfer_coefficient, status}`; use in the
  Tafel region (raises near the equilibrium potential).
- `che_diagram(spec, U=None)` → CHE free-energy diagram (no solve).
- `limiting_potential(spec)` → `{limiting_potential}` (reduction mechanisms only).

`target` is a gas-product species name; defaults to a net-produced gas species.

## Decision guide

- Reactor: fixed-condition rate/DRC → `differential`; conversion with flow →
  `cstr`; time evolution → `batch` (use `solve_transient`, not these tools; a steady-state
  solve on a `batch` reactor raises).
- `coordinates`: default `"linear"`; use `"log"` for stiff near-equilibrium
  mechanisms (e.g. water-gas shift) — `solve`/`analyze` auto-warm-start it.
- Fast steps you want to assume equilibrated → `equilibrated: true` (removes
  stiffness; reports degree of rate control = 0 for those steps by construction).
- If `degree_of_rate_control` returns `drc: null`, a coverage is pinned near 0/1
  (often a tiny coverage below `active_tol`) — retry with `coordinates="log"`.

## Units (important)

Unit-agnostic — keep `H, S, Cp, Ea, R` consistent: eV & eV/K → `R = 8.617e-5`;
J/mol & J/(mol·K) → `R = 8.314`. Apparent `Ea` returns in the energy units of `R`.
