# Module review: `discopt.mkm`

Scope: all of `src/discopt/mkm/` (~5,800 lines) reviewed for **correctness**,
**completeness**, and **performance**, cross-checked against the documented
behavior in `AGENTS.md` and the test suite. Findings marked *(verified)* were
reproduced by executing the failing path in this repo's venv, not just by
reading the code.

Test-suite baseline at `fe81188`: **94 passed, 4 failed/errored**, all four
from a single root cause (see T1).

---

## Round 2 — adversarial re-review of the fixes

After all round-1 fixes landed on `main`, a second correctness pass re-examined
the ~970 lines of new fix code (four independent reviewers, each verifying by
execution). **The fixes hold up**: no high- or medium-severity bug was found in
any of them. The highest-risk introductions were each verified numerically —
the M1 solution snapshot (0.0 DRC drift across a re-solve), the P1 shared
rate-expression cache, the P6 banded CV solve (max diff 0.0 vs a dense
reference), the H1 electrochemical reverse-rate shift (byte-matches the numeric
reference), the H7 estimator handle reset (recovers parameters exactly), and the
M12 temperature rescale.

Round 2 produced one genuine (pre-existing, low) bug plus small hardening:

- **formula.py** — `parse_formula("H2O-2")` returned `{H:2, O:2}`: a charge/config
  digit after a stripped `-` was absorbed as the previous element's subscript.
  The tokenizer now emits separators as their own tokens so they break
  element–digit adjacency (`{H:2, O:1}`). Fixed; regression test added.
- **numeric.py** — the warm-start physical-root check rejected coverages `< 0`
  but not `> 1`; made symmetric (rejects out-of-`[0,1]` roots, clamps tiny
  excursions). Hardening.
- **estimate.py** — documented that explicit-rate (`kf`/`Keq`) fits should pass
  an `Observation.theta0` warm start (they otherwise risk an `unbounded` solve).

Follow-up cleanup: the M5 least-squares residual check turned out to sit on a
fully dead path — `method="least_squares"` (and the `auto` fallback to it)
always returns `unbounded` from the discopt backend, so the check was never
reached. Rather than keep an absolute-threshold guard on unreachable code, the
whole least-squares machinery was removed: `method` is now `{"auto",
"feasibility"}` (equivalent), and a `method="least_squares"` call — reachable
via the agent/MCP `method` argument — now raises a clear error pointing at
`coordinates="log"` instead of the confusing `unbounded`. The M12 `T ≤ 4·T_nom`
cap and the H5 `flux_tol` normalization were left as-is: both fail *loudly*
rather than returning wrong results.

---

## Round 3 — under-reviewed files + round-2 edge cases

A third pass targeted the code that got the *lightest* scrutiny in rounds 1–2
(the site-balance / quasi-equilibrium assembly, the transient/PFR DAE solves,
the AD/sensitivity core behind DRC, the symbolic LHHW derivation, and
discovery/rendering) plus adversarial edge-cases of the round-2 diff. Three
reviewers, all verifying by execution — **no high- or medium-severity bug was
found anywhere**. The AD machinery was cross-checked against independent
finite-difference re-solves (DRC/TRC/apparent orders all match to ~8 digits),
and the symbolic rate law matched hand-known Langmuir–Hinshelwood forms.

Round 3 produced only low/doc-level items, all fixed:

- **examples.py** — the shipped `adiabatic_cstr` example failed `validate()`:
  its product `"B"` infers to element boron while reactant `"A"` infers to no
  element, a false imbalance. Gave `A`/`B`/`A*` an explicit shared
  `composition` (it is an A→B isomerization), so the example now validates.
- **numeric.py** — extended the warm-start physical-root check to the free-site
  coverage, so a spurious root with each θ ≤ 1 but Σθ > 1 (⇒ free < 0) is also
  rejected (the round-2 change only covered individual adsorbate coverages).
- **drc.py / mcp_server.py** — doc fixes: the thermodynamic-rate-control
  docstring's leading formula had a stray `1/(k_BT)` factor (the code and the
  second formula were correct), and the MCP `solve` tool still listed the
  removed `least_squares` method.

---

## Resolution status

All findings were addressed across four commits on `claude/module-review-ibadto`.
The suite now stands at **135 passed, 0 failed** (98 original tests, all green,
plus 37 new regression tests in `tests/test_review_fixes.py`, one or more per
finding). Every fix was verified by the full suite before the next phase.

| Finding | Status | Commit |
|---|---|---|
| H1 explicit-Keq electrochemical detailed balance | fixed | `8609857` |
| H2 log solve ignores reactor gas balance | fixed | `8609857` |
| H3 batch steady-state returns garbage | fixed (now raises) | `8609857` |
| H4 symbolic truncates fractional coefficients | fixed | `8609857` |
| H5 select flux screen drops zero-net-flux steps | fixed (one-way screen) | `8609857` |
| H6 equilibrated steps vanish from numeric paths | fixed (now raise) | `8609857` |
| H7 estimate assumes plain Arrhenius | fixed | `8609857` |
| M5 unverified "optimal" solves | fixed (residual check) | `8609857` |
| M6 log residual scaling forward-only | fixed | `8609857` |
| T1 WGS log-solve NLP failure | fixed (warm-start box) | `8609857` |
| M1 re-solve invalidates earlier solution | fixed (snapshots) | `498fcc2` |
| M2 parser splits ionic species (`H+`) | fixed | `498fcc2` |
| M3 energy-diagram TS ignores BV shift | fixed | `498fcc2` |
| M4 energy balance loses NASA7/Shomate Cp | fixed | `498fcc2` |
| M7 MILP capacity bound assumes activities ≤ 1 | fixed | `498fcc2` |
| M8 `_drc_table` silent failures / ignored warm start | fixed | `498fcc2` |
| M9 spec accepts typos / cross-field misuse | fixed (`extra="forbid"`) | `498fcc2` |
| M10 `to_spec` lossy (thermo, reactor type) | fixed | `498fcc2` |
| M11 `Site.density` documented but dead | fixed (doc-only) | `498fcc2` |
| M12 non-isothermal solve starts at ~10 K | fixed (rescaled T) | `498fcc2` |
| M13 `active_tol` default degrades linear DRC | doc clarification | `498fcc2` |
| C1 no agent/MCP electrochemistry surface | fixed (4 tools added) | `e853915` |
| C2 MCP tools drop args | fixed | `e853915` |
| C3 `analyze` swallows apparent-kinetics failures | fixed (note) | `e853915` |
| C4 `alpha` silently inert without interactions | fixed (warns) | `e853915` |
| C5 multidentate adsorbates unsupported | won't fix (documented limitation) | — |
| C6 dead fit warm-start knobs | fixed (theta0 wired; init doc'd; lb guard) | `e853915` |
| C7 limiting_potential assumes reduction | fixed (raises for oxidation) | `e853915` |
| C8 CV vs current-density sign conflict | fixed (doc-only) | `e853915` |
| C9 render hides electrochemistry | fixed | `e853915` |
| C10 route numbers int-rounded; routes lack caveat | fixed | `e853915` |
| C11 hard-coded gas `ub=1e6` | fixed (adaptive) | `e853915` |
| C12 numeric `baseH` mirror mismatch | fixed (raises) | `e853915` |
| P1 rate-expression graph duplication | fixed (shared rate_cache) | `1b875fb` |
| P2 uncached JAX compile in `_eval` | skipped (unsafe for DRC) | — |
| P3 `net_stoich` rebuilt every call | fixed (cached, ~17×) | `1b875fb` |
| P4 rate constants rebuilt per iteration | fixed (hoisted) | `1b875fb` |
| P5 select rebuilds models via spec round-trip | skipped (marginal/risky) | — |
| P6 pure-Python CV Thomas solve | fixed (banded, ~8×) | `1b875fb` |

**Deviations from the raw finding** (each an honest, least-invasive choice):
H3/H6/C7 turn silent-wrong-answers into clear errors rather than adding niche
support; M11/M13/C8 are doc-only where the code was already correct; C6 wires
`Observation.theta0` but documents (rather than force-wires) `FitParam.init`
because discopt has no per-variable start hook; M4 and the energy path raise
`NotImplementedError` for the niche energy-balance-plus-equilibrated combination;
P2 and P5 were skipped because they could not be shown behavior-preserving for
the sensitivity/selection paths. C5 (multidentate adsorbates) remains a
documented limitation — the site balance is one-site-per-adsorbate — but it
errors loudly rather than mis-solving.

---

## High-severity correctness

### H1. Explicit-`Keq` electrochemical steps ignore the potential — detailed balance broken *(verified)*
`kinetics.py:58` (`k_reverse`) and `kinetics.py:96` (`equilibrium_residual`)
use the raw user-supplied constant for explicit-rate steps:

```python
keq = rxn.Keq_param if rxn.explicit_rate else K_eq(rxn, T_expr, R, Tref, theta)
```

`Keq_param` is wired from the bare spec value (`model.py:231,235`) with no
`exp(-nFU/RT)` shift, so for a faradaic step given as `kf`/`Keq` the reverse
rate carries the *same* `exp(-β·nFU/RT)` Butler–Volmer factor as the forward
rate instead of the complementary `exp(+(1-β)·nFU/RT)`. This contradicts
`k_forward`'s own docstring ("``K_eq`` holds the full ``n F U`` shift"), the
documented CHE convention in `AGENTS.md`, and the NumPy mirror, which applies
the shift correctly (`numeric.py:69`):

```python
kr_j = kf_j / (rxn.Keq * np.exp(-ec / (R * T)))
```

Verified: for `A + * <=> A*` with `kf=1, Keq=1, n_electrons=1, β=0.5, U=0.5 V,
T=298`, the discopt path gives `kr = 5.9e-05` while the numeric path (and
physics) gives `kr = 1.7e+04` — a factor `exp(FU/RT) ≈ 2.9e8`. Every
polarization curve / Tafel analysis built from DFT-table (`kf`,`Keq`) steps is
grossly wrong, and the log-coordinate warm start (numeric) disagrees with the
solve it seeds. The same omission pins `equilibrated: true` faradaic steps
with explicit `Keq` at their U=0 equilibrium at every potential. No test
exercises `kf`/`Keq` + `n_electrons`.

**Fix**: multiply `Keq_param` by `dm.exp(-ec_shift(rxn) / (R * T_expr))` in
both places (or wire the shift into the parameter expression once).

### H2. `coordinates="log"` never imposes the reactor gas balance *(verified)*
`_solve_log` (`steady_state.py:299`) calls `reactor.create_gas(m, mkm)` but —
unlike the linear path (`steady_state.py:191`) — never adds
`reactor.gas_residuals(...)`. For CSTR / MassTransferReactor / RDE the gas
concentrations are free `continuous` variables constrained by nothing.
Verified: CO oxidation in a CSTR with `coordinates="log"` returns
`status: "optimal"` with gas `{CO: -8.7e-24, O2: -4.3e-24, CO2: 5e5}` (bound
midpoints) vs. the correct linear answer `{CO: 0.883, O2: 0.441, CO2: 0.117}`.
Silent — no error, no warning.

Compounding it, the auto-warm-start path (`agent.py:35`, also
`steady_state.py:269`) reads `getattr(reactor, "pressures", {})`; a CSTR has
`inlet`, not `pressures`, so the warm start is computed at all-zero gas
(verified: it returns a *negative* coverage `-1.1e-274`), and the ±`log_box`
search box is then centered near `log(1e-300) ≈ -690`, excluding every
physical coverage. `energy=` is also silently dropped in the log branch
(`steady_state.py:168-171`).

**Fix**: add the gas residuals (scaled) in `_solve_log`, or raise for
`dynamic_gas` reactors; make the warm start reactor-aware.

### H3. `solve_steady_state` on a `Batch` reactor returns "optimal" garbage *(verified)*
`Batch.create_gas` (`reactors.py:256-258`) creates free `C_*` variables "for
API symmetry" but inherits `gas_residuals -> []`, so the gas is unconstrained.
The spec schema accepts `type: "batch"` (`spec.py:101`), so
`agent.solve(spec)` / MCP `solve` on a batch spec reaches this path and
returns confidently wrong JSON (verified: gas `{CO: 593, O2: 5e5, CO2: 5e5}`,
`status: "optimal"`). Batch is documented "transient only" — the steady-state
path should raise.

### H4. `symbolic.py` truncates fractional stoichiometric coefficients *(verified)*
`symbolic.py:50,77` cast coefficients with `int(c)` / `int(nu)`. The spec
parser accepts fractional coefficients (`0.5 O2 + * <=> O*`), and `int(0.5)=0`
silently deletes that species from the derived closed-form rate law.
Verified: the LHHW expression for a mechanism with `0.5 O2` contains no
`P_O2` term at all. Use `sympy.Rational`/float exponents instead.

### H5. `select.py` flux screen drops zero-net-flux equilibrium steps
`reduce_by_drc` (`select.py:194-211`) screens on |net rate of progress|. A
quasi-equilibrated spectator adsorption step (`X + * <=> X*`, X* inert) has
net flux ≈ 0 at steady state yet can pin coverage near 1 and control every
rate. It is dropped at any `flux_tol > 0`; the verification then fails with
advice ("lower flux_tol") that cannot help since the flux is identically ~0.
The docstring's claim that removing negligible-net-flux steps "provably cannot
change the steady state" is false for equilibrium steps — screen on
`max(|forward|, |reverse|)` (one-way fluxes) instead.

### H6. `equilibrated: true` steps silently vanish from every numeric path
`numeric.rate_constants` sets `kf = kr = 0` for equilibrated steps
(`numeric.py:51-53`) and imposes no equilibrium relation. Consequently
`steady_state_numeric`, `integrate_coverages`, `turnover_frequency`, and *all*
of `select.py` (`_numeric_state`, `_flux_and_tof`, `_score_subset`,
`_select_milp` — where the flux cap becomes 0) compute the steady state of a
*different mechanism* with those steps deleted, with no warning. Since
`AGENTS.md` actively recommends `equilibrated: true`, mechanism reduction /
selection on such specs is silently wrong. Guard these entry points the way
`_solve_log` does (`steady_state.py:258-263`).

### H7. `estimate.py` assumes every step is plain Arrhenius
`_MKMExperiment.create_model` (`estimate.py:183-189`) wires only
`A_param`/`Ea_param` (and `beta` for faradaic steps). For explicit-rate steps
it creates `m.parameter("A_j", None)` from `rxn.A is None` while `k_forward`
actually reads the never-wired `rxn.kf_param`; `Keq_param`, `alpha_param`, and
`sp._interaction_params` are likewise not reset (compare `model.py:198-241`),
so fitting a previously-solved model with interactions mixes parameter handles
from two different discopt models. `net_rate` is also called without
`extents` (`estimate.py:212`), giving equilibrated steps a bare rate law they
must not have. `fit_kinetics` on any spec using `kf`/`Keq` or
`equilibrated: true` crashes or silently misbehaves.

---

## Medium-severity correctness

### M1. Re-solving invalidates earlier solutions (shared-state mutation) *(verified)*
`wire_parameters` (`model.py:196-242`) overwrites parameter handles stored on
the shared `Species`/`Reaction` objects on every solve, while
`SteadyStateSolution` lazily rebuilds rate expressions from those *current*
handles against its *old* model/result. Verified: solve, change `m.T`, solve
again → `degree_of_rate_control(sol1, ...)` and `sol1.to_dict()` raise
`ValueError: Parameter 'A_2' not found in model`. This breaks the documented
sweep workflow ("set `model.U` … re-solve") whenever the earlier solution is
touched afterward. Snapshot the handle maps on the solution instead.

### M2. Reaction-string parser splits on bare `+` — ionic species mangled *(verified)*
`_parse_side` (`spec.py:42`) does `side.split("+")`, so
`"O2 + H+ + * -> OOH*"` parses `H+` as species `H` (verified). In an
`H*`/`H`-containing mechanism it silently binds the wrong species; otherwise
it errors misleadingly. The package's own ORR example dodges this only by
using the Unicode name `H⁺` (`electrochem/examples.py:35`). Split on
whitespace-delimited `+` tokens (e.g. `re.split(r"(?<=\s)\+(?=\s)", ...)`) or
require spaces around `+`.

### M3. Energy-diagram transition states ignore the electrochemical barrier shift
`viz.py:41-44` places the TS at `states[-1] + rxn.Ea` (bare chemical barrier)
while the state energies from `numeric.reaction_free_energy` include the full
`nFU` shift at the model's current `U`. Per `AGENTS.md` the forward barrier
shifts by `β·nFU`; for `orr_4e()` (built at `U=0.8`) transition states are
drawn *below* product states, and `report.py:46` embeds this wrong diagram in
every HTML report of an electrochemical mechanism.

### M4. Energy balance loses the flow term for polynomial-thermo species
`_cstr_energy_residual` (`steady_state.py:241-247`) and
`mixture_heat_capacity` (`energy.py:71`) fall back to the constant `g.Cp`
(0.0) when a species' Cp lives in a NASA7/Shomate model (`Cp_param is None`,
`model.py:203`), so `cp_in = 0` and the balance degenerates to `q + Q = 0` —
unphysical solved temperature, no warning. Related crash: `heat_release_rate`
(`energy.py:62`) calls `rate_of_progress` without `extents`, so energy balance
+ equilibrated steps hits `kf_param is None` (`TypeError`).

### M5. `method="least_squares"` (and the `"auto"` fallback) can report a non-steady state as "optimal"
The least-squares problem (`steady_state.py:204-209`) is "optimal" at any
local minimum of the residual norm; there is no post-solve check that
`Σ r² ≈ 0`. Exactly in the hard cases where feasibility failed, a
non-steady-state point flows into TOF/DRC with `status == "optimal"`. A
one-line residual-norm check closes this. Similarly `steady_state_numeric`
(`numeric.py:147-150`) calls `fsolve` without `full_output`, suppresses all
warnings, and returns non-converged (even negative-coverage) iterates as the
answer — and `select.py`'s `_physical` only box-checks the values.

### M6. Log-mode residual scaling uses forward fluxes only
`fwd_mag` (`steady_state.py:273-280`) omits the reverse flux (the `kr`
computed on line 270 is unused). For reverse-dominated warm starts or
zero-`conc0` reactants the scale collapses to the `1e-30` floor and the
constraint becomes `~1e30`-overscaled/unsolvable. Include
`kr·Π(products)` in `mags`.

### M7. MILP capacity bound assumes activities ≤ 1
`_select_milp` (`select.py:484`) caps step flux at `max(kf, kr)`, justified by
"activities ≤ 1" — false for partial pressures > 1 (tests already use P=2).
At 10 bar the true flux `kf·P` exceeds the cap, making feasible mechanisms
infeasible or forcing extra steps in. `_build_select_model`
(`select.py:277-282`) already does this correctly with `pmax ** order`.

### M8. `_drc_table` contract failures
`select.py:124-127` passes `theta0` to the linear solve, but
`solve_steady_state` only consumes `theta0` in log coordinates
(`steady_state.py:127` — also a public API trap: linear warm starts are
silently ignored). And when both solve attempts fail at every condition,
`select.py:133-134` leaves DRC at the initialized `0.0` — indistinguishable
from "not rate-controlling" (the agent layer reports `drc: null` instead).

### M9. Spec validation silently accepts typos and cross-field misuse
- No `model_config = ConfigDict(extra="forbid")` on any spec model
  (`spec.py:54-124`): `ModelSpec(reactons=...)` builds with `reactions=[]`;
  `ReactionSpec(n_electron=1)` is silently non-electrochemical — and
  `agent.validate` reports `ok: true` *(verified)*.
- `{"type": "cstr", "pressures": {...}}` builds a CSTR with an empty inlet
  (all-zero feed) *(verified)*; differential `pressures` keyed by an adsorbate
  resolve and are silently ignored.

### M10. `to_spec` round-trip is silently lossy
`spec.py:212-254`: NASA7/Shomate `thermo` models are dropped (re-import gets
flat-zero thermodynamics, *verified*), only differential reactors export
(CSTR/batch → no `reactor` section), callable-`H` species export with no `H`.

### M11. `Site.density` is documented but dead
`species.py:89-90` claims density converts rates to coverage derivatives, and
`AGENTS.md`'s example sets it — but nothing in the rate/ODE/steady-state path
ever reads it *(verified by grep)*. `density: 2.5` gives bit-identical results
to `1.0`; only the reactor-level `cat_density` actually enters.

### M12. Non-isothermal steady solve starts temperature at ~10 K
`m.continuous("T_var", lb=1.0, ub=1e5)` (`steady_state.py:177`) goes through
discopt's `_safe_x0` clip to `[-10, 10]`, so the energy-balance solve starts
where Arrhenius factors underflow. `transient.py:64-80` and `pfr.py:164-177`
both special-case exactly this; the steady-state path doesn't.

### M13. Default `active_tol=1e-3` degrades linear-coordinate DRC
The docstring itself warns tiny coverages need `active_tol` below them
(`steady_state.py:161-166`), yet the default treats any coverage < 1e-3 as
bound-active, nulling/degrading DRC — the `drc: null` → "retry with log"
advice in `AGENTS.md` is partly a workaround for this default.

---

## Completeness gaps

- **C1. No agent/MCP surface for electrochemistry.** `AGENTS.md` tells agents
  to use `discopt.mkm.electrochem` for current/Tafel/CHE/volcano, but every
  function there takes live model/solution objects; `mcp_server.py` registers
  only the 7 core tools, and `agent.py` never reports a faradaic current. The
  package's primary (spec/MCP) interface cannot produce any electrochemical
  observable.
- **C2. MCP tools drop arguments** (`mcp_server.py:42-46,70-74`): `solve`
  lacks `method`, `report` lacks `coordinates`, despite "tools mirror the
  `agent` functions".
- **C3. `agent.analyze` swallows apparent-kinetics failures** with a bare
  `except Exception: pass` (`agent.py:158-162`); keys silently vanish, unlike
  `apparent_kinetics`'s `note`. Related inconsistency: `solve`/`drc` raise raw
  exceptions for a reactorless spec while `validate`/`analyze` degrade
  gracefully.
- **C4. `alpha` (BEP) is silently inert** unless lateral interactions are
  registered (`model.py:239-241` gates `alpha_param` on `has_interactions`;
  `k_forward` applies it only in the Arrhenius branch). `AGENTS.md` documents
  it unconditionally.
- **C5. Multidentate adsorbates unsupported**: site balance hard-codes one
  site per adsorbate (`assemble.py:28-37`); `CO3*` occupying 2 sites cannot be
  modeled (at least `check_site_balance` errors loudly).
- **C6. Dead fit knobs**: `FitParam.init` ("used to center the search box")
  is computed and discarded (`estimate.py:160`), and `Observation.theta0` is
  never read (`estimate.py:119`) — stiff fits get no warm start. Also
  `np.log(fp.lb)` with `lb <= 0` yields `-inf`/`nan` bounds undiagnosed
  (`estimate.py:162`).
- **C7. `limiting_potential`/volcano assume reduction steps**
  (`electrochem/thermo.py:53`, `electrochem/optimize.py:43`): dividing by
  `n_electrons` flips the exergonicity inequality for oxidation steps
  (`n < 0`, explicitly supported elsewhere) — OER-style mechanisms get a
  meaningless answer instead of an error.
- **C8. Sign-convention conflict inside `electrochem`**: `analysis.py` defines
  reduction current positive; `cv.py:93` returns cathodic-negative. Both are
  self-documented but nothing warns when comparing them.
- **C9. Rendering hides electrochemistry**: mechanism tables show no
  `n_electrons`/`beta` and titles omit `U` (`render.py:82-107`); the installed
  `skill/SKILL.md` omits electrochemical spec fields entirely.
- **C10. `agent.py:89` int-rounds route stoichiometric numbers**; genuine
  half-integer routes are reported wrong (e.g. `[1, 1.5]` → `[1, 2]`). And
  `reaction_routes` (`stoichiometry.py:89-116`) returns an arbitrary
  orthonormal null-space basis whose vectors can be mixtures of true
  Horiuti–Temkin routes, without the disclaimer `conservation_laws` carries.
- **C11. Hard-coded gas bound `ub=1e6`** (`reactors.py:108,182,258`) silently
  clips legitimate concentration scales (e.g. Pa-based models at ~10 bar).
- **C12. Mirror-contract mismatch**: `numeric.baseH` returns `0.0` for a
  callable `H` with `theta=None` where the symbolic path raises
  (`thermo.py:36`).

---

## Performance

- **P1. Expression-graph duplication.** Each `net_rate` call rebuilds the full
  `k_forward`/`k_reverse`/`K_eq`/`g_species` subtree per (species, reaction)
  pair (`kinetics.py:102-132`), so build size (and JAX compile time) scales
  O(n_species × n_reactions × thermo-subtree) instead of O(n_reactions);
  with NASA7/Shomate species the duplicated subtrees are large. Build each
  reaction's rate expression once and reuse it.
- **P2. Re-tracing per evaluation.** Every `SteadyStateSolution._eval`
  recompiles a fresh JAX function (`analysis/sensitivity.py` compile is
  uncached), and `to_dict()` rebuilds every rate expression from scratch;
  `report`/`analyze` repeat all of it.
- **P3. `net_stoich()` allocates a fresh dict on every call**
  (`reaction.py:74-82`) and is called inside every `fsolve` residual / ODE RHS
  evaluation and every expression build — millions of avoidable dict builds;
  the stoichiometry is immutable after `step()`, so cache it.
- **P4. `numeric.rate_constants` is rebuilt per RHS evaluation**
  (`numeric.py:136,224`) even when kinetics are coverage-independent (no
  interactions, no callable H) — hoistable constants dominating the stiff
  integration hot path.
- **P5. `select.py` rebuilds models via spec round-trips per candidate subset**
  (`select.py:98-110`; O(n²) greedy, O(2ⁿ) exhaustive/Pareto), each with
  multi-seed `fsolve` per condition; the spec dict and stoichiometry maps are
  invariant and could be computed once.
- **P6. `electrochem/cv.py` runs a pure-Python Thomas solve per time step**
  (~2M interpreter iterations per voltammogram); only row 0 of the matrix
  changes between steps — use `scipy.linalg.solve_banded` or precompute the
  constant forward sweep.

---

## Tests

- **T1. Deterministic failures at HEAD** (reproduced twice in a clean
  `uv sync` environment): the full-model WGS solve with `coordinates="log"`
  (the flagship log-coordinate use case in `AGENTS.md`) fails with POUNCE NLP
  status `error` (`steady_state.py:312` → `differentiable_solve_l3` raises).
  This fails `tests/test_quasi_equilibrium.py::test_qea_reproduces_full_ssa_rate`
  directly and errors all three `tests/test_wgs.py` tests through the
  module-scoped `wgs_solution` fixture (same solve at `test_wgs.py:21`).
  The log path has no fallback equivalent to `method="auto"`.
- **T2. Coverage gaps aligned with the bugs above**: no test combines
  `kf`/`Keq` with `n_electrons` (H1); none solves a CSTR in log coordinates
  (H2) or a batch spec at steady state (H3); no `select`/`estimate` test uses
  `equilibrated: true` or explicit-rate steps (H5–H7); fractional
  coefficients never hit `symbolic.py` (H4); no `+`-suffixed ionic species
  name appears in spec-parser tests (M2).

---

## What checked out clean

Verified correct while hunting: Campbell DRC implementation (perturbs the
pre-exponential so `kf` and the derived `kr` move together at fixed `Keq`;
`(A/r)·dr/dA = d ln r/d ln A`), equilibrated-step DRC ≡ 0; apparent
`Ea = RT²·d ln r/dT` sign/units; CHE shift sign chain for Arrhenius faradaic
steps (β forward / (1−β) derived reverse; detailed balance holds at every U);
Tafel-slope and `alpha_app` signs; Levich coefficient and the
Koutecky–Levich mixed-control residual; ORR example thermochemistry (ΔG sum
−4.92 eV); CV Robin boundary-condition algebra and scan-rate/`dt`
consistency; stoichiometric rank / conservation-law counting; CSTR/PFR/batch
balance conventions; matplotlib figures closed in `report.py`; JSON
serialization (`float()` casts) in `agent.py`; MCP tool→function bindings;
`che_free_energies`/`limiting_potential` restore `mkm.U` in `finally`.

## Suggested priorities

1. **H1/H2/H3** — silent wrong answers behind documented, spec-reachable
   options (`kf`+`Keq`+`n_electrons`; `coordinates="log"` with CSTR; batch
   steady state). Fix or make them raise.
2. **H5/H6/H7** — `select`/`estimate` on QEA/explicit-rate mechanisms; guard
   the numeric paths.
3. **M5 + T1** — add residual-norm verification after every "optimal" solve;
   give the log path a fallback. This also converts many silent failures
   above into loud ones.
4. **M9** — `extra="forbid"` on the pydantic specs is one line per model and
   fixes the biggest agent-facing validation gap.
5. The performance items (P1–P4) are structural but pay off across every
   solve; P3 is a one-line cache.
