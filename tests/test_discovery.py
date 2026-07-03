"""Mechanism discovery: POUNCE reaction-network of a PES -> mkm spec."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import discopt.mkm as mk
from discopt.mkm.analysis import stoichiometry as st

# Whole file: discovering a mechanism from a PES then building/solving it is very
# slow (a large POUNCE reaction-network solve). Deselected in CI via -m "not slow".
pytestmark = pytest.mark.slow


def _pes():
    # tilted symmetric quartic: minima near x = -2,-1,1,2 at distinct energies
    def V(p):
        x, y = p[0], p[1]
        return (x**2 - 1.0) ** 2 * (x**2 - 4.0) ** 2 / 8.0 + 0.5 * y**2 + 0.2 * x

    g = jax.grad(lambda p: V(jnp.asarray(p)))
    h = jax.hessian(lambda p: V(jnp.asarray(p)))
    return (lambda p: float(V(jnp.asarray(p))),
            lambda p: np.asarray(g(np.asarray(p, float))),
            lambda p: np.asarray(h(np.asarray(p, float))))


def test_discovered_spec_builds_a_consistent_model():
    fun, grad, hess = _pes()
    rn, spec = mk.discover_mechanism(fun, np.array([0.0, 0.0]), grad=grad, hess=hess,
                                     A=1e13, T=500.0, n_states=4, n_transition_states=4,
                                     patience=12, seed=0)
    assert len(spec["adsorbates"]) == len(rn.minima) >= 3
    assert len(spec["reactions"]) == len(rn.edges) >= 1

    # state energies match the minima; barriers positive
    energies = sorted(round(a["H"], 3) for a in spec["adsorbates"])
    assert energies == sorted(round(float(m.f), 3) for m in rn.minima)
    assert all(r["Ea"] > 0 for r in spec["reactions"])

    # the spec builds and is a structurally valid (site-conserving) model
    m, _ = mk.from_spec(spec)
    assert st.check_site_balance(m) == []
    assert st.n_independent_reactions(m) == len(m.reactions)
