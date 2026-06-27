"""discopt_mkm: microkinetic modeling on top of the discopt modeling language.

Declare heterogeneous-catalysis species and reactions, then solve the
automatically assembled mole balances at steady state or transiently, with
temperature entering kinetics (Arrhenius) and thermodynamics (K_eq) from one
consistent source, and Campbell degree of rate control computed by automatic
differentiation.

Example
-------
>>> import discopt_mkm as mk
>>> m   = mk.Model("co_ox", T=500)
>>> s   = m.site("Pt", density=1.0)
>>> CO  = m.gas("CO",  H=0.0,   S=0.0020)
>>> O2  = m.gas("O2",  H=0.0,   S=0.0021)
>>> CO2 = m.gas("CO2", H=-2.93, S=0.0023)
>>> COs = m.adsorbate("CO*", site=s, H=-1.5, S=0.0012)
>>> Os  = m.adsorbate("O*",  site=s, H=-1.2, S=0.0009)
>>> m.step(CO + s >> COs,             A=1e6, Ea=0.0)
>>> m.step(O2 + 2 * s >> 2 * Os,      A=1e6, Ea=0.0)
>>> m.step(COs + Os >> CO2 + 2 * s,   A=1e13, Ea=0.8)
"""

from discopt_mkm.analysis import (
    apparent_activation_energy,
    apparent_orders,
    degree_of_rate_control,
    thermo_rate_control,
)
from discopt_mkm import agent
from discopt_mkm import electrochem
from discopt_mkm.discovery import discover_mechanism, spec_from_reaction_network
from discopt_mkm.energy import EnergyBalance
from discopt_mkm.report import report_html, write_report
from discopt_mkm.select import (
    fit_rate_law,
    pareto_subgraph,
    reduce_by_drc,
    select_subgraph,
)
from discopt_mkm.spec import ModelSpec, from_spec, from_yaml, to_spec
from discopt_mkm.symbolic import lumped_rate_expression
from discopt_mkm.viz import energy_diagram, interactive_network, network_graph, to_dot
from discopt_mkm.estimate import FitParam, MKMEstimationResult, Observation, fit_kinetics
from discopt_mkm.model import MicrokineticModel
from discopt_mkm.pfr import PFRSolution, solve_pfr
from discopt_mkm.reaction import Reaction
from discopt_mkm.reactors import (
    CSTR,
    Batch,
    DifferentialReactor,
    MassTransferReactor,
    Reactor,
    RotatingDiskElectrode,
    levich_coefficient,
)
from discopt_mkm.species import Adsorbate, GasSpecies, Site, Species
from discopt_mkm.steady_state import SteadyStateSolution, solve_steady_state
from discopt_mkm.thermo_models import GeneralThermo, NASA7, Shomate, ThermoModel
from discopt_mkm.transient import TransientSolution, solve_transient

# user-facing alias matching the documented `mk.Model(...)` syntax
Model = MicrokineticModel

__all__ = [
    "Model",
    "MicrokineticModel",
    "Species",
    "GasSpecies",
    "Site",
    "Adsorbate",
    "Reaction",
    "Reactor",
    "DifferentialReactor",
    "CSTR",
    "Batch",
    "MassTransferReactor",
    "RotatingDiskElectrode",
    "levich_coefficient",
    "solve_steady_state",
    "SteadyStateSolution",
    "solve_transient",
    "TransientSolution",
    "solve_pfr",
    "PFRSolution",
    "EnergyBalance",
    "degree_of_rate_control",
    "thermo_rate_control",
    "apparent_orders",
    "apparent_activation_energy",
    "lumped_rate_expression",
    "energy_diagram",
    "network_graph",
    "interactive_network",
    "to_dot",
    "report_html",
    "write_report",
    "from_spec",
    "from_yaml",
    "to_spec",
    "discover_mechanism",
    "spec_from_reaction_network",
    "reduce_by_drc",
    "select_subgraph",
    "pareto_subgraph",
    "fit_rate_law",
    "ModelSpec",
    "agent",
    "electrochem",
    "FitParam",
    "Observation",
    "fit_kinetics",
    "MKMEstimationResult",
    "ThermoModel",
    "NASA7",
    "Shomate",
    "GeneralThermo",
]
