"""Warp-metric parameter search and exotic-matter minimization.

Built on the warpfactory.grid pipeline: an Ansatz maps a parameter dict
to a metric, evaluate_ansatz solves the Einstein field equations and
scores the energy-condition violations, and scan_parameters /
minimize_exotic_matter drive grid searches and derivative-free scipy
minimizers over that scoring.
"""

from .ansatz import Ansatz, CallableAnsatz
from .evaluate import EvaluationResult, evaluate_ansatz
from .search import minimize_exotic_matter, scan_parameters

__all__ = [
    "Ansatz",
    "CallableAnsatz",
    "EvaluationResult",
    "evaluate_ansatz",
    "scan_parameters",
    "minimize_exotic_matter",
]
