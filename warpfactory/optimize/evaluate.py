"""Physical scoring of ansatz metrics via the grid pipeline."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..grid.energy_conditions import do_frame_transfer, get_energy_conditions
from ..grid.solver import GridSolver
from ..grid.tensor import SpacetimeTensor
from .ansatz import Ansatz


@dataclass
class EvaluationResult:
    """Energy-condition scoring of one metric configuration.

    All densities are geometric (1/m^2); multiply by
    warpfactory.grid.si_energy_factor() for J/m^3. Integrated
    quantities use the coordinate volume element dx dy dz per time
    slice, matching how downstream figures of merit compare
    configurations of the same grid.
    """

    params: Dict[str, float]
    null_map: np.ndarray
    weak_map: np.ndarray
    rho: np.ndarray
    null_ok: bool
    weak_ok: bool
    exotic_matter: float
    total_energy: float

    @property
    def valid(self) -> bool:
        return self.null_ok and self.weak_ok


def evaluate_ansatz(ansatz: Ansatz, params: Dict[str, float],
                    order: int = 4, num_angular_vec: int = 30,
                    num_time_vec: int = 5,
                    tolerance: float = 1e-10,
                    solver: Optional[GridSolver] = None
                    ) -> EvaluationResult:
    """Solve the EFE for an ansatz configuration and score it.

    Parameters
    ----------
    ansatz : Ansatz
        Metric family under study
    params : Dict[str, float]
        Parameter values passed to ansatz.build
    order : int
        Finite difference order for the solver (ignored when solver
        is given)
    num_angular_vec, num_time_vec : int
        Observer sampling used for the energy-condition maps
    tolerance : float
        Violations larger than -tolerance count as satisfied
    solver : GridSolver, optional
        Reuse an existing solver instance

    Returns
    -------
    EvaluationResult
    """
    metric = ansatz.build(params)
    if solver is None:
        solver = GridSolver(order=order)
    stress_energy = solver.solve(metric)

    null_map = get_energy_conditions(stress_energy, metric, "Null",
                                     num_angular_vec=num_angular_vec)
    weak_map = get_energy_conditions(stress_energy, metric, "Weak",
                                     num_angular_vec=num_angular_vec,
                                     num_time_vec=num_time_vec)
    rho = do_frame_transfer(metric, stress_energy).tensor[0, 0]

    dv = float(np.prod(metric.scaling[1:]))
    exotic = float(np.sum(np.clip(-weak_map, 0.0, None))*dv)
    total = float(np.sum(rho)*dv)

    return EvaluationResult(
        params=dict(params),
        null_map=null_map, weak_map=weak_map, rho=rho,
        null_ok=bool(null_map.min() >= -tolerance),
        weak_ok=bool(weak_map.min() >= -tolerance),
        exotic_matter=exotic, total_energy=total)
