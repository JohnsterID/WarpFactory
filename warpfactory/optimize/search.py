"""Parameter-space search drivers over evaluate_ansatz."""

import itertools
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from .ansatz import Ansatz
from .evaluate import EvaluationResult, evaluate_ansatz

# Objective value assigned when the solver produces non-finite output
# (e.g. a horizon forms inside the grid); large enough to steer
# Nelder-Mead/Powell away without overflowing their arithmetic.
_PENALTY = 1e50


def scan_parameters(
    ansatz: Ansatz,
    param_grid: Dict[str, Sequence[float]],
    fixed_params: Optional[Dict[str, float]] = None,
    **evaluate_kwargs,
) -> List[EvaluationResult]:
    """Evaluate an ansatz over the Cartesian product of a value grid.

    Parameters
    ----------
    ansatz : Ansatz
        Metric family under study
    param_grid : Dict[str, Sequence[float]]
        Values to sweep per parameter name
    fixed_params : Dict[str, float], optional
        Parameters held constant across the sweep
    evaluate_kwargs
        Forwarded to evaluate_ansatz

    Returns
    -------
    list of EvaluationResult, in itertools.product order
    """
    fixed_params = fixed_params or {}
    names = list(param_grid)
    results = []
    for combo in itertools.product(*param_grid.values()):
        params = {**fixed_params, **dict(zip(names, combo))}
        results.append(evaluate_ansatz(ansatz, params, **evaluate_kwargs))
    return results


def minimize_exotic_matter(
    ansatz: Ansatz,
    initial_params: Dict[str, float],
    fixed_params: Optional[Dict[str, float]] = None,
    method: str = "Nelder-Mead",
    max_iterations: int = 50,
    objective: Optional[Callable[[EvaluationResult], float]] = None,
    callback: Optional[Callable[[Dict[str, float], float], None]] = None,
    **evaluate_kwargs,
) -> OptimizeResult:
    """Minimize integrated exotic matter over ansatz parameters.

    Uses derivative-free scipy minimizers since the objective is a
    finite-difference EFE solve. Non-finite evaluations are penalized
    rather than propagated so the simplex can recover from unphysical
    corners of parameter space.

    Parameters
    ----------
    ansatz : Ansatz
        Metric family under study
    initial_params : Dict[str, float]
        Starting point; its keys define the search dimensions
    fixed_params : Dict[str, float], optional
        Parameters excluded from the search
    method : str
        A derivative-free scipy.optimize.minimize method
        ("Nelder-Mead" or "Powell")
    max_iterations : int
        scipy maxiter option
    objective : Callable, optional
        Score to minimize given an EvaluationResult; defaults to
        EvaluationResult.exotic_matter
    callback : Callable, optional
        Invoked as callback(params, objective_value) after every
        evaluation
    evaluate_kwargs
        Forwarded to evaluate_ansatz

    Returns
    -------
    scipy.optimize.OptimizeResult
        With x mapped back to a params dict in the extra attribute
        best_params
    """
    if method not in ("Nelder-Mead", "Powell"):
        raise ValueError(
            "method must be a derivative-free minimizer: 'Nelder-Mead' or 'Powell'"
        )
    fixed_params = fixed_params or {}
    names = list(initial_params)
    x0 = np.array([initial_params[k] for k in names], dtype=float)
    score = objective or (lambda result: result.exotic_matter)

    def scipy_objective(x: np.ndarray) -> float:
        params = {**fixed_params, **dict(zip(names, x))}
        try:
            result = evaluate_ansatz(ansatz, params, **evaluate_kwargs)
            value = score(result)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            value = _PENALTY
        if not np.isfinite(value):
            value = _PENALTY
        if callback is not None:
            callback(params, value)
        return value

    opt = minimize(
        scipy_objective, x0, method=method, options={"maxiter": max_iterations}
    )
    opt.best_params = {**fixed_params, **dict(zip(names, opt.x))}
    return opt
