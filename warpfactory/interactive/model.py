"""UI-agnostic explorer model shared by the Qt and Jupyter front ends.

Owns the metric catalog and the evaluate pipeline (metric -> stress-
energy -> diagnostics) so that neither front end re-implements the
computation. The Qt explorer consumes the cheap metric/stress-energy
part; the Jupyter explorer additionally requests diagnostics (energy
conditions, horizons, causal structure, quantum inequality).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..analyzer import EnergyConditions
from ..metrics import (
    AlcubierreMetric,
    LentzMetric,
    MinkowskiMetric,
    VanDenBroeckMetric,
    WarpShellMetric,
)
from ..physics import CausalStructure, FordRomanInequality
from ..solver import EnergyTensor
from ..spacetime import HorizonFinder

METRIC_CATALOG = {
    "Alcubierre": (AlcubierreMetric, {"v_s": 2.0, "R": 1.0, "sigma": 0.5}),
    "Lentz": (LentzMetric, {"v_s": 2.0, "R": 1.0, "sigma": 0.5}),
    "Van Den Broeck": (
        VanDenBroeckMetric,
        {"v_s": 2.0, "R": 1.0, "B": 2.0, "sigma": 0.5},
    ),
    "Warp Shell": (
        WarpShellMetric,
        {"v_s": 2.0, "R": 1.0, "thickness": 0.2, "sigma": 0.5},
    ),
    "Minkowski": (MinkowskiMetric, {}),
}


@dataclass
class ExplorationResult:
    """Everything the front ends can display for one metric evaluation."""

    metric_name: str
    params: Dict[str, float]
    x: np.ndarray
    metric: Dict[str, np.ndarray]
    stress_energy: Dict[str, np.ndarray]
    conditions: Optional[Dict[str, bool]] = None
    horizons: Optional[Dict[str, np.ndarray]] = None
    light_cone_tilt: Optional[np.ndarray] = None
    causality_violations: Optional[np.ndarray] = None
    quantum_inequality: Optional[Dict[str, float]] = None


class ExplorerModel:
    """Evaluate catalog metrics on a 1-D x line with optional diagnostics."""

    def __init__(self, x: Optional[np.ndarray] = None):
        self.x = np.linspace(-8.0, 8.0, 200) if x is None else np.asarray(x, float)
        self.energy_solver = EnergyTensor()
        self.energy_conditions = EnergyConditions()
        self.horizon_finder = HorizonFinder()
        self.causal = CausalStructure()
        self.quantum_inequality = FordRomanInequality()

    def defaults(self, metric_name: str) -> Dict[str, float]:
        """Default parameters of a catalog metric."""
        return dict(METRIC_CATALOG[metric_name][1])

    def evaluate(
        self,
        metric_name: str,
        params: Optional[Dict[str, float]] = None,
        diagnostics: bool = False,
    ) -> ExplorationResult:
        """Run the metric -> stress-energy pipeline, optionally with diagnostics.

        Parameters
        ----------
        metric_name : str
            Key into METRIC_CATALOG
        params : Dict[str, float], optional
            Metric parameters; defaults from the catalog when omitted
        diagnostics : bool
            Also evaluate energy conditions, horizon/ergosphere
            surfaces, light-cone structure, and (for bubble metrics
            with v_s, R, sigma) the Ford-Roman quantum inequality
        """
        metric_cls, catalog_defaults = METRIC_CATALOG[metric_name]
        if params is None:
            params = dict(catalog_defaults)

        x = self.x
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        metric = metric_cls().calculate(x, y, z, 0.0, **params)
        stress_energy = self.energy_solver.calculate_from_metric(metric, x)

        result = ExplorationResult(
            metric_name=metric_name,
            params=dict(params),
            x=x,
            metric=metric,
            stress_energy=stress_energy,
        )
        if diagnostics:
            self._add_diagnostics(result)
        return result

    def _add_diagnostics(self, result: ExplorationResult) -> None:
        x = result.x
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        result.conditions = {
            "weak": self.energy_conditions.check_weak(result.stress_energy),
            "null": self.energy_conditions.check_null(result.stress_energy),
            "strong": self.energy_conditions.check_strong(result.stress_energy),
            "dominant": self.energy_conditions.check_dominant(result.stress_energy),
        }
        result.horizons = self.horizon_finder.find_horizons(result.metric, x, y, z)
        if "g_tx" in result.metric and "g_xx" in result.metric:
            result.light_cone_tilt = self.causal.light_cone_tilt(result.metric, x, y, z)
            result.causality_violations = self.causal.find_causality_violations(
                result.metric, x, y, z
            )
        params = result.params
        if {"v_s", "R", "sigma"} <= params.keys():
            result.quantum_inequality = self.quantum_inequality.check_warp_bubble(
                v_b=params["v_s"], R=params["R"], sigma=params["sigma"]
            )
