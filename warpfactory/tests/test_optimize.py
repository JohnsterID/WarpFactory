"""Tests for the parameter search module (warpfactory.optimize)."""

from functools import partial

import numpy as np
import pytest

from warpfactory.grid import alcubierre_metric, minkowski_metric
from warpfactory.optimize import (
    Ansatz,
    CallableAnsatz,
    evaluate_ansatz,
    minimize_exotic_matter,
    scan_parameters,
)

GRID_SIZE = (1, 24, 24, 24)
SPACING = 0.5
WORLD_CENTER = (0.0,) + tuple((n - 1) * SPACING / 2 for n in GRID_SIZE[1:])
FAST_EVAL = {"num_angular_vec": 10, "num_time_vec": 3}


def alcubierre_ansatz():
    builder = partial(
        alcubierre_metric,
        GRID_SIZE,
        WORLD_CENTER,
        grid_scale=(1, SPACING, SPACING, SPACING),
    )
    return CallableAnsatz(builder, ["v", "R", "sigma"], name="alcubierre")


class MinkowskiAnsatz(Ansatz):
    name = "minkowski"
    param_names = ("unused",)

    def build(self, params):
        return minkowski_metric(GRID_SIZE)


class TestCallableAnsatz:
    def test_builds_metric(self):
        ansatz = alcubierre_ansatz()
        metric = ansatz.build({"v": 0.5, "R": 3.0, "sigma": 2.0})
        assert metric.grid_shape == GRID_SIZE
        assert metric.name == "Alcubierre"

    def test_rejects_unknown_parameter(self):
        ansatz = alcubierre_ansatz()
        with pytest.raises(ValueError, match="warp_factor"):
            ansatz.build({"v": 0.5, "warp_factor": 9.0})

    def test_default_name_from_builder(self):
        ansatz = CallableAnsatz(alcubierre_metric, ["v"])
        assert ansatz.name == "alcubierre_metric"


class TestEvaluateAnsatz:
    def test_minkowski_is_valid_with_zero_energy(self):
        result = evaluate_ansatz(MinkowskiAnsatz(), {}, order=2, **FAST_EVAL)
        assert result.valid
        assert result.exotic_matter == 0.0
        assert result.total_energy == 0.0

    def test_alcubierre_is_invalid_with_exotic_matter(self):
        result = evaluate_ansatz(
            alcubierre_ansatz(), {"v": 0.5, "R": 3.0, "sigma": 2.0}, **FAST_EVAL
        )
        assert not result.null_ok
        assert not result.weak_ok
        assert not result.valid
        assert result.exotic_matter > 0.0
        assert result.total_energy < 0.0

    def test_maps_have_grid_shape(self):
        result = evaluate_ansatz(
            alcubierre_ansatz(), {"v": 0.5, "R": 3.0, "sigma": 2.0}, **FAST_EVAL
        )
        assert result.null_map.shape == GRID_SIZE
        assert result.weak_map.shape == GRID_SIZE
        assert result.rho.shape == GRID_SIZE

    def test_params_recorded(self):
        params = {"v": 0.3, "R": 3.0, "sigma": 2.0}
        result = evaluate_ansatz(alcubierre_ansatz(), params, **FAST_EVAL)
        assert result.params == params


class TestScanParameters:
    def test_exotic_matter_grows_with_velocity(self):
        results = scan_parameters(
            alcubierre_ansatz(),
            {"v": [0.1, 0.3, 0.5]},
            fixed_params={"R": 3.0, "sigma": 2.0},
            **FAST_EVAL,
        )
        exotic = [r.exotic_matter for r in results]
        assert exotic == sorted(exotic)
        assert exotic[0] > 0.0

    def test_product_order_and_params(self):
        results = scan_parameters(
            alcubierre_ansatz(),
            {"v": [0.1, 0.2], "R": [2.0, 3.0]},
            fixed_params={"sigma": 2.0},
            **FAST_EVAL,
        )
        combos = [(r.params["v"], r.params["R"]) for r in results]
        assert combos == [(0.1, 2.0), (0.1, 3.0), (0.2, 2.0), (0.2, 3.0)]


class TestMinimizeExoticMatter:
    def test_drives_velocity_to_zero(self):
        # Alcubierre exotic matter scales as v^2, so the global
        # minimum over v is flat spacetime.
        opt = minimize_exotic_matter(
            alcubierre_ansatz(),
            {"v": 0.5},
            fixed_params={"R": 3.0, "sigma": 2.0},
            max_iterations=25,
            **FAST_EVAL,
        )
        assert abs(opt.best_params["v"]) < 0.05
        assert opt.fun < 0.01

    def test_callback_sees_every_evaluation(self):
        seen = []
        minimize_exotic_matter(
            alcubierre_ansatz(),
            {"v": 0.4},
            fixed_params={"R": 3.0, "sigma": 2.0},
            max_iterations=3,
            callback=lambda params, value: seen.append((params["v"], value)),
            **FAST_EVAL,
        )
        assert len(seen) >= 3
        assert all(np.isfinite(v) for _, v in seen)

    def test_custom_objective(self):
        # Minimizing total |energy| instead of exotic matter must also
        # drive toward flat spacetime.
        opt = minimize_exotic_matter(
            alcubierre_ansatz(),
            {"v": 0.3},
            fixed_params={"R": 3.0, "sigma": 2.0},
            max_iterations=20,
            objective=lambda result: abs(result.total_energy),
            **FAST_EVAL,
        )
        assert abs(opt.best_params["v"]) < 0.1

    def test_rejects_gradient_methods(self):
        with pytest.raises(ValueError, match="derivative-free"):
            minimize_exotic_matter(alcubierre_ansatz(), {"v": 0.5}, method="BFGS")
