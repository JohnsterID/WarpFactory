"""Tests for Hawking-Ellis classification (warpfactory.grid.hawking_ellis).

Ground truths are the canonical stress-energy forms of Hawking & Ellis
Sec 4.3 / Martin-Moruno & Visser (arXiv:1702.05915): perfect fluids are
Type I with closed-form energy-condition margins, null dust is Type II,
the nilpotent k-m form is Type III, and a pure energy flux exceeding
its density is Type IV (violating all conditions for all observers).
"""

import numpy as np
import pytest

from warpfactory.grid import (
    GridSolver,
    SpacetimeTensor,
    alcubierre_metric,
    change_tensor_index,
    get_energy_conditions,
    hawking_ellis_classify,
    invariant_energy_conditions,
    local_mixed_stress_energy,
    minkowski_metric,
    type_i_witnesses,
)

ETA_INV = np.diag([-1.0, 1.0, 1.0, 1.0])


def point_stress_energy(T4):
    """A single-point contravariant stress-energy SpacetimeTensor."""
    arr = np.zeros((4, 4, 1, 1, 1, 1))
    arr[:, :, 0, 0, 0, 0] = T4
    return SpacetimeTensor(tensor=arr, type="stress-energy", index="contravariant")


def perfect_fluid(rho, p, velocity=0.0):
    """T^ab = (rho + p) u^a u^b + p eta^ab, boosted along x."""
    gamma = 1.0 / np.sqrt(1.0 - velocity**2)
    u = np.array([gamma, gamma * velocity, 0.0, 0.0])
    return point_stress_energy((rho + p) * np.outer(u, u) + p * ETA_INV)


@pytest.fixture(scope="module")
def flat_point():
    return minkowski_metric((1, 1, 1, 1))


class TestCanonicalTypes:
    def test_perfect_fluid_is_type_i(self, flat_point):
        result = hawking_ellis_classify(perfect_fluid(1.0, 0.1), flat_point)
        assert result.type_map.ravel()[0] == 1
        np.testing.assert_allclose(result.rho.ravel(), 1.0, atol=1e-12)
        np.testing.assert_allclose(result.pressures.ravel(), 0.1, atol=1e-12)

    def test_vacuum_is_type_i_with_zero_margins(self, flat_point):
        vacuum = point_stress_energy(np.zeros((4, 4)))
        result = hawking_ellis_classify(vacuum, flat_point)
        assert result.type_map.ravel()[0] == 1
        for condition in ("Null", "Weak", "Dominant", "Strong"):
            margin = invariant_energy_conditions(
                vacuum, flat_point, condition, classification=result
            )
            np.testing.assert_allclose(margin, 0.0, atol=1e-12)

    def test_null_dust_is_type_ii(self, flat_point):
        k = np.array([1.0, 1.0, 0.0, 0.0])
        result = hawking_ellis_classify(point_stress_energy(np.outer(k, k)), flat_point)
        assert result.type_map.ravel()[0] == 2

    def test_nilpotent_km_form_is_type_iii(self, flat_point):
        k = np.array([1.0, 1.0, 0.0, 0.0])
        m = np.array([0.0, 0.0, 1.0, 0.0])
        T4 = np.outer(k, m) + np.outer(m, k)
        result = hawking_ellis_classify(point_stress_energy(T4), flat_point)
        assert result.type_map.ravel()[0] == 3

    def test_pure_flux_is_type_iv(self, flat_point):
        flux = np.zeros((4, 4))
        flux[0, 1] = flux[1, 0] = 0.5
        result = hawking_ellis_classify(point_stress_energy(flux), flat_point)
        assert result.type_map.ravel()[0] == 4
        np.testing.assert_allclose(result.complex_magnitude.ravel(), 0.5, atol=1e-12)


class TestInvariantMargins:
    def test_type_i_closed_forms(self, flat_point):
        rho, p = 1.0, 0.1
        fluid = perfect_fluid(rho, p)
        expected = {
            "Null": rho + p,
            "Weak": rho,
            "Strong": rho + p,
            "Dominant": rho - p,
        }
        for condition, value in expected.items():
            margin = invariant_energy_conditions(fluid, flat_point, condition)
            np.testing.assert_allclose(margin.ravel(), value, atol=1e-12)

    def test_type_iv_violates_every_condition(self, flat_point):
        flux = np.zeros((4, 4))
        flux[0, 1] = flux[1, 0] = 0.5
        tensor = point_stress_energy(flux)
        for condition in ("Null", "Weak", "Dominant", "Strong"):
            margin = invariant_energy_conditions(tensor, flat_point, condition)
            np.testing.assert_allclose(margin.ravel(), -0.5, atol=1e-12)

    def test_unknown_condition_raises(self, flat_point):
        with pytest.raises(ValueError):
            invariant_energy_conditions(perfect_fluid(1.0, 0.0), flat_point, "Averaged")


class TestBoostInvariance:
    def test_classification_is_frame_independent(self, flat_point):
        # The differentiator over Eulerian sampling: a boosted perfect
        # fluid must classify identically to the fluid at rest.
        rest = hawking_ellis_classify(perfect_fluid(1.0, 0.1), flat_point)
        boosted = hawking_ellis_classify(perfect_fluid(1.0, 0.1, 0.6), flat_point)
        assert boosted.type_map.ravel()[0] == rest.type_map.ravel()[0] == 1
        np.testing.assert_allclose(boosted.rho, rest.rho, atol=1e-12)
        np.testing.assert_allclose(
            np.sort(boosted.pressures, axis=0),
            np.sort(rest.pressures, axis=0),
            atol=1e-12,
        )

    def test_margins_are_frame_independent(self, flat_point):
        for condition in ("Null", "Weak", "Dominant", "Strong"):
            rest = invariant_energy_conditions(
                perfect_fluid(1.0, 0.1), flat_point, condition
            )
            boosted = invariant_energy_conditions(
                perfect_fluid(1.0, 0.1, 0.9), flat_point, condition
            )
            np.testing.assert_allclose(boosted, rest, atol=1e-10, err_msg=condition)


class TestMixedTensor:
    def test_fluid_mixed_form(self, flat_point):
        T_mixed = local_mixed_stress_energy(perfect_fluid(2.0, 0.5), flat_point)
        np.testing.assert_allclose(
            T_mixed[:, :, 0, 0, 0, 0], np.diag([-2.0, 0.5, 0.5, 0.5]), atol=1e-12
        )


@pytest.fixture(scope="module", params=[0.5, 2.0])
def alcubierre_case(request):
    n, h = 24, 0.5
    grid_size = (1, n, n, n)
    world_center = (0.0,) + tuple((m - 1) * h / 2 for m in grid_size[1:])
    metric = alcubierre_metric(
        grid_size,
        world_center,
        v=request.param,
        R=3.0,
        sigma=2.0,
        grid_scale=(1, h, h, h),
    )
    T = GridSolver(order=4).solve(metric)
    return metric, T


class TestAlcubierreGrid:
    def test_wall_is_type_iv_dominated(self, alcubierre_case):
        # Rotational-shift bubble walls have no rest frame; the far
        # field is vacuum Type I. Holds at sub- and superluminal v_s.
        metric, T = alcubierre_case
        result = hawking_ellis_classify(T, metric)
        assert set(np.unique(result.type_map)) <= {1, 2, 3, 4}
        assert np.count_nonzero(result.type_map == 4) > 1000
        assert result.type_map[0, 0, 0, 0] == 1

    def test_invariant_check_catches_all_eulerian_violations(self, alcubierre_case):
        # NEC violation is observer-independent as a verdict: any point
        # the Eulerian sampling flags must also fail the invariant test.
        metric, T = alcubierre_case
        eulerian = get_energy_conditions(T, metric, "Null", num_angular_vec=50)
        invariant = invariant_energy_conditions(T, metric, "Null")
        assert not np.any(np.isnan(invariant))
        flagged = eulerian < -1e-9
        assert flagged.any()
        assert np.all(invariant[flagged] < 1e-12)

    def test_precomputed_classification_matches(self, alcubierre_case):
        metric, T = alcubierre_case
        classification = hawking_ellis_classify(T, metric)
        direct = invariant_energy_conditions(T, metric, "Weak")
        reused = invariant_energy_conditions(
            T, metric, "Weak", classification=classification
        )
        np.testing.assert_allclose(reused, direct)


class TestTypeIWitnesses:
    def test_witness_attains_margin_flat(self, flat_point):
        # Anisotropic NEC-violating fluid: worst null direction is the
        # p = -2 axis, T(k, k) = rho + p_min = -1 exactly.
        T4 = np.diag([1.0, -2.0, 0.5, 0.5])
        tensor = point_stress_energy(T4)
        classification = hawking_ellis_classify(tensor, flat_point)
        observer, null_witness = type_i_witnesses(classification, flat_point)
        u = observer[:, 0, 0, 0, 0]
        k = null_witness[:, 0, 0, 0, 0]
        eta = np.diag([-1.0, 1.0, 1.0, 1.0])
        T_cov = eta @ T4 @ eta
        assert abs(k @ eta @ k) < 1e-12
        np.testing.assert_allclose(u @ eta @ u, -1.0, atol=1e-12)
        np.testing.assert_allclose(k @ T_cov @ k, -1.0, atol=1e-12)
        np.testing.assert_allclose(u @ T_cov @ u, 1.0, atol=1e-12)

    def test_witness_boost_invariant_value(self, flat_point):
        # The attained contraction is a scalar: identical for the
        # boosted fluid even though the witness components differ.
        eta = np.diag([-1.0, 1.0, 1.0, 1.0])
        T4 = np.diag([1.0, -2.0, 0.5, 0.5])
        v = 0.6
        gamma = 1.0 / np.sqrt(1.0 - v**2)
        boost = np.eye(4)
        boost[0, 0] = boost[2, 2] = gamma
        boost[0, 2] = boost[2, 0] = gamma * v
        T4b = boost @ T4 @ boost.T
        tensor = point_stress_energy(T4b)
        classification = hawking_ellis_classify(tensor, flat_point)
        _, null_witness = type_i_witnesses(classification, flat_point)
        k = null_witness[:, 0, 0, 0, 0]
        T_cov = eta @ T4b @ eta
        assert abs(k @ eta @ k) < 1e-12
        np.testing.assert_allclose(k @ T_cov @ k, -1.0, atol=1e-10)

    def test_alcubierre_witnesses_verify_on_grid(self, alcubierre_case):
        # On the curved metric the witness must be null/timelike w.r.t.
        # the FULL metric g and attain the invariant null margin at
        # every Type I point; non-Type-I points must be NaN.
        metric, T = alcubierre_case
        classification = hawking_ellis_classify(T, metric)
        observer, null_witness = type_i_witnesses(classification, metric)
        T_cov = change_tensor_index(T, "covariant", metric).tensor
        g = metric.tensor
        type_i = classification.type_map == 1
        assert type_i.any() and (~type_i).any()

        k_norm = np.einsum("m...,mn...,n...->...", null_witness, g, null_witness)
        u_norm = np.einsum("m...,mn...,n...->...", observer, g, observer)
        contraction = np.einsum(
            "m...,mn...,n...->...", null_witness, T_cov, null_witness
        )
        margin = invariant_energy_conditions(
            T, metric, "Null", classification=classification
        )
        assert np.abs(k_norm[type_i]).max() < 1e-12
        np.testing.assert_allclose(u_norm[type_i], -1.0, atol=1e-12)
        np.testing.assert_allclose(contraction[type_i], margin[type_i], atol=1e-8)
        assert (observer[0][type_i] > 0).all()
        assert np.isnan(null_witness[0][~type_i]).all()
        assert np.isnan(observer[0][~type_i]).all()
