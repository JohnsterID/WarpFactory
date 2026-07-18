"""Tests for the conformal trace anomaly evaluator."""

import numpy as np
import pytest

from warpfactory.physics import ConformalTraceAnomaly


def schwarzschild(M=1.0, r_min=3.0, r_max=10.0, n=141):
    r = np.linspace(r_min, r_max, n)
    theta = np.full_like(r, np.pi / 2)
    f = 1 - 2 * M / r
    metric = {
        "g_tt": -f,
        "g_rr": 1 / f,
        "g_theta_theta": r**2,
        "g_phi_phi": r**2 * np.sin(theta) ** 2,
    }
    return metric, {"r": r, "theta": theta}


def de_sitter(H=0.1, n=181):
    r = np.linspace(0.5, 5.0, n)
    theta = np.full_like(r, np.pi / 2)
    f = 1 - H**2 * r**2
    metric = {
        "g_tt": -f,
        "g_rr": 1 / f,
        "g_theta_theta": r**2,
        "g_phi_phi": r**2 * np.sin(theta) ** 2,
    }
    return metric, {"r": r, "theta": theta}


class TestConformalTraceAnomaly:
    def test_schwarzschild_scalar_matches_published_value(self):
        # Vacuum: C^2 = E = K = 48 M^2/r^6, and for one conformal
        # scalar c - a = 2/(5760 pi^2), so
        # <T> = 48 M^2 / (2880 pi^2 r^6) = M^2/(60 pi^2 r^6)
        # (Duff, hep-th/9308075; standard Page-approximation input).
        M = 1.0
        metric, coords = schwarzschild(M)
        trace = ConformalTraceAnomaly(n_scalar=1).trace(metric, coords)
        r = coords["r"]
        expected = M**2 / (60 * np.pi**2 * r**6)
        interior = slice(5, -5)
        err = np.abs(trace[interior] - expected[interior]).max()
        assert err / expected[interior].max() < 1e-4

    def test_de_sitter_scalar_is_constant_negative(self):
        # Conformally flat: C^2 = 0, E = 24 H^4, so
        # <T> = -24 a H^4 = -H^4/(240 pi^2) for one scalar.
        H = 0.1
        metric, coords = de_sitter(H)
        anomaly = ConformalTraceAnomaly(n_scalar=1)
        trace = anomaly.trace(metric, coords)
        expected = -(H**4) / (240 * np.pi**2)
        interior = slice(5, -5)
        np.testing.assert_allclose(trace[interior], expected, rtol=1e-5)
        assert np.abs(anomaly.weyl_squared(metric, coords)[interior]).max() < 1e-10

    def test_field_content_scales_central_charges(self):
        # Standard-model-like counts add linearly: on vacuum
        # (C^2 = E) the trace scales as (c - a) per species.
        metric, coords = schwarzschild()
        interior = slice(5, -5)
        t_scalar = ConformalTraceAnomaly(n_scalar=1).trace(metric, coords)
        t_fermion = ConformalTraceAnomaly(n_scalar=0, n_fermion=1).trace(metric, coords)
        t_vector = ConformalTraceAnomaly(n_scalar=0, n_vector=1).trace(metric, coords)
        # (c - a): scalar 2, fermion 7, vector -26 (in 1/5760pi^2 units)
        np.testing.assert_allclose(
            t_fermion[interior], 3.5 * t_scalar[interior], rtol=1e-10
        )
        np.testing.assert_allclose(
            t_vector[interior], -13.0 * t_scalar[interior], rtol=1e-10
        )

    def test_box_r_term_vanishes_for_constant_curvature(self):
        metric, coords = de_sitter()
        anomaly = ConformalTraceAnomaly(n_scalar=1)
        interior = slice(5, -5)
        with_xi = anomaly.trace(metric, coords, box_r_coefficient=1.0)
        without = anomaly.trace(metric, coords)
        assert np.abs(with_xi[interior] - without[interior]).max() < 1e-4

    def test_backreaction_ratio_flags_planck_scale_curvature(self):
        # For M = 1 in geometric units the anomaly ~ 1/r^6 rivals a
        # classical scale of the same order near the wall; against a
        # huge classical scale it is negligible.
        metric, coords = schwarzschild()
        anomaly = ConformalTraceAnomaly(n_scalar=1)
        r = coords["r"]
        interior = slice(5, -5)
        tiny = anomaly.backreaction_ratio(metric, coords, np.full_like(r, 1e6))
        comparable = anomaly.backreaction_ratio(
            metric, coords, 1.0 / (60 * np.pi**2 * r**6)
        )
        assert tiny[interior].max() < 1e-10
        np.testing.assert_allclose(comparable[interior], 1.0, rtol=1e-4)

    def test_rejects_empty_field_content(self):
        with pytest.raises(ValueError, match="at least one field"):
            ConformalTraceAnomaly(n_scalar=0)
        with pytest.raises(ValueError, match="non-negative"):
            ConformalTraceAnomaly(n_scalar=-1)
