"""Conformal (trace) anomaly of quantum fields in curved spacetime.

The one honest, state-independent piece of the renormalized quantum
stress-energy tensor <T_munu>_ren: for massless conformally coupled
fields the trace <T^mu_mu>_ren is a local, closed-form function of
curvature invariants (Duff, hep-th/9308075),

    <T^mu_mu>_ren = c C_abcd C^abcd - a E + xi Box R

where C^2 is the Weyl tensor squared, E is the Gauss-Bonnet invariant,
and the Box R coefficient is renormalization-scheme dependent (it can
be shifted arbitrarily by a local R^2 counterterm), so it defaults to
zero here and is exposed as a knob. The central charges per field
species, in units of 1/(5760 pi^2):

    real conformal scalar:   c = 3,   a = 1
    Dirac (4-component) fermion: c = 18,  a = 11
    gauge vector:            c = 36,  a = 62

The full tensor <T_munu>_ren additionally needs a choice of vacuum
state (Boulware/Unruh/Hartle-Hawking, or their warp-bubble analogues)
and a state-dependent traceless part -- that is the open research
program. The trace alone already answers a sharp physical question:
where the anomaly is comparable to the classical stress-energy scale,
semiclassical backreaction cannot be neglected and the classical warp
metric is not self-consistent.

Closed-form ground truth used by the tests: Schwarzschild (vacuum, so
C^2 = E = K = 48 M^2/r^6) gives <T> = (c - a) 48 M^2 / r^6, which for
a single conformal scalar is the published M^2/(60 pi^2 r^6); the de
Sitter static patch (conformally flat, C^2 = 0, E = 24 H^4, R const)
gives <T> = -24 a H^4.

Spherically symmetric metrics only, riding on the same closed-form
curvature machinery as the Gauss-Bonnet/EFT watchdog. Geometric units
(G = c = 1); the anomaly is returned in 1/length^4.
"""

from typing import Dict

import numpy as np

from ..analyzer.scalar_invariants import ScalarInvariants
from ..solver.finite_difference import FiniteDifference


class ConformalTraceAnomaly:
    """Trace anomaly evaluator for a chosen massless field content.

    Parameters
    ----------
    n_scalar : int
        Number of real conformally coupled scalar fields
    n_fermion : int
        Number of Dirac (4-component) fermion fields
    n_vector : int
        Number of gauge vector fields
    order : int
        Finite difference accuracy order for the curvature invariants
    """

    def __init__(
        self,
        n_scalar: int = 1,
        n_fermion: int = 0,
        n_vector: int = 0,
        order: int = 4,
    ):
        if n_scalar < 0 or n_fermion < 0 or n_vector < 0:
            raise ValueError("field counts must be non-negative")
        if n_scalar + n_fermion + n_vector == 0:
            raise ValueError("at least one field species is required")
        norm = 1.0 / (5760.0 * np.pi**2)
        self.c_weyl = (3 * n_scalar + 18 * n_fermion + 36 * n_vector) * norm
        self.a_euler = (n_scalar + 11 * n_fermion + 62 * n_vector) * norm
        self.invariants = ScalarInvariants(order=order)
        self.fd = FiniteDifference(order=order)

    def weyl_squared(
        self, metric: Dict[str, np.ndarray], coords: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Weyl invariant C^2 = K - 2 R_ab R^ab + R^2 / 3."""
        return (
            self.invariants.kretschmann(metric, coords)
            - 2.0 * self.invariants.ricci_squared(metric, coords)
            + self.invariants.ricci_scalar(metric, coords) ** 2 / 3.0
        )

    def box_ricci_scalar(
        self, metric: Dict[str, np.ndarray], coords: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Covariant Laplacian Box R of the Ricci scalar radial profile.

        For a static spherical metric depending on r only,
        Box R = (1/sqrt(-g)) d_r ( sqrt(-g) g^rr d_r R ).
        """
        r = np.asarray(coords["r"], dtype=float)
        R = self.invariants.ricci_scalar(metric, coords)
        g_tt = np.asarray(metric["g_tt"], dtype=float)
        g_rr = np.asarray(metric["g_rr"], dtype=float)
        g_thth = np.asarray(metric["g_theta_theta"], dtype=float)
        g_phph = np.asarray(metric["g_phi_phi"], dtype=float)
        sqrt_g = np.sqrt(np.abs(g_tt * g_rr * g_thth * g_phph))
        flux = sqrt_g / g_rr * self.fd.derivative1(R, r)
        return self.fd.derivative1(flux, r) / sqrt_g

    def trace(
        self,
        metric: Dict[str, np.ndarray],
        coords: Dict[str, np.ndarray],
        box_r_coefficient: float = 0.0,
    ) -> np.ndarray:
        """<T^mu_mu>_ren = c C^2 - a E + xi Box R on the radial grid.

        Parameters
        ----------
        metric, coords : Dict[str, np.ndarray]
            Spherical metric components and coordinates
        box_r_coefficient : float
            Scheme-dependent Box R coefficient xi; zero by default
        """
        anomaly = self.c_weyl * self.weyl_squared(
            metric, coords
        ) - self.a_euler * self.invariants.gauss_bonnet(metric, coords)
        if box_r_coefficient != 0.0:
            anomaly = anomaly + box_r_coefficient * self.box_ricci_scalar(
                metric, coords
            )
        return anomaly

    def backreaction_ratio(
        self,
        metric: Dict[str, np.ndarray],
        coords: Dict[str, np.ndarray],
        classical_trace_scale: np.ndarray,
        floor: float = 1e-30,
    ) -> np.ndarray:
        """|anomaly| / |classical stress-energy trace scale|.

        Where this ratio approaches unity the semiclassical quantum
        contribution rivals the classical source and the classical
        metric is not self-consistent; where it is tiny, ignoring
        backreaction is justified. The floor regularizes vacuum points
        where the classical trace vanishes.
        """
        classical = np.maximum(np.abs(np.asarray(classical_trace_scale)), floor)
        return np.abs(self.trace(metric, coords)) / classical
