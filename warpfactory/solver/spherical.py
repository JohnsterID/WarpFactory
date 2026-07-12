"""Curvature of static spherically symmetric metrics.

For a metric of the form
    ds^2 = -A(r) dt^2 + B(r) dr^2 + C(r) dOmega^2
all curvature quantities reduce to closed-form expressions in A, B, C and
their radial derivatives (verified symbolically with sympy). The radial
derivatives are taken numerically, so any spherically symmetric metric is
supported -- nothing here assumes Schwarzschild.
"""

from typing import Dict

import numpy as np

from .finite_difference import FiniteDifference


class SphericalCurvature:
    """Christoffel symbols and curvature scalars for spherical metrics."""

    def __init__(self, order: int = 4):
        self.fd = FiniteDifference(order=order)

    def _profiles(self, metric: Dict[str, np.ndarray], r: np.ndarray):
        A = -np.asarray(metric["g_tt"], dtype=float)
        B = np.asarray(metric["g_rr"], dtype=float)
        C = np.asarray(metric["g_theta_theta"], dtype=float)
        dA = self.fd.derivative1(A, r)
        dB = self.fd.derivative1(B, r)
        dC = self.fd.derivative1(C, r)
        d2A = self.fd.derivative2(A, r)
        d2C = self.fd.derivative2(C, r)
        return A, B, C, dA, dB, dC, d2A, d2C

    def christoffel(
        self, metric: Dict[str, np.ndarray], coords: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Nonzero Christoffel symbols, keyed as "upper_lowerlower"."""
        r = np.asarray(coords["r"], dtype=float)
        theta = np.asarray(coords.get("theta", np.full_like(r, np.pi / 2)))
        A, B, C, dA, dB, dC, _, _ = self._profiles(metric, r)

        sin_t, cos_t = np.sin(theta), np.cos(theta)
        return {
            "t_tt": np.zeros_like(r),
            "t_tr": dA / (2 * A),
            "r_tt": dA / (2 * B),
            "r_rr": dB / (2 * B),
            "r_thetatheta": -dC / (2 * B),
            "r_phiphi": -(sin_t**2) * dC / (2 * B),
            "theta_rtheta": dC / (2 * C),
            "theta_phiphi": -sin_t * cos_t,
            "phi_rphi": dC / (2 * C),
            "phi_thetaphi": cos_t / np.where(sin_t == 0, np.nan, sin_t),
        }

    def ricci_tensor(
        self, metric: Dict[str, np.ndarray], coords: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        r = np.asarray(coords["r"], dtype=float)
        theta = np.asarray(coords.get("theta", np.full_like(r, np.pi / 2)))
        A, B, C, dA, dB, dC, d2A, d2C = self._profiles(metric, r)

        R_tt = (
            d2A / (2 * B)
            + dA * dC / (2 * B * C)
            - dA * dB / (4 * B**2)
            - dA**2 / (4 * A * B)
        )
        R_rr = (
            -d2C / C
            + dC**2 / (2 * C**2)
            + dB * dC / (2 * B * C)
            - d2A / (2 * A)
            + dA * dB / (4 * A * B)
            + dA**2 / (4 * A**2)
        )
        R_thth = 1 - d2C / (2 * B) + dB * dC / (4 * B**2) - dA * dC / (4 * A * B)
        R_phph = R_thth * np.sin(theta) ** 2
        return {
            "R_tt": R_tt,
            "R_rr": R_rr,
            "R_theta_theta": R_thth,
            "R_phi_phi": R_phph,
        }

    def ricci_scalar(
        self, metric: Dict[str, np.ndarray], coords: Dict[str, np.ndarray]
    ) -> np.ndarray:
        r = np.asarray(coords["r"], dtype=float)
        A, B, C, dA, dB, dC, d2A, d2C = self._profiles(metric, r)
        return (
            2 / C
            - 2 * d2C / (B * C)
            + dC**2 / (2 * B * C**2)
            + dB * dC / (B**2 * C)
            - d2A / (A * B)
            - dA * dC / (A * B * C)
            + dA * dB / (2 * A * B**2)
            + dA**2 / (2 * A**2 * B)
        )

    def kretschmann(
        self, metric: Dict[str, np.ndarray], coords: Dict[str, np.ndarray]
    ) -> np.ndarray:
        r = np.asarray(coords["r"], dtype=float)
        A, B, C, dA, dB, dC, d2A, d2C = self._profiles(metric, r)
        return (
            4 / C**2
            - 2 * dC**2 / (B * C**3)
            + 2 * d2C**2 / (B**2 * C**2)
            - 2 * dC**2 * d2C / (B**2 * C**3)
            + 3 * dC**4 / (4 * B**2 * C**4)
            - 2 * dB * dC * d2C / (B**3 * C**2)
            + dB * dC**3 / (B**3 * C**3)
            + dB**2 * dC**2 / (2 * B**4 * C**2)
            + d2A**2 / (A**2 * B**2)
            + dA**2 * dC**2 / (2 * A**2 * B**2 * C**2)
            - dA * d2A * dB / (A**2 * B**3)
            + dA**2 * dB**2 / (4 * A**2 * B**4)
            - dA**2 * d2A / (A**3 * B**2)
            + dA**3 * dB / (2 * A**3 * B**3)
            + dA**4 / (4 * A**4 * B**2)
        )
