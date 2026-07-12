"""Israel thin-shell junction conditions for spherical hypersurfaces.

Matches two static spherically symmetric spacetimes across a timelike
hypersurface r = a (Israel 1966; Poisson, "A Relativist's Toolkit",
ch. 3). With metrics of the form

    ds^2 = -A(r) dt^2 + B(r) dr^2 + C(r) dOmega^2

the mixed extrinsic curvature components of the shell are

    K^tau_tau     = A' / (2 A sqrt(B))
    K^theta_theta = K^phi_phi = C' / (2 C sqrt(B))

evaluated on each side. Mixed components are invariant under a
rescaling of the coordinate time, so the interior and exterior t
coordinates need not be matched; the first junction condition reduces
to continuity of the areal radius, C_in(a) = C_out(a).

The Lanczos equation converts the extrinsic-curvature jump
[K^i_j] = K^i_j(out) - K^i_j(in) into the surface stress-energy

    S^i_j = -([K^i_j] - [K] delta^i_j) / (8 pi)

whose nonzero components for a spherical shell are the surface energy
density sigma = -S^tau_tau = -[K^theta_theta] / (4 pi) and the surface
pressure p = S^theta_theta = ([K^tau_tau] + [K^theta_theta]) / (8 pi).
Geometric units G = c = 1 throughout, matching the rest of the package.
"""

from typing import Dict

import numpy as np

from ..solver import FiniteDifference


class IsraelJunction:
    """Extrinsic curvature jumps and thin-shell surface stress-energy."""

    def __init__(self, order: int = 4):
        self.fd = FiniteDifference(order=order)

    def extrinsic_curvature(
        self,
        metric: Dict[str, np.ndarray],
        coords: Dict[str, np.ndarray],
        radius: float,
    ) -> Dict[str, float]:
        """Mixed extrinsic curvature K^i_j of the surface r = radius.

        Radial derivatives of the metric profiles are taken with finite
        differences on the sampled grid and interpolated at the shell.

        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Spherical metric components g_tt, g_rr, g_theta_theta
            sampled on coords["r"]
        coords : Dict[str, np.ndarray]
            Radial grid; the shell radius must lie inside it
        radius : float
            Areal position of the hypersurface

        Returns
        -------
        Dict[str, float]
            "K_tau_tau", "K_theta_theta" (= K^phi_phi), and "trace"
            [K] = K^tau_tau + 2 K^theta_theta
        """
        r = np.asarray(coords["r"], dtype=float)
        if not (r.min() <= radius <= r.max()):
            raise ValueError(
                f"shell radius {radius} outside the sampled grid [{r.min()}, {r.max()}]"
            )
        A = -np.asarray(metric["g_tt"], dtype=float)
        B = np.asarray(metric["g_rr"], dtype=float)
        C = np.asarray(metric["g_theta_theta"], dtype=float)
        dA = self.fd.derivative1(A, r)
        dC = self.fd.derivative1(C, r)

        def at(profile: np.ndarray) -> float:
            return float(np.interp(radius, r, profile))

        sqrt_B = np.sqrt(at(B))
        K_tau_tau = at(dA) / (2 * at(A) * sqrt_B)
        K_theta_theta = at(dC) / (2 * at(C) * sqrt_B)
        return {
            "K_tau_tau": K_tau_tau,
            "K_theta_theta": K_theta_theta,
            "trace": K_tau_tau + 2 * K_theta_theta,
        }

    def surface_stress_energy(
        self,
        inner_metric: Dict[str, np.ndarray],
        inner_coords: Dict[str, np.ndarray],
        outer_metric: Dict[str, np.ndarray],
        outer_coords: Dict[str, np.ndarray],
        radius: float,
        rtol: float = 1e-6,
    ) -> Dict[str, float]:
        """Surface stress-energy of the thin shell at r = radius.

        Parameters
        ----------
        inner_metric, inner_coords : Dict[str, np.ndarray]
            Spacetime on the r < radius side
        outer_metric, outer_coords : Dict[str, np.ndarray]
            Spacetime on the r > radius side
        radius : float
            Shell position, inside both sampled grids
        rtol : float
            Tolerance for the first junction condition (continuity of
            the areal radius across the shell)

        Returns
        -------
        Dict[str, float]
            "surface_density" : sigma, energy per unit proper area
            "surface_pressure" : p, isotropic tangential pressure
            "K_jump_tau_tau", "K_jump_theta_theta", "K_jump_trace" :
            extrinsic curvature discontinuities

        Raises
        ------
        ValueError
            If the induced angular metrics disagree at the shell (the
            first junction condition fails)
        """
        C_in = float(
            np.interp(
                radius,
                np.asarray(inner_coords["r"], dtype=float),
                np.asarray(inner_metric["g_theta_theta"], dtype=float),
            )
        )
        C_out = float(
            np.interp(
                radius,
                np.asarray(outer_coords["r"], dtype=float),
                np.asarray(outer_metric["g_theta_theta"], dtype=float),
            )
        )
        if not np.isclose(C_in, C_out, rtol=rtol):
            raise ValueError(
                "first junction condition violated: areal radius jumps "
                f"across the shell (g_theta_theta {C_in} inside vs "
                f"{C_out} outside at r = {radius})"
            )

        K_in = self.extrinsic_curvature(inner_metric, inner_coords, radius)
        K_out = self.extrinsic_curvature(outer_metric, outer_coords, radius)
        jump_tau = K_out["K_tau_tau"] - K_in["K_tau_tau"]
        jump_theta = K_out["K_theta_theta"] - K_in["K_theta_theta"]

        return {
            "surface_density": -jump_theta / (4 * np.pi),
            "surface_pressure": (jump_tau + jump_theta) / (8 * np.pi),
            "K_jump_tau_tau": jump_tau,
            "K_jump_theta_theta": jump_theta,
            "K_jump_trace": jump_tau + 2 * jump_theta,
        }
