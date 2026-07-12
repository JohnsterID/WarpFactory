"""CMB/interstellar photon blueshift hazard maps for warp bubbles.

Photons swept up by a moving warp bubble are blueshifted relative to
the crew: a head-on photon entering a region with shift velocity v_x
is measured by Eulerian observers with frequency ratio 1/(1 - v_x)
relative to the far field, which diverges as the local shift reaches
the speed of light. This module traces null geodesics through the
sampled metric, tracks the photon frequency measured by the local
Eulerian congruence, and sweeps incidence angles to map where on the
bubble wall incoming radiation is most dangerous.

Frequency bookkeeping uses two exact facts for the stationary sampled
slice: the photon energy E = -g_ta k^a is conserved along the ray
(Killing vector d/dt), and the Eulerian observer 4-velocity is
u^a = -alpha g^{a t} with lapse alpha = 1/sqrt(-g^{tt}). Writing the
null 4-momentum as k^a = A (1, v) with v the coordinate velocity, the
conserved E fixes the affine scale A at every output point, so no
separate momentum transport equation is integrated.
"""

from typing import Dict

import numpy as np
from scipy.integrate import solve_ivp

from .interpolation import MetricLine


class CMBBlueshiftHazard:
    """Blueshift of photons crossing a warp bubble, per incidence angle.

    Parameters
    ----------
    t_max : float
        Maximum coordinate time to follow each photon
    dt : float
        Output sampling step along the ray
    """

    def __init__(self, t_max: float = 20.0, dt: float = 0.05):
        self.t_max = t_max
        self.dt = dt

    @staticmethod
    def _eulerian_velocity(g: np.ndarray) -> np.ndarray:
        """Eulerian (normal) observer 4-velocity u^a = -alpha g^{a t}."""
        g_inv = np.linalg.inv(g)
        alpha = 1.0 / np.sqrt(-g_inv[0, 0])
        return -alpha * g_inv[:, 0]

    def _frequency(self, line: MetricLine, position, velocity, energy) -> float:
        """Photon frequency omega = -g_ab k^a u^b seen by the local
        Eulerian observer, with the affine scale fixed by the conserved
        energy: k^a = A (1, v), A = -E / (g_ta w^a)."""
        g = line.tensor_at(position[0])
        w = np.concatenate([[1.0], velocity])
        affine_scale = -energy / (g[0] @ w)
        u = self._eulerian_velocity(g)
        return float(-affine_scale * (w @ g @ u))

    def trace_frequency_shift(
        self,
        metric: Dict[str, np.ndarray],
        start_position: np.ndarray,
        direction: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Trace one photon and its Eulerian-measured frequency ratio.

        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components sampled on the standard x grid
        start_position : np.ndarray
            Photon launch point (3-vector), normally in the far field
        direction : np.ndarray
            Spatial propagation direction (3-vector, any norm)

        Returns
        -------
        Dict[str, np.ndarray]
            "times", "positions" (N, 3), "blueshift" (omega relative to
            the launch point), "max_blueshift", "max_position"
        """
        line = MetricLine(metric)
        start_position = np.asarray(start_position, dtype=float)
        v0 = line.null_velocity(start_position, np.asarray(direction, dtype=float))

        g0 = line.tensor_at(start_position[0])
        w0 = np.concatenate([[1.0], v0])
        energy = float(-(g0[0] @ w0))
        omega_emitted = self._frequency(line, start_position, v0, energy)

        def rhs(t, y):
            pos, vel = y[:3], y[3:]
            return np.concatenate([vel, line.coordinate_acceleration(pos, vel)])

        x_edge = 0.98 * float(np.max(np.abs(line.x)))

        def exit_grid(t, y):
            return x_edge - abs(y[0])

        exit_grid.terminal = True

        sol = solve_ivp(
            fun=rhs,
            t_span=(0.0, self.t_max),
            y0=np.concatenate([start_position, v0]),
            t_eval=np.arange(0.0, self.t_max, self.dt),
            events=exit_grid,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
        )
        if not sol.success:
            raise RuntimeError(f"Ray integration failed: {sol.message}")

        positions = sol.y[:3].T
        velocities = sol.y[3:].T
        blueshift = np.array(
            [
                self._frequency(line, pos, vel, energy) / omega_emitted
                for pos, vel in zip(positions, velocities)
            ]
        )
        peak = int(np.argmax(blueshift))
        return {
            "times": sol.t,
            "positions": positions,
            "blueshift": blueshift,
            "max_blueshift": float(blueshift[peak]),
            "max_position": positions[peak],
        }

    def hazard_map(
        self,
        metric: Dict[str, np.ndarray],
        n_angles: int = 9,
        max_angle: float = np.pi / 3,
        start_x: float = 4.5,
    ) -> Dict[str, np.ndarray]:
        """Peak blueshift versus photon incidence angle.

        Photons are launched from ahead of the bubble (+x far field)
        toward it; angle 0 is a head-on ray along -x, larger angles
        tilt the ray into the y direction.

        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components sampled on the standard x grid
        n_angles : int
            Number of incidence angles in [0, max_angle]
        max_angle : float
            Largest incidence angle in radians (must stay below pi/2
            so rays actually travel toward the bubble)
        start_x : float
            Launch x position, inside the grid but in the far field

        Returns
        -------
        Dict[str, np.ndarray]
            "angles", "max_blueshift" per angle, "max_positions"
            (n_angles, 3) where each peak occurs
        """
        if not 0 < max_angle < np.pi / 2:
            raise ValueError("max_angle must lie in (0, pi/2)")
        angles = np.linspace(0.0, max_angle, n_angles)
        peaks = np.empty(n_angles)
        peak_positions = np.empty((n_angles, 3))
        for i, angle in enumerate(angles):
            direction = np.array([-np.cos(angle), np.sin(angle), 0.0])
            result = self.trace_frequency_shift(
                metric, np.array([start_x, 0.0, 0.0]), direction
            )
            peaks[i] = result["max_blueshift"]
            peak_positions[i] = result["max_position"]
        return {
            "angles": angles,
            "max_blueshift": peaks,
            "max_positions": peak_positions,
        }
