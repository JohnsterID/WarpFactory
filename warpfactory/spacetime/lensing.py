"""Gravitational lensing calculations."""

from typing import Dict, List

import numpy as np
from scipy.integrate import solve_ivp

from .interpolation import MetricLine


class GravitationalLensing:
    """Trace null geodesics (light rays) through the sampled spacetime.

    Rays are integrated in coordinate-time parametrization using the
    exact reduced geodesic equation (see
    MetricLine.coordinate_acceleration), which holds for null rays
    because the affine factor cancels. Initial coordinate velocities are
    normalized to the local light cone.
    """

    def _null_velocity(
        self, line: MetricLine, position: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Scale a spatial direction onto the future light cone.

        Solves g_tt + 2 g_ti (s n^i) + g_ij (s n^i)(s n^j) = 0 for the
        larger root s > 0 (future-directed ray along +n).
        """
        g = line.tensor_at(position[0])
        n = direction / np.linalg.norm(direction)
        a = n @ g[1:, 1:] @ n
        b = 2.0 * (g[0, 1:] @ n)
        c = g[0, 0]
        disc = b * b - 4 * a * c
        if disc < 0:
            raise ValueError(
                "No null direction along the requested spatial "
                f"direction at x={position[0]:.3f}"
            )
        s = (-b + np.sqrt(disc)) / (2 * a)
        return s * n

    def _setup_ray_bundle(
        self,
        source_pos: np.ndarray,
        observer_pos: np.ndarray,
        bundle_radius: float,
        n_rays: int,
    ) -> List[Dict]:
        """Initial conditions for a circular bundle of rays."""
        direction = observer_pos - source_pos
        direction = direction / np.linalg.norm(direction)

        if abs(direction[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([0.0, 1.0, 0.0])
        right = np.cross(direction, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, direction)

        rays = []
        for i in range(n_rays):
            theta = 2 * np.pi * i / n_rays
            offset = bundle_radius * (right * np.cos(theta) + up * np.sin(theta))
            pos = source_pos + offset
            rays.append(
                {
                    "position": pos.copy(),
                    "direction": direction.copy(),
                    "path": [pos.copy()],
                    "time_delay": 0.0,
                }
            )
        return rays

    def _propagate_ray(self, line: MetricLine, ray: Dict, dt: float = 0.1) -> None:
        """Advance a ray by coordinate time dt along its null geodesic."""
        velocity = self._null_velocity(line, ray["position"], ray["direction"])

        def rhs(t, y):
            pos, vel = y[:3], y[3:]
            return np.concatenate([vel, line.coordinate_acceleration(pos, vel)])

        sol = solve_ivp(
            fun=rhs,
            t_span=(0.0, dt),
            y0=np.concatenate([ray["position"], velocity]),
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
        )
        ray["position"] = sol.y[:3, -1]
        new_dir = sol.y[3:, -1]
        ray["direction"] = new_dir / np.linalg.norm(new_dir)
        ray["path"].append(ray["position"].copy())
        ray["time_delay"] += dt

    def trace_light_rays(
        self,
        metric: Dict[str, np.ndarray],
        source_pos: np.ndarray,
        observer_pos: np.ndarray,
        bundle_radius: float,
        n_rays: int,
    ) -> List[Dict]:
        """Trace a bundle of light rays from source toward observer.

        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components sampled on the standard x grid
        source_pos, observer_pos : np.ndarray
            Endpoints (3-vectors)
        bundle_radius : float
            Radius of the circular ray bundle
        n_rays : int
            Number of rays in the bundle

        Returns
        -------
        List[Dict]
            Ray trajectories: position, direction, path, time_delay,
            plus magnification/shear/convergence once arrived
        """
        line = MetricLine(metric)
        rays = self._setup_ray_bundle(source_pos, observer_pos, bundle_radius, n_rays)
        initial_area = bundle_radius**2

        # A ray has arrived once it crosses the observer plane (the plane
        # through observer_pos normal to the source->observer direction)
        axis = observer_pos - source_pos
        axis = axis / np.linalg.norm(axis)

        max_steps = 200
        for _ in range(max_steps):
            for ray in rays:
                if ray.get("arrived"):
                    continue
                self._propagate_ray(line, ray)
                if (ray["position"] - observer_pos) @ axis >= 0:
                    ray["arrived"] = True
            if all(r.get("arrived") for r in rays):
                break

        # Bundle optics from the change of the bundle cross-section
        center = np.mean([r["position"] for r in rays], axis=0)
        offsets = np.array([r["position"] - center for r in rays])
        final_area = np.mean(np.sum(offsets**2, axis=1))
        magnification = initial_area / final_area if final_area > 0 else np.inf

        # Convergence/shear from the anisotropy of the final bundle shape
        spreads = np.sqrt(np.sum(offsets**2, axis=1))
        mean_spread = np.mean(spreads)
        anisotropy = (
            (np.max(spreads) - np.min(spreads)) / (2 * mean_spread)
            if mean_spread > 0
            else 0.0
        )
        convergence = 1.0 - mean_spread / bundle_radius

        for ray in rays:
            ray["magnification"] = magnification
            ray["shear"] = anisotropy
            ray["convergence"] = convergence
        return rays

    def analyze_bundle(self, rays: List[Dict]) -> Dict[str, float]:
        """Aggregate optical properties of a traced ray bundle."""
        return {
            "magnification": float(
                np.mean([r.get("magnification", 1.0) for r in rays])
            ),
            "shear": float(np.mean([r.get("shear", 0.0) for r in rays])),
            "convergence": float(np.mean([r.get("convergence", 0.0) for r in rays])),
        }
