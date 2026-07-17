"""Averaged null energy condition (ANEC) along axial null geodesics.

The pointwise energy conditions can be violated by quantum fields, but
the averaged null energy condition

    integral of T_ab k^a k^b dlambda  >=  0

along a complete null geodesic (k^a = dx^a/dlambda, lambda affine) is
conjectured to hold for physically admissible matter, so it is the
sharper test for warp-drive viability than any single-point check.

The evaluator rides on the 1-D axial-slice machinery: the null ray is
integrated in coordinate time with MetricLine.coordinate_acceleration
(exact for null geodesics), and the affine scale A = dt/dlambda is
recovered algebraically at every output point from the conserved
Killing energy E = -g_ta k^a of the stationary slice (the same
bookkeeping CMBBlueshiftHazard uses), so no momentum transport
equation is integrated:

    k^a = A (1, v),  A = -E / (g_ta w^a),  w = (1, v)
    integral T_ab k^a k^b dlambda = integral A T_ab w^a w^b dt.

The affine normalization is fixed by A = 1 at the launch point;
rescaling the affine parameter multiplies the integral by a positive
constant, so the sign of the result -- which is what the condition
tests -- is normalization-independent.
"""

from typing import Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp

from ..solver.tensor_utils import components_to_tensor
from ..spacetime.interpolation import MetricLine


class AveragedNullEnergy:
    """ANEC integral for null rays along the x axis of a 1-D slice.

    Rays launched along +/- x stay on the axis: the sampled metric
    depends only on x, so the y and z Christoffel components vanish
    for axial motion.

    Parameters
    ----------
    t_max : float
        Maximum coordinate time to follow the ray
    dt : float
        Output sampling step along the ray
    """

    def __init__(self, t_max: float = 40.0, dt: float = 0.02):
        self.t_max = t_max
        self.dt = dt

    def integrate(
        self,
        metric: Dict[str, np.ndarray],
        stress_energy: Dict[str, np.ndarray],
        x_start: float,
        direction: float = 1.0,
        x: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Trace one axial null geodesic and accumulate the ANEC integral.

        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Covariant metric components sampled on the x grid
        stress_energy : Dict[str, np.ndarray]
            Covariant stress-energy components ("T_tt", ...) on the
            same grid, e.g. from EnergyTensor.calculate_from_metric
        x_start : float
            Launch position on the axis, normally in the far field
        direction : float
            +1 to propagate toward +x, -1 toward -x
        x : np.ndarray, optional
            Grid the components were sampled on; defaults to the
            package standard uniform grid on [-5, 5]

        Returns
        -------
        Dict[str, np.ndarray]
            "anec": the integral (affine normalization A = 1 at
            launch; negative means ANEC violation), "times",
            "positions" (x along the ray), "integrand"
            (A T_ab w^a w^b, per unit coordinate time)
        """
        line = MetricLine(metric) if x is None else MetricLine(metric, x)
        T_cov = components_to_tensor(stress_energy, "T")

        position = np.array([x_start, 0.0, 0.0])
        v0 = line.null_velocity(position, np.array([direction, 0.0, 0.0]))
        g0 = line.tensor_at(x_start)
        w0 = np.concatenate([[1.0], v0])
        killing_energy = float(-(g0[0] @ w0))

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            pos, vel = y[:3], y[3:]
            return np.concatenate([vel, line.coordinate_acceleration(pos, vel)])

        x_edge = 0.98 * float(np.max(np.abs(line.x)))
        if abs(x_start) >= x_edge:
            raise ValueError(
                f"x_start={x_start} is outside the ray-termination "
                f"boundary |x| < {x_edge:.3f}; launch inside the grid"
            )

        def exit_grid(t: float, y: np.ndarray) -> float:
            return x_edge - abs(float(y[0]))

        exit_grid.terminal = True  # type: ignore[attr-defined]

        sol = solve_ivp(
            fun=rhs,
            t_span=(0.0, self.t_max),
            y0=np.concatenate([position, v0]),
            t_eval=np.arange(0.0, self.t_max, self.dt),
            events=exit_grid,
            method="RK45",
            rtol=1e-10,
            atol=1e-10,
        )
        if not sol.success:
            raise RuntimeError(f"Ray integration failed: {sol.message}")

        ray_x = sol.y[0]
        velocities = sol.y[3:]

        integrand = np.empty(len(sol.t))
        for i, (xi, vel) in enumerate(zip(ray_x, velocities.T)):
            g = line.tensor_at(xi)
            w = np.concatenate([[1.0], vel])
            affine_scale = -killing_energy / (g[0] @ w)
            T_here = np.array(
                [
                    [np.interp(xi, line.x, T_cov[mu, nu]) for nu in range(4)]
                    for mu in range(4)
                ]
            )
            integrand[i] = affine_scale * (w @ T_here @ w)

        anec = float(np.sum((integrand[1:] + integrand[:-1]) / 2 * np.diff(sol.t)))
        return {
            "anec": anec,
            "times": sol.t,
            "positions": ray_x,
            "integrand": integrand,
        }
