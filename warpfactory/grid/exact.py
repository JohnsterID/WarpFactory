"""Exact curvature from hyper-dual automatic differentiation.

Alternative front end to the finite-difference GridSolver for metrics
available as analytic functions of the coordinates: the metric
derivatives d_k g and d_k d_n g are propagated exactly (to machine
precision) through the metric function with hyper-dual numbers, and the
downstream curvature algebra is shared with the FD solver
(ricci_from_derivatives / stress_energy_from_derivatives), so the two
pipelines can only differ by the derivative source.

This removes the O(h^p) stencil truncation error entirely for analytic
metrics -- the same accuracy an autodiff backend (JAX) delivers --
without adding a dependency. It cannot replace the FD solver for
metrics that exist only as sampled data (warp shells built from TOV
integrations, piecewise definitions like the Lentz soliton), where
finite differences remain the honest derivative estimate.

A metric function takes the four coordinate arrays (t, x, y, z), which
may be HyperDual, and returns a 4 x 4 nested sequence of components
built with numpy-compatible arithmetic. Constant components may be
plain floats.
"""

from typing import Callable, Dict, Sequence, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp

from ..solver.tensor_utils import inverse_tensor
from .hyperdual import HyperDual
from .metrics import GridSize, Vector4, _base_metric, _world_coordinates
from .shape_functions import alcubierre_shape
from .solver import (
    christoffel_from_derivatives,
    stress_energy_from_derivatives,
)
from .tensor import SpacetimeTensor

Coordinate = Union[float, np.ndarray, HyperDual]
MetricFunction = Callable[..., Sequence[Sequence[object]]]


def alcubierre_metric_fn(v: float, R: float, sigma: float) -> MetricFunction:
    """Analytic Alcubierre metric as a coordinate function.

    Same physics as alcubierre_metric (bubble center at xs = v t,
    alpha = 1, flat spatial metric, beta^x = -v f(r)), expressed as a
    smooth function so hyper-dual differentiation applies. r = 0 (the
    exact bubble center) is a coordinate singularity of the radial
    profile; keep evaluation points off it, as symmetric even-sized
    grids naturally do.
    """

    def metric(
        t: Coordinate, x: Coordinate, y: Coordinate, z: Coordinate
    ) -> Sequence[Sequence[object]]:
        r = np.sqrt((x - v * t) ** 2 + y**2 + z**2)
        beta_x = -v * alcubierre_shape(r, R, sigma)
        g_tt = -1.0 + beta_x * beta_x
        return [
            [g_tt, beta_x, 0.0, 0.0],
            [beta_x, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

    return metric


def exact_metric_derivatives(
    metric_fn: MetricFunction,
    t: Coordinate,
    x: Coordinate,
    y: Coordinate,
    z: Coordinate,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact g, d_k g, d_k d_n g at the given coordinates.

    Seeds the hyper-dual eps1/eps2 parts on each of the ten coordinate
    pairs (k <= n) and evaluates the metric function once per pair;
    first derivatives ride on the dual parts, mixed second derivatives
    on the cross part, all exact to machine precision.

    Parameters
    ----------
    metric_fn : callable
        Metric component function of (t, x, y, z)
    t, x, y, z : array_like
        Coordinate arrays, mutually broadcastable

    Returns
    -------
    g : np.ndarray
        Metric, shape (4, 4) + broadcast shape
    dg : np.ndarray
        dg[k, mu, nu] = d_k g_munu, shape (4, 4, 4) + broadcast shape
    d2g : np.ndarray
        d2g[k, n, mu, nu] = d_k d_n g_munu, shape (4, 4, 4, 4) + it
    """
    coords = [np.asarray(c, dtype=float) for c in (t, x, y, z)]
    shape = np.broadcast(*coords).shape

    g = np.zeros((4, 4) + shape)
    dg = np.zeros((4, 4, 4) + shape)
    d2g = np.zeros((4, 4, 4, 4) + shape)

    for k in range(4):
        for n in range(k, 4):
            seeded = [
                HyperDual(c, 1.0 if i == k else 0.0, 1.0 if i == n else 0.0)
                for i, c in enumerate(coords)
            ]
            components = metric_fn(*seeded)
            for mu in range(4):
                for nu in range(4):
                    entry = components[mu][nu]
                    if isinstance(entry, HyperDual):
                        if k == 0 and n == 0:
                            g[mu, nu] = entry.f
                        dg[k, mu, nu] = entry.f1
                        dg[n, mu, nu] = entry.f2
                        d2g[k, n, mu, nu] = entry.f12
                        d2g[n, k, mu, nu] = entry.f12
                    elif k == 0 and n == 0:
                        g[mu, nu] = entry
    return g, dg, d2g


class ExactGridSolver:
    """EFE solver with exact (hyper-dual) metric derivatives.

    Drop-in counterpart of GridSolver for analytic metric functions:
    same output conventions (contravariant stress-energy, geometric
    units), but the curvature carries no finite-difference truncation
    error, so the result is exact to round-off at every point,
    including grid boundaries.

    Parameters
    ----------
    metric_fn : callable
        Metric component function of (t, x, y, z); see module docstring
    name : str
        Metric name recorded on returned tensors
    """

    def __init__(self, metric_fn: MetricFunction, name: str = "exact"):
        self.metric_fn = metric_fn
        self.name = name

    def metric_at(
        self, t: Coordinate, x: Coordinate, y: Coordinate, z: Coordinate
    ) -> np.ndarray:
        """Covariant metric at arbitrary points, shape (4, 4) + shape."""
        coords = [np.asarray(c, dtype=float) for c in (t, x, y, z)]
        shape = np.broadcast(*coords).shape
        g = np.zeros((4, 4) + shape)
        components = self.metric_fn(*coords)
        for mu in range(4):
            for nu in range(4):
                g[mu, nu] = components[mu][nu]
        return g

    def christoffel_at(
        self, t: Coordinate, x: Coordinate, y: Coordinate, z: Coordinate
    ) -> np.ndarray:
        """Exact Gamma^a_bc at arbitrary points, shape (4, 4, 4) + shape."""
        g, dg, _ = exact_metric_derivatives(self.metric_fn, t, x, y, z)
        return christoffel_from_derivatives(inverse_tensor(g), dg)

    def null_tangent(
        self,
        t: float,
        x: float,
        y: float,
        z: float,
        spatial_direction: Sequence[float],
    ) -> np.ndarray:
        """Future-directed null 4-vector along a spatial direction.

        Solves g_tt kt^2 + 2 g_ti kt ki + g_ij ki kj = 0 for the larger
        root kt at a single point, with k^i the given spatial direction.
        """
        g = self.metric_at(t, x, y, z)
        ki = np.asarray(spatial_direction, dtype=float)
        a = g[0, 0]
        b = 2.0 * (g[0, 1:] @ ki)
        c = ki @ g[1:, 1:] @ ki
        disc = np.sqrt(b * b - 4.0 * a * c)
        kt = max((-b + disc) / (2.0 * a), (-b - disc) / (2.0 * a))
        return np.concatenate([[kt], ki])

    def stress_energy_at(
        self,
        t: Coordinate,
        x: Coordinate,
        y: Coordinate,
        z: Coordinate,
        contravariant: bool = False,
    ) -> np.ndarray:
        """Exact stress-energy at arbitrary points, shape (4, 4) + shape.

        Covariant T_munu by default; contravariant=True raises both
        indices (matching GridSolver.solve output).
        """
        g, dg, d2g = exact_metric_derivatives(self.metric_fn, t, x, y, z)
        g_inv = inverse_tensor(g)
        T_cov = stress_energy_from_derivatives(g, g_inv, dg, d2g)
        if contravariant:
            return np.asarray(
                np.einsum("ab...,ai...,bj...->ij...", T_cov, g_inv, g_inv)
            )
        return T_cov

    def solve(
        self,
        grid_size: GridSize,
        world_center: Vector4,
        grid_scale: Vector4 = (1, 1, 1, 1),
        contravariant: bool = True,
    ) -> SpacetimeTensor:
        """Stress-energy on a uniform grid (GridSolver.solve counterpart).

        Grid conventions match the grid metric builders: point
        (it, ix, iy, iz) sits at physical coordinates
        i * grid_scale - world_center, zero-based.
        """
        t, x, y, z = _world_coordinates(grid_size, world_center, grid_scale)
        grid_shape = tuple(int(n) for n in grid_size)
        T = self.stress_energy_at(t, x, y, z, contravariant=contravariant)
        T = np.broadcast_to(T, (4, 4) + grid_shape).copy()
        return SpacetimeTensor(
            tensor=T,
            type="stress-energy",
            index="contravariant" if contravariant else "covariant",
            coords="cartesian",
            scaling=(
                float(grid_scale[0]),
                float(grid_scale[1]),
                float(grid_scale[2]),
                float(grid_scale[3]),
            ),
            name=self.name,
            params={"derivatives": "hyperdual-exact"},
        )

    def metric_on_grid(
        self,
        grid_size: GridSize,
        world_center: Vector4,
        grid_scale: Vector4 = (1, 1, 1, 1),
    ) -> SpacetimeTensor:
        """Covariant metric SpacetimeTensor on a uniform grid."""
        t, x, y, z = _world_coordinates(grid_size, world_center, grid_scale)
        grid_shape = tuple(int(n) for n in grid_size)
        g = np.broadcast_to(self.metric_at(t, x, y, z), (4, 4) + grid_shape).copy()
        return _base_metric(
            self.name, grid_scale, {"derivatives": "hyperdual-exact"}, g
        )


class ExactNullGeodesicANEC:
    """ANEC along arbitrary null geodesics of an exact metric function.

    Extends the axial evaluator in warpfactory.physics.anec to off-axis
    rays: the 1-D slice API can only launch rays down the x axis
    because it has no metric data off it, but an analytic metric
    function is defined everywhere, so the full affinely-parametrized
    geodesic equation

        dk^a/dlambda = -Gamma^a_bc k^b k^c

    is integrated directly with exact hyper-dual Christoffels (no grid
    interpolation) and the integrand T_ab k^a k^b is exact pointwise.
    Affine normalization follows the launch tangent; rescaling it
    multiplies the integral by a positive constant, so the sign of the
    result is normalization-independent.

    Parameters
    ----------
    solver : ExactGridSolver
        Exact solver wrapping the metric function
    lambda_max : float
        Maximum affine parameter to follow the ray
    num_samples : int
        Output samples along the ray for the integral
    """

    def __init__(
        self,
        solver: ExactGridSolver,
        lambda_max: float = 60.0,
        num_samples: int = 800,
    ):
        self.solver = solver
        self.lambda_max = lambda_max
        self.num_samples = num_samples

    def integrate(
        self,
        start: Sequence[float],
        spatial_direction: Sequence[float],
        exit_radius: float = 12.0,
        comoving_velocity: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Trace one null geodesic and accumulate the ANEC integral.

        Parameters
        ----------
        start : sequence of 4 floats
            Launch event (t, x, y, z); off-axis impact parameters are
            simply nonzero y or z here
        spatial_direction : sequence of 3 floats
            Initial spatial tangent; the null time component is solved
            from the metric at the launch event
        exit_radius : float
            Ray terminates when its distance from the (possibly
            moving) bubble center exceeds this radius
        comoving_velocity : float
            Bubble center speed along x used by the exit test (v for
            xs = v t metrics, 0 for static metrics)

        Returns
        -------
        Dict[str, np.ndarray]
            "anec": the integral (negative means ANEC violation),
            "lambdas", "positions" (4 x N events), "tangents"
            (4 x N null tangents), "integrand" (T_ab k^a k^b), and
            "null_residual" (max |g_ab k^a k^b| along the ray, a
            solution-quality check)
        """
        launch = np.asarray(start, dtype=float)
        k0 = self.solver.null_tangent(
            launch[0], launch[1], launch[2], launch[3], spatial_direction
        )

        def rhs(lam: float, y: np.ndarray) -> np.ndarray:
            event, k = y[:4], y[4:]
            Gamma = self.solver.christoffel_at(*event)
            return np.concatenate([k, -np.einsum("abc,b,c->a", Gamma, k, k)])

        def exit_bubble_frame(lam: float, y: np.ndarray) -> float:
            xs = comoving_velocity * y[0]
            return float(
                exit_radius - np.sqrt((y[1] - xs) ** 2 + y[2] ** 2 + y[3] ** 2)
            )

        exit_bubble_frame.terminal = True  # type: ignore[attr-defined]

        sol = solve_ivp(
            fun=rhs,
            t_span=(0.0, self.lambda_max),
            y0=np.concatenate([launch, k0]),
            dense_output=True,
            events=exit_bubble_frame,
            method="RK45",
            rtol=1e-10,
            atol=1e-10,
            max_step=0.1,
        )
        if not sol.success:
            raise RuntimeError(f"Null geodesic integration failed: {sol.message}")

        lambdas = np.linspace(0.0, sol.t[-1], self.num_samples)
        states = sol.sol(lambdas)
        positions, tangents = states[:4], states[4:]

        integrand = np.empty(self.num_samples)
        null_residual = 0.0
        for i in range(self.num_samples):
            event, k = positions[:, i], tangents[:, i]
            T_cov = self.solver.stress_energy_at(*event)
            integrand[i] = k @ T_cov @ k
            g = self.solver.metric_at(*event)
            null_residual = max(null_residual, abs(float(k @ g @ k)))

        anec = float(np.sum((integrand[1:] + integrand[:-1]) / 2.0 * np.diff(lambdas)))
        return {
            "anec": anec,
            "lambdas": lambdas,
            "positions": positions,
            "tangents": tangents,
            "integrand": integrand,
            "null_residual": null_residual,
        }
