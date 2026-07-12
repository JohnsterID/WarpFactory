"""Pointwise energy-condition maps for stress-energy on 4-D grids.

Python port of the MATLAB getEnergyConditions.m pipeline
(doFrameTransfer.m, getEulerianTransformationMatrix.m,
generateUniformField.m, getEvenPointsOnSphere.m). The stress-energy
tensor is transformed into the local Eulerian (orthonormal) frame and
contracted against a sampled field of null or timelike observers; the
returned map holds the most-violating evaluation at every grid point
(negative values indicate violation).

Deliberate divergence from the MATLAB original: once components are in
the local orthonormal frame, indices are raised and lowered with the
Minkowski metric for ALL conditions. getEnergyConditions.m lowers with
the full coordinate metric for Null/Weak but with Minkowski for
Strong/Dominant; that inconsistency is resolved here in favor of the
local-frame (Minkowski) convention.
"""

import warnings
from typing import Optional, Tuple, Union

import numpy as np

from .tensor import SpacetimeTensor, change_tensor_index, verify_tensor

CONDITIONS = ("null", "weak", "dominant", "strong")

_ETA_DIAG = np.array([-1.0, 1.0, 1.0, 1.0])


def even_points_on_sphere(radius: float, count: int) -> np.ndarray:
    """Approximately even points on a sphere via a Fibonacci lattice.

    Port of getEvenPointsOnSphere.m. Returns an array of shape
    (3, count).
    """
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    indices = np.arange(count)
    theta = 2.0 * np.pi * indices / golden_ratio
    phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / count)
    return radius * np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    )


def generate_uniform_field(
    field_type: str, num_angular_vec: int = 100, num_time_vec: int = 10
) -> np.ndarray:
    """Uniformly sampled null or timelike observer 4-vectors.

    Port of generateUniformField.m. Vectors are normalized to unit
    Euclidean 4-norm, matching the MATLAB convention.

    Returns
    -------
    np.ndarray
        Shape (4, num_angular_vec) for "nulllike"; shape
        (4, num_angular_vec, num_time_vec) for "timelike", where each
        time shell jj holds spatial speed 1 - jj/(num_time_vec - 1).
    """
    field_type = field_type.lower()
    if field_type == "nulllike":
        field = np.ones((4, num_angular_vec))
        field[1:] = even_points_on_sphere(1.0, num_angular_vec)
    elif field_type == "timelike":
        shells = np.linspace(0.0, 1.0, num_time_vec)
        field = np.ones((4, num_angular_vec, num_time_vec))
        for jj, bb in enumerate(shells):
            field[1:, :, jj] = even_points_on_sphere(1.0 - bb, num_angular_vec)
    else:
        raise ValueError(
            'Vector field type not generated, use either: "nulllike", "timelike"'
        )
    return field / np.sqrt(np.sum(field**2, axis=0))


def eulerian_transformation_matrix(g: np.ndarray) -> np.ndarray:
    """Transformation M with M^T g M = eta at every grid point.

    Port of getEulerianTransformationMatrix.m: the explicit Cholesky
    decomposition of a symmetric metric with signature (-,+,+,+),
    yielding the unique lower-triangular M with positive diagonal.
    g has shape (4, 4) + grid_shape.
    """
    factor0 = g[3, 3]
    factor1 = -(g[2, 3] ** 2) + g[2, 2] * factor0
    factor2 = (
        2.0 * g[1, 2] * g[1, 3] * g[2, 3]
        - g[3, 3] * g[1, 2] ** 2
        - g[2, 2] * g[1, 3] ** 2
        + g[1, 1] * factor1
    )
    factor3 = (
        -2.0 * g[3, 3] * g[0, 1] * g[0, 2] * g[1, 2]
        + 2.0 * g[0, 2] * g[0, 3] * g[1, 2] * g[1, 3]
        + 2.0 * g[0, 1] * g[0, 2] * g[1, 3] * g[2, 3]
        + 2.0 * g[0, 1] * g[0, 3] * g[1, 2] * g[2, 3]
        - g[0, 1] ** 2 * g[2, 3] ** 2
        - g[0, 2] ** 2 * g[1, 3] ** 2
        - g[0, 3] ** 2 * g[1, 2] ** 2
        + g[2, 2] * (-2.0 * g[0, 1] * g[0, 3] * g[1, 3] + g[3, 3] * g[0, 1] ** 2)
        + g[1, 1]
        * (
            -2.0 * g[0, 2] * g[0, 3] * g[2, 3]
            + g[3, 3] * g[0, 2] ** 2
            + g[2, 2] * g[0, 3] ** 2
        )
        - g[0, 0] * factor2
    )

    M = np.zeros_like(g)
    with np.errstate(invalid="ignore", divide="ignore"):
        M[0, 0] = np.sqrt(factor2 / factor3)
        norm10 = np.sqrt(factor2 * factor3)
        M[1, 0] = (
            g[0, 1] * g[2, 3] ** 2
            + g[0, 2] * g[1, 2] * g[3, 3]
            - g[0, 2] * g[1, 3] * g[2, 3]
            - g[0, 3] * g[1, 2] * g[2, 3]
            + g[0, 3] * g[1, 3] * g[2, 2]
            - g[0, 1] * g[2, 2] * g[3, 3]
        ) / norm10
        M[2, 0] = (
            g[0, 2] * g[1, 3] ** 2
            - g[0, 3] * g[1, 2] * g[1, 3]
            + g[0, 1] * g[1, 2] * g[3, 3]
            - g[0, 1] * g[1, 3] * g[2, 3]
            - g[0, 2] * g[1, 1] * g[3, 3]
            + g[0, 3] * g[1, 1] * g[2, 3]
        ) / norm10
        M[3, 0] = (
            g[0, 3] * g[1, 2] ** 2
            - g[0, 2] * g[1, 2] * g[1, 3]
            - g[0, 1] * g[1, 2] * g[2, 3]
            + g[0, 1] * g[1, 3] * g[2, 2]
            + g[0, 2] * g[1, 1] * g[2, 3]
            - g[0, 3] * g[1, 1] * g[2, 2]
        ) / norm10

        M[1, 1] = np.sqrt(factor1 / factor2)
        norm21 = np.sqrt(factor1 * factor2)
        M[2, 1] = (g[1, 3] * g[2, 3] - g[1, 2] * g[3, 3]) / norm21
        M[3, 1] = (g[1, 2] * g[2, 3] - g[1, 3] * g[2, 2]) / norm21

        M[2, 2] = np.sqrt(factor0 / factor1)
        M[3, 2] = -g[2, 3] / np.sqrt(factor0 * factor1)

        M[3, 3] = np.sqrt(1.0 / factor0)

    if np.isinf(M).any() or np.isnan(M).any():
        warnings.warn(
            "Eulerian transformation has non-finite entries -- "
            "numerical precision insufficient"
        )
    return M


def do_frame_transfer(
    metric: SpacetimeTensor, energy_tensor: SpacetimeTensor, frame: str = "eulerian"
) -> SpacetimeTensor:
    """Transform the stress-energy tensor into the Eulerian frame.

    Port of doFrameTransfer.m: T_local = M^T T_cov M with M from the
    Cholesky decomposition of the metric, then relabeled contravariant
    in the local orthonormal frame (T^0i = -T_0i under eta).
    """
    if frame.lower() != "eulerian":
        raise ValueError(
            f"Unsupported frame '{frame}'; only 'eulerian' "
            "frame transfer is implemented"
        )
    if (energy_tensor.frame or "").lower() == "eulerian":
        return energy_tensor
    if not verify_tensor(metric):
        raise ValueError(
            "Metric failed verification; see verify_tensor(metric, quiet=False)"
        )
    if not verify_tensor(energy_tensor):
        raise ValueError(
            "Stress-energy failed verification; see "
            "verify_tensor(energy_tensor, quiet=False)"
        )

    T_cov = change_tensor_index(energy_tensor, "covariant", metric)
    g_cov = change_tensor_index(metric, "covariant")
    M = eulerian_transformation_matrix(np.asarray(g_cov.tensor, dtype=float))

    T_local = np.einsum("am...,ab...,bn...->mn...", M, T_cov.tensor, M)
    for i in range(1, 4):
        T_local[0, i] *= -1.0
        T_local[i, 0] *= -1.0

    return SpacetimeTensor(
        tensor=T_local,
        type=energy_tensor.type,
        index="contravariant",
        coords=energy_tensor.coords,
        scaling=energy_tensor.scaling,
        name=energy_tensor.name,
        params=dict(energy_tensor.params),
        frame="eulerian",
    )


def _lower_with_eta(T_contra: np.ndarray) -> np.ndarray:
    """T_mn = eta_ma eta_nb T^ab for local-frame components."""
    signs = np.einsum("m,n->mn", _ETA_DIAG, _ETA_DIAG)
    return T_contra * signs.reshape((4, 4) + (1,) * (T_contra.ndim - 2))


def get_energy_conditions(
    energy_tensor: SpacetimeTensor,
    metric: SpacetimeTensor,
    condition: str,
    num_angular_vec: int = 100,
    num_time_vec: int = 10,
    return_vec: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    """Energy-condition violation map (MATLAB getEnergyConditions).

    Parameters
    ----------
    energy_tensor : SpacetimeTensor
        Stress-energy tensor on the grid
    metric : SpacetimeTensor
        Metric the stress-energy was solved from
    condition : str
        "Null", "Weak", "Dominant", or "Strong"
    num_angular_vec : int
        Number of spatial directions sampled on the unit sphere
    num_time_vec : int
        Number of temporal shells (Weak/Strong only)
    return_vec : bool
        Also return the per-observer evaluations and the observer field

    Returns
    -------
    map : np.ndarray
        Most-violating evaluation at every grid point (negative values
        indicate violation); shape grid_shape
    vec, vec_field : np.ndarray, optional
        Only when return_vec is True: all evaluations, shape
        grid_shape + (num_angular_vec,) or + (num_angular_vec,
        num_time_vec), and the observer vectors used
    """
    condition = condition.lower()
    if condition not in CONDITIONS:
        raise ValueError(
            "Incorrect energy condition input, use either: "
            '"Null", "Weak", "Dominant", "Strong"'
        )
    if metric.coords.lower() != "cartesian":
        warnings.warn(
            "Evaluation not verified for coordinate systems other than Cartesian!"
        )
    if not verify_tensor(metric):
        raise ValueError(
            "Metric failed verification; see verify_tensor(metric, quiet=False)"
        )
    if not verify_tensor(energy_tensor):
        raise ValueError(
            "Stress-energy failed verification; see "
            "verify_tensor(energy_tensor, quiet=False)"
        )

    local = do_frame_transfer(metric, energy_tensor, "eulerian")
    T_cov = _lower_with_eta(np.asarray(local.tensor, dtype=float))
    grid_shape = metric.grid_shape

    field_type = "nulllike" if condition in ("null", "dominant") else "timelike"
    vec_field = generate_uniform_field(field_type, num_angular_vec, num_time_vec)

    violation_map = np.full(grid_shape, np.nan)
    vec = None

    if condition == "null":
        if return_vec:
            vec = np.zeros(grid_shape + (num_angular_vec,))
        for ii in range(num_angular_vec):
            k = vec_field[:, ii]
            contraction = np.einsum("mn...,m,n->...", T_cov, k, k)
            violation_map = np.fmin(violation_map, contraction)
            if return_vec:
                vec[..., ii] = contraction

    elif condition == "weak":
        if return_vec:
            vec = np.zeros(grid_shape + (num_angular_vec, num_time_vec))
        for jj in range(num_time_vec):
            for ii in range(num_angular_vec):
                t = vec_field[:, ii, jj]
                contraction = np.einsum("mn...,m,n->...", T_cov, t, t)
                violation_map = np.fmin(violation_map, contraction)
                if return_vec:
                    vec[..., ii, jj] = contraction

    elif condition == "strong":
        if return_vec:
            vec = np.zeros(grid_shape + (num_angular_vec, num_time_vec))
        trace = np.einsum("m,mm...->...", _ETA_DIAG, T_cov)
        eta = np.diag(_ETA_DIAG).reshape((4, 4) + (1,) * len(grid_shape))
        effective = T_cov - 0.5 * trace * eta
        for jj in range(num_time_vec):
            for ii in range(num_angular_vec):
                t = vec_field[:, ii, jj]
                contraction = np.einsum("mn...,m,n->...", effective, t, t)
                violation_map = np.fmin(violation_map, contraction)
                if return_vec:
                    vec[..., ii, jj] = contraction

    else:  # dominant
        if return_vec:
            vec = np.zeros(grid_shape + (num_angular_vec,))
        # T^m_n = eta^ma T_an: raising the first local-frame index
        # flips the sign of the time row only.
        T_mixed = T_cov.copy()
        T_mixed[0] *= -1.0
        for ii in range(num_angular_vec):
            k = vec_field[:, ii]
            flux = -np.einsum("mn...,n->m...", T_mixed, k)
            norm = np.einsum("m,m...->...", _ETA_DIAG, flux**2)
            # Signed magnitude: positive norm means a spacelike
            # (violating) flux; the final sign flip makes negative
            # values violating, consistent with the other conditions.
            signed = np.sign(norm) * np.sqrt(np.abs(norm))
            violation_map = np.fmax(violation_map, signed)
            if return_vec:
                vec[..., ii] = signed
        violation_map = -violation_map
        if return_vec:
            vec = -vec

    if return_vec:
        return violation_map, vec, vec_field
    return violation_map
