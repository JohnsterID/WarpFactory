"""Hawking-Ellis classification and observer-independent energy conditions.

Extension beyond the MATLAB original. The Eulerian sampling in
get_energy_conditions evaluates the energy conditions for one family of
observers; a violation seen only by some boosted observer is invisible
to it. The Hawking-Ellis classification (Hawking & Ellis 1973, Sec 4.3;
Martin-Moruno & Visser, arXiv:1702.05915) instead works with the
eigenstructure of the mixed stress-energy tensor T^a_b in a local
orthonormal frame, which is frame-independent:

- Type I: T^a_b has a timelike eigenvector. In the eigenframe
  T = diag(rho, p1, p2, p3) and every energy condition reduces to a
  closed-form inequality in (rho, p_i) that is exact for ALL observers.
- Type II/III: only null eigenvectors (double/triple degenerate root).
  Measure-zero boundary cases for finite-difference data.
- Type IV: the spectrum contains a complex-conjugate pair. No rest
  frame exists and every pointwise energy condition is violated for
  every observer.

Because the classification never builds a preferred observer, it stays
well-defined at all warp speeds, including superluminal bubbles.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .energy_conditions import (
    CONDITIONS,
    do_frame_transfer,
    eulerian_transformation_matrix,
)
from .tensor import SpacetimeTensor, change_tensor_index

_ETA_DIAG = np.array([-1.0, 1.0, 1.0, 1.0])


@dataclass
class HawkingEllisResult:
    """Pointwise Hawking-Ellis classification of a stress-energy field.

    Attributes
    ----------
    type_map : np.ndarray
        Integer Hawking-Ellis type (1-4) at every grid point;
        shape grid_shape
    rho : np.ndarray
        Energy density in the eigenframe (minus the eigenvalue on the
        most timelike eigenvector); exact for Type I points
    pressures : np.ndarray
        The three principal pressures, shape (3,) + grid_shape; exact
        for Type I points
    eigenvalues : np.ndarray
        Raw complex spectrum of T^a_b, shape (4,) + grid_shape
    complex_magnitude : np.ndarray
        |Im| of the complex pair at Type IV points, 0 elsewhere; a
        severity scale for the unconditional Type IV violation
    eigenvectors : np.ndarray
        Eigenvector columns of T^a_b in the local orthonormal frame,
        shape (4, 4) + grid_shape with [:, k] the k-th eigenvector;
        consumed by type_i_witnesses
    rho_index : np.ndarray
        Column index of the most timelike eigenvector (the one whose
        eigenvalue is -rho); shape grid_shape
    jordan_parameter : np.ndarray
        Jordan-block parameter f of the canonical Type II form
        (Martin-Moruno & Visser eq 2.15) at Type II points, 0
        elsewhere. Extracted as T_ab l^a l^b / 4 with l the null
        vector complementary to the degenerate null eigenvector k
        (k^0 = l^0 = 1 normalization). Its sign is boost-invariant
        and enters every Type II energy condition: f < 0 violates
        them all even when the eigenvalue margins vanish
        (e.g. negative-energy null dust)
    nilpotent_magnitude : np.ndarray
        Frobenius norm of T^a_b - lambda I at Type III points, 0
        elsewhere; the severity scale of the unconditional Type III
        violation (the cubic nilpotent piece violates every pointwise
        energy condition)
    """

    type_map: np.ndarray
    rho: np.ndarray
    pressures: np.ndarray
    eigenvalues: np.ndarray
    complex_magnitude: np.ndarray
    eigenvectors: np.ndarray
    rho_index: np.ndarray
    jordan_parameter: np.ndarray
    nilpotent_magnitude: np.ndarray


def local_mixed_stress_energy(
    energy_tensor: SpacetimeTensor, metric: SpacetimeTensor
) -> np.ndarray:
    """Mixed T^a_b in the local orthonormal (Eulerian) frame.

    Reuses do_frame_transfer for the orthonormal-frame contravariant
    components, then lowers the second index with eta. Shape
    (4, 4) + grid_shape.
    """
    local = do_frame_transfer(metric, energy_tensor, "eulerian")
    T_mixed = np.asarray(local.tensor, dtype=float).copy()
    T_mixed[:, 0] *= -1.0
    return T_mixed


def _has_cubic_jordan_block(
    matrices: np.ndarray, degenerate_root: np.ndarray, tolerance: float
) -> np.ndarray:
    """Type III test on A = T - lambda I for the degenerate root: a
    Jordan block of size >= 3 exists iff rank(A^2) > rank(A^3).
    Eigenvalue multiplicity alone cannot separate Types II and III.
    """
    shifted = matrices - degenerate_root[:, np.newaxis, np.newaxis] * np.eye(4)
    scale = np.linalg.norm(shifted, axis=(1, 2))
    shifted = shifted / np.maximum(scale, np.finfo(float).tiny)[:, None, None]
    # The eigensolver locates a defective root only to O(eps^(1/k)), so
    # residuals up to that size in the shifted powers are noise.
    rank_tol = max(tolerance, np.finfo(float).eps ** (1.0 / 3.0))

    def rank(matrix: np.ndarray) -> np.ndarray:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        return np.asarray(np.count_nonzero(singular_values > rank_tol, axis=1))

    squared = shifted @ shifted
    return rank(squared) > rank(squared @ shifted)


def _jordan_block_parameter(matrices: np.ndarray, null_vecs: np.ndarray) -> np.ndarray:
    """Jordan-block parameter f of the canonical Type II form.

    In the canonical frame (Martin-Moruno & Visser eq 2.15) the Type II
    stress-energy is T^ab = f k^a k^b + Type-I diagonal with k = e_0 +
    e_1 the degenerate null eigenvector; contracting the covariant
    tensor with the complementary null vector l = e_0 - e_1 isolates f
    as T_ab l^a l^b / 4 because eta(k, l) = -2 kills every other term.

    Parameters
    ----------
    matrices : np.ndarray
        Mixed T^a_b in the local orthonormal frame, shape (n, 4, 4)
    null_vecs : np.ndarray
        Degenerate null eigenvectors, shape (n, 4), any normalization

    Returns
    -------
    np.ndarray
        f at every point, shape (n,)
    """
    k = null_vecs.copy()
    time_component = k[:, 0]
    safe = np.where(np.abs(time_component) < np.finfo(float).tiny, 1.0, time_component)
    k = k / safe[:, np.newaxis]
    l_vec = k.copy()
    l_vec[:, 1:] *= -1.0

    # T_ab = eta_ac T^c_b
    T_cov = _ETA_DIAG[np.newaxis, :, np.newaxis] * matrices
    return np.einsum("na,nab,nb->n", l_vec, T_cov, l_vec) / 4.0


def hawking_ellis_classify(
    energy_tensor: SpacetimeTensor,
    metric: SpacetimeTensor,
    tolerance: float = 1e-9,
) -> HawkingEllisResult:
    """Classify the stress-energy at every grid point into Types I-IV.

    Parameters
    ----------
    energy_tensor : SpacetimeTensor
        Stress-energy tensor on the grid
    metric : SpacetimeTensor
        Metric the stress-energy was solved from
    tolerance : float
        Relative spectral tolerance separating a genuine complex pair
        (Type IV) or a degenerate root from numerical round-off

    Notes
    -----
    rho and pressures are eigenframe quantities and exact only at
    Type I points. At Type II/III points they are the real (Segre)
    part of the degenerate spectrum; the Jordan-block parameter that
    also enters the Type II conditions is not extracted.
    """
    T_mixed = local_mixed_stress_energy(energy_tensor, metric)
    grid_shape = T_mixed.shape[2:]
    num_points = int(np.prod(grid_shape))

    matrices = np.moveaxis(T_mixed.reshape(4, 4, num_points), 2, 0)
    eigenvalues, eigenvectors = np.linalg.eig(matrices)

    # Scale from the matrix norm, not the spectrum: nilpotent (Type
    # II/III) matrices have zero eigenvalues but order-one entries.
    tensor_scale = np.linalg.norm(matrices, axis=(1, 2))
    floor = tolerance * max(tensor_scale.max(), 1.0)
    # A defective (Jordan) root of block size k perturbs by O(eps^(1/k)),
    # so Type II/III points show spurious complex pairs up to
    # ~eps^(1/3) * scale; a genuine Type IV pair must exceed that.
    defective_noise = np.finfo(float).eps ** (1.0 / 3.0)
    threshold = np.maximum(np.maximum(tolerance, defective_noise) * tensor_scale, floor)

    imag_mag = np.abs(eigenvalues.imag).max(axis=1)
    is_type_iv = imag_mag > threshold

    # eta-norms of the (Euclidean-normalized) eigenvectors; a genuinely
    # complex eigenvector cannot be timelike, so the real part is safe
    # for the real-spectrum points this is used on.
    vec_real = eigenvectors.real
    vec_norm2 = np.sum(vec_real**2, axis=1)
    vec_norm2 = np.where(vec_norm2 == 0.0, 1.0, vec_norm2)
    eta_norms = np.einsum("m,nmk->nk", _ETA_DIAG, vec_real**2) / vec_norm2

    has_timelike = np.any(eta_norms < -tolerance, axis=1) & ~is_type_iv
    rho_index = np.argmin(eta_norms, axis=1)

    point_index = np.arange(num_points)
    rho_flat = -eigenvalues[point_index, rho_index].real

    keep = np.ones((num_points, 4), dtype=bool)
    keep[point_index, rho_index] = False
    pressures_flat = eigenvalues.real[keep].reshape(num_points, 3)

    type_flat = np.ones(num_points, dtype=int)
    degenerate = ~is_type_iv & ~has_timelike
    type_flat[degenerate] = 2
    nilpotent_flat = np.zeros(num_points)
    if np.any(degenerate):
        deg_idx = np.flatnonzero(degenerate)
        cubic = _has_cubic_jordan_block(
            matrices[deg_idx], rho_flat[deg_idx] * -1.0, tolerance
        )
        type_flat[deg_idx[cubic]] = 3
        cubic_idx = deg_idx[cubic]
        shifted = matrices[cubic_idx] + rho_flat[cubic_idx][
            :, np.newaxis, np.newaxis
        ] * np.eye(4)
        nilpotent_flat[cubic_idx] = np.linalg.norm(shifted, axis=(1, 2))
    type_flat[is_type_iv] = 4

    jordan_flat = np.zeros(num_points)
    type_ii_idx = np.flatnonzero(type_flat == 2)
    if len(type_ii_idx) > 0:
        jordan_flat[type_ii_idx] = _jordan_block_parameter(
            matrices[type_ii_idx],
            vec_real[type_ii_idx, :, rho_index[type_ii_idx]],
        )

    complex_flat = np.where(is_type_iv, imag_mag, 0.0)

    return HawkingEllisResult(
        type_map=type_flat.reshape(grid_shape),
        rho=rho_flat.reshape(grid_shape),
        pressures=np.moveaxis(pressures_flat, 1, 0).reshape((3,) + grid_shape),
        eigenvalues=np.moveaxis(eigenvalues, 1, 0).reshape((4,) + grid_shape),
        complex_magnitude=complex_flat.reshape(grid_shape),
        eigenvectors=np.moveaxis(eigenvectors, 0, 2).reshape((4, 4) + grid_shape),
        rho_index=rho_index.reshape(grid_shape),
        jordan_parameter=jordan_flat.reshape(grid_shape),
        nilpotent_magnitude=nilpotent_flat.reshape(grid_shape),
    )


def invariant_energy_conditions(
    energy_tensor: SpacetimeTensor,
    metric: SpacetimeTensor,
    condition: str,
    tolerance: float = 1e-9,
    classification: Optional[HawkingEllisResult] = None,
) -> np.ndarray:
    """Observer-independent energy-condition margin map.

    Counterpart of get_energy_conditions that holds for ALL observers
    at once instead of a sampled family. Margins are the exact
    eigenvalue inequalities at Type I points:

    - null: min_i(rho + p_i)
    - weak: min(rho, null margin)
    - strong: min(null margin, rho + p1 + p2 + p3)
    - dominant: min_i(rho - |p_i|)

    Type IV points violate every condition unconditionally; their
    margin is -|Im| of the complex pair as a severity scale. Type II
    points apply the Type I formulas to the degenerate (Segre)
    spectrum AND require the Jordan-block parameter f >= 0
    (Martin-Moruno & Visser Sec 4.2: every Type II energy condition
    demands it), so the margin is the minimum of the two -- a
    negative-energy null dust with vanishing eigenvalues is correctly
    flagged. Type III points violate every pointwise condition (the
    cubic nilpotent piece must vanish for any of them to hold); their
    margin is -|nilpotent| as a severity scale.

    Parameters
    ----------
    energy_tensor : SpacetimeTensor
        Stress-energy tensor on the grid
    metric : SpacetimeTensor
        Metric the stress-energy was solved from
    condition : str
        "Null", "Weak", "Dominant", or "Strong"
    tolerance : float
        Passed to hawking_ellis_classify when classification is None
    classification : HawkingEllisResult, optional
        Precomputed classification to avoid a second eigensolve

    Returns
    -------
    np.ndarray
        Margin at every grid point (negative values indicate a
        violation seen by some observer); shape grid_shape
    """
    condition = condition.lower()
    if condition not in CONDITIONS:
        raise ValueError(
            "Incorrect energy condition input, use either: "
            '"Null", "Weak", "Dominant", "Strong"'
        )
    if classification is None:
        classification = hawking_ellis_classify(energy_tensor, metric, tolerance)

    rho = classification.rho
    pressures = classification.pressures

    null_margin = np.min(rho[np.newaxis] + pressures, axis=0)
    if condition == "null":
        margin = null_margin
    elif condition == "weak":
        margin = np.minimum(rho, null_margin)
    elif condition == "strong":
        margin = np.minimum(null_margin, rho + np.sum(pressures, axis=0))
    else:  # dominant
        margin = np.min(rho[np.newaxis] - np.abs(pressures), axis=0)

    type_ii = classification.type_map == 2
    margin = np.where(
        type_ii, np.minimum(margin, classification.jordan_parameter), margin
    )
    type_iii = classification.type_map == 3
    margin = np.where(type_iii, -classification.nilpotent_magnitude, margin)
    type_iv = classification.type_map == 4
    return np.where(type_iv, -classification.complex_magnitude, margin)


def type_i_witnesses(
    classification: HawkingEllisResult, metric: SpacetimeTensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form worst observers at Type I points, in coordinates.

    At a Type I point the eigenframe is an orthonormal tetrad
    {e_0, e_1, e_2, e_3} and the null contraction for
    k = e_0 + sum_i c_i e_i (sum c_i^2 = 1) is rho + sum_i c_i^2 p_i,
    minimized by putting all weight on the smallest pressure. The
    worst null direction is therefore k = e_0 + e_min in closed form,
    and T_ab k^a k^b = rho + p_min reproduces the invariant null
    margin exactly. Local-frame vectors map to coordinates with the
    Eulerian transformation M (v_coord = M v_local, since
    M^T g M = eta).

    Parameters
    ----------
    classification : HawkingEllisResult
        Output of hawking_ellis_classify (its eigenvectors are reused;
        no second eigensolve happens here)
    metric : SpacetimeTensor
        Metric the classification was computed against

    Returns
    -------
    observer : np.ndarray
        Future-directed unit timelike eigenframe observer u^a in
        coordinate components, shape (4,) + grid_shape; the observer
        measuring the invariant energy density rho
    null_witness : np.ndarray
        Null vector k^a in coordinate components attaining the
        invariant null margin, shape (4,) + grid_shape

    Both are NaN at non-Type-I points, where no timelike eigenvector
    (and hence no closed-form witness) exists.
    """
    grid_shape = classification.type_map.shape
    num_points = int(np.prod(grid_shape))

    eigenvectors = classification.eigenvectors.real.reshape(4, 4, num_points)
    rho_index = classification.rho_index.reshape(num_points)
    pressures = classification.pressures.reshape(3, num_points)

    point_index = np.arange(num_points)
    e_time = eigenvectors[:, rho_index, point_index]

    # Unit-normalize in eta and orient future (positive local t
    # component; M preserves time orientation since M[0,0] > 0).
    time_norm2 = -np.einsum("m,mk->k", _ETA_DIAG, e_time**2)
    e_time = e_time / np.sqrt(np.abs(time_norm2))
    e_time = e_time * np.sign(e_time[0])

    spatial_columns = np.empty((4, 3, num_points))
    keep = np.ones((num_points, 4), dtype=bool)
    keep[point_index, rho_index] = False
    for row in range(4):
        spatial_columns[row] = eigenvectors[row].T[keep].reshape(num_points, 3).T
    min_index = np.argmin(pressures, axis=0)
    e_min = spatial_columns[:, min_index, point_index]
    # eta-Gram-Schmidt against e_time: inside a degenerate eigenspace
    # the returned vectors need not be eta-orthogonal, and the witness
    # is only exactly null when eta(e_time, e_min) = 0.
    overlap = np.einsum("m,mk,mk->k", _ETA_DIAG, e_time, e_min)
    e_min = e_min + overlap * e_time
    residual2 = np.abs(np.einsum("m,mk->k", _ETA_DIAG, e_min**2))
    # Isotropic-within-noise points (e.g. numerical vacuum) annihilate
    # under the projection; every null direction is equally worst there,
    # so complete the tetrad with the projected local x axis instead.
    degenerate_dir = residual2 < np.finfo(float).eps
    if np.any(degenerate_dir):
        axis = np.zeros((4, int(degenerate_dir.sum())))
        axis[1] = 1.0
        overlap = np.einsum("m,mk,mk->k", _ETA_DIAG, e_time[:, degenerate_dir], axis)
        e_min[:, degenerate_dir] = axis + overlap * e_time[:, degenerate_dir]
        residual2 = np.abs(np.einsum("m,mk->k", _ETA_DIAG, e_min**2))
    e_min = e_min / np.sqrt(residual2)

    g_cov = change_tensor_index(metric, "covariant")
    M = eulerian_transformation_matrix(np.asarray(g_cov.tensor, dtype=float))
    M_flat = M.reshape(4, 4, num_points)

    observer = np.einsum("mnk,nk->mk", M_flat, e_time)
    null_witness = np.einsum("mnk,nk->mk", M_flat, e_time + e_min)

    not_type_i = (classification.type_map != 1).reshape(num_points)
    observer[:, not_type_i] = np.nan
    null_witness[:, not_type_i] = np.nan

    return (
        observer.reshape((4,) + grid_shape),
        null_witness.reshape((4,) + grid_shape),
    )
