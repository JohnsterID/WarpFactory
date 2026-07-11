"""Spacetime tensor container and index management for 4-D grids.

Python equivalent of the MATLAB tensor struct plus verifyTensor.m and
changeTensorIndex.m. A grid tensor stores all 16 components as a single
array of shape (4, 4, Nt, Nx, Ny, Nz) with coordinate order (t, x, y, z).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..solver.tensor_utils import inverse_tensor

VALID_INDICES = ("covariant", "contravariant", "mixedupdown", "mixeddownup")


@dataclass
class SpacetimeTensor:
    """A rank-2 tensor field sampled on a uniform 4-D grid.

    Attributes
    ----------
    tensor : np.ndarray
        Component array of shape (4, 4) + grid_shape, grid_shape being
        (Nt, Nx, Ny, Nz)
    type : str
        "metric" or "stress-energy"
    index : str
        One of "covariant", "contravariant", "mixedupdown", "mixeddownup"
    coords : str
        Coordinate system; only "cartesian" is supported
    scaling : Tuple[float, float, float, float]
        Uniform grid spacing (dt, dx, dy, dz)
    name : str
        Human-readable metric/tensor name
    params : Dict
        Construction parameters recorded by the metric builders
    """

    tensor: np.ndarray
    type: str = "metric"
    index: str = "covariant"
    coords: str = "cartesian"
    scaling: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    name: str = ""
    params: Dict = field(default_factory=dict)

    @property
    def grid_shape(self) -> Tuple[int, ...]:
        return self.tensor.shape[2:]


def verify_tensor(t: SpacetimeTensor, quiet: bool = True) -> bool:
    """Validate a SpacetimeTensor's structure (MATLAB verifyTensor).

    Returns True when the tensor has a known type, a (4, 4) + 4-D grid
    component array, cartesian coordinates, and a valid index.
    """
    problems = []
    if t.type.lower() not in ("metric", "stress-energy"):
        problems.append(f"unknown tensor type '{t.type}'")
    arr = np.asarray(t.tensor)
    if arr.ndim != 6 or arr.shape[:2] != (4, 4):
        problems.append(
            f"tensor must have shape (4, 4, Nt, Nx, Ny, Nz), got {arr.shape}")
    if t.coords.lower() != "cartesian":
        problems.append(f"unsupported coordinates '{t.coords}'")
    if t.index.lower() not in VALID_INDICES:
        problems.append(f"unknown index '{t.index}'")
    if len(t.scaling) != 4:
        problems.append("scaling must have 4 entries (dt, dx, dy, dz)")

    if problems and not quiet:
        for p in problems:
            print(f"verify_tensor: {p}")
    return not problems


def _raise_or_lower_both(tensor: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """out_ij = sum_ab in_ab transform_ai transform_bj."""
    return np.einsum("ab...,ai...,bj...->ij...", tensor, transform, transform)


def _mix_first(tensor: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """out_ij = sum_a in_aj transform_ai (change first index)."""
    return np.einsum("aj...,ai...->ij...", tensor, transform)


def _mix_second(tensor: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """out_ij = sum_a in_ia transform_aj (change second index)."""
    return np.einsum("ia...,aj...->ij...", tensor, transform)


def change_tensor_index(input_tensor: SpacetimeTensor, index: str,
                        metric: Optional[SpacetimeTensor] = None) -> SpacetimeTensor:
    """Return a copy of the tensor re-expressed in the requested index.

    Python port of changeTensorIndex.m. For metric tensors only
    covariant <-> contravariant conversions are allowed (pointwise
    matrix inverse). Other tensors require the metric as third argument
    and support all conversions between covariant, contravariant and
    both mixed forms.
    """
    index = index.lower()
    if index not in VALID_INDICES:
        raise ValueError(
            f"index must be one of {VALID_INDICES}, got '{index}'")

    current = input_tensor.index.lower()

    if input_tensor.type.lower() == "metric":
        if index in ("mixedupdown", "mixeddownup") or \
                current in ("mixedupdown", "mixeddownup"):
            raise ValueError("Metric tensors cannot use mixed indices")
        if current == index:
            new_array = input_tensor.tensor.copy()
        else:
            new_array = inverse_tensor(input_tensor.tensor)
        return SpacetimeTensor(
            tensor=new_array, type=input_tensor.type, index=index,
            coords=input_tensor.coords, scaling=input_tensor.scaling,
            name=input_tensor.name, params=dict(input_tensor.params))

    if metric is None:
        raise ValueError(
            "metric is required when changing the index of non-metric tensors")
    if metric.index.lower() in ("mixedupdown", "mixeddownup"):
        raise ValueError("Metric tensor cannot be used in mixed index")

    g_down, g_up = _metric_both_forms(metric)

    if current == index:
        new_array = input_tensor.tensor.copy()
    elif (current, index) == ("covariant", "contravariant"):
        new_array = _raise_or_lower_both(input_tensor.tensor, g_up)
    elif (current, index) == ("contravariant", "covariant"):
        new_array = _raise_or_lower_both(input_tensor.tensor, g_down)
    elif (current, index) == ("contravariant", "mixedupdown"):
        new_array = _mix_second(input_tensor.tensor, g_down)
    elif (current, index) == ("contravariant", "mixeddownup"):
        new_array = _mix_first(input_tensor.tensor, g_down)
    elif (current, index) == ("covariant", "mixedupdown"):
        new_array = _mix_first(input_tensor.tensor, g_up)
    elif (current, index) == ("covariant", "mixeddownup"):
        new_array = _mix_second(input_tensor.tensor, g_up)
    elif (current, index) == ("mixedupdown", "contravariant"):
        new_array = _mix_second(input_tensor.tensor, g_up)
    elif (current, index) == ("mixedupdown", "covariant"):
        new_array = _mix_first(input_tensor.tensor, g_down)
    elif (current, index) == ("mixeddownup", "covariant"):
        new_array = _mix_second(input_tensor.tensor, g_down)
    elif (current, index) == ("mixeddownup", "contravariant"):
        new_array = _mix_first(input_tensor.tensor, g_up)
    else:
        raise ValueError(
            f"unsupported index conversion '{current}' -> '{index}'")

    return SpacetimeTensor(
        tensor=new_array, type=input_tensor.type, index=index,
        coords=input_tensor.coords, scaling=input_tensor.scaling,
        name=input_tensor.name, params=dict(input_tensor.params))


def _metric_both_forms(metric: SpacetimeTensor) -> Tuple[np.ndarray, np.ndarray]:
    if metric.index.lower() == "covariant":
        g_down = metric.tensor
        g_up = inverse_tensor(g_down)
    else:
        g_up = metric.tensor
        g_down = inverse_tensor(g_up)
    return g_down, g_up
