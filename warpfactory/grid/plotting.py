"""Slice plotting for grid tensors.

Ports of MATLAB plotTensor.m, plotThreePlusOne.m and getSliceData.m on
matplotlib. Each plot function returns the created Figure objects so
callers (and tests) can inspect or save them without a display.
"""

from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..visualizer.colormaps import ColorMaps
from .tensor import SpacetimeTensor, verify_tensor
from .three_plus_one import three_plus_one_decomposer

_AXIS_LABELS = ("t", "x", "y", "z")


def get_slice_data(
    tensor_field: np.ndarray,
    sliced_planes: Sequence[int],
    slice_locations: Sequence[int],
) -> np.ndarray:
    """Extract a 2-D slice from a 4-D grid component.

    Parameters
    ----------
    tensor_field : np.ndarray
        Grid array of shape (Nt, Nx, Ny, Nz)
    sliced_planes : sequence of 2 ints
        The two coordinate axes (0..3) held fixed
    slice_locations : sequence of 2 ints
        Grid indices along the fixed axes

    Returns
    -------
    np.ndarray
        The remaining two axes as a 2-D array (in ascending axis order)
    """
    if sliced_planes[0] == sliced_planes[1]:
        raise ValueError("sliced planes must be two different axes")
    index = [slice(None)] * 4
    for axis, location in zip(sliced_planes, slice_locations):
        if not 0 <= location < tensor_field.shape[axis]:
            raise ValueError(
                f"slice location {location} outside axis {axis} "
                f"of size {tensor_field.shape[axis]}"
            )
        index[axis] = location
    return tensor_field[tuple(index)]


def _shown_axes(sliced_planes: Sequence[int]) -> Tuple[int, int]:
    remaining = sorted(set(range(4)) - set(sliced_planes))
    return remaining[0], remaining[1]


def _default_slice(grid_shape, sliced_planes):
    return [grid_shape[axis] // 2 for axis in sliced_planes]


def _plot_slice(data2d: np.ndarray, title: str, xlabel: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots()
    im = ax.pcolormesh(data2d.T, cmap=ColorMaps().redblue(), shading="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def plot_tensor(
    tensor: SpacetimeTensor,
    sliced_planes: Sequence[int] = (0, 3),
    slice_locations: Optional[Sequence[int]] = None,
) -> List[plt.Figure]:
    """Plot the unique components of a grid tensor on a 2-D slice.

    Port of plotTensor.m: covariant/contravariant tensors plot the 10
    unique symmetric components; mixed-index tensors plot all 16.
    """
    if not verify_tensor(tensor):
        raise ValueError("Tensor failed verification")

    if slice_locations is None:
        slice_locations = _default_slice(tensor.grid_shape, sliced_planes)

    symbol = "g" if tensor.type.lower() == "metric" else "T"
    index = tensor.index.lower()
    if index in ("mixedupdown", "mixeddownup"):
        components = [(mu, nu) for mu in range(4) for nu in range(4)]
    else:
        components = [(mu, nu) for mu in range(4) for nu in range(mu, 4)]

    ax1, ax2 = _shown_axes(sliced_planes)
    figures = []
    for mu, nu in components:
        data2d = get_slice_data(tensor.tensor[mu, nu], sliced_planes, slice_locations)
        if index == "covariant":
            title = f"{symbol}_{{{_AXIS_LABELS[mu]}{_AXIS_LABELS[nu]}}}"
        elif index == "contravariant":
            title = f"{symbol}^{{{_AXIS_LABELS[mu]}{_AXIS_LABELS[nu]}}}"
        elif index == "mixedupdown":
            title = f"{symbol}^{{{_AXIS_LABELS[mu]}}}_{{{_AXIS_LABELS[nu]}}}"
        else:
            title = f"{symbol}_{{{_AXIS_LABELS[mu]}}}^{{{_AXIS_LABELS[nu]}}}"
        figures.append(_plot_slice(data2d, title, _AXIS_LABELS[ax1], _AXIS_LABELS[ax2]))
    return figures


def plot_three_plus_one(
    metric: SpacetimeTensor,
    sliced_planes: Sequence[int] = (0, 3),
    slice_locations: Optional[Sequence[int]] = None,
) -> List[plt.Figure]:
    """Plot the ADM lapse, shift and spatial metric on a 2-D slice.

    Port of plotThreePlusOne.m: one figure for alpha, three for beta_i,
    six for the unique gamma_ij.
    """
    if metric.type.lower() != "metric":
        raise ValueError("plot_three_plus_one requires a metric tensor")
    if not verify_tensor(metric):
        raise ValueError("Metric failed verification")

    if slice_locations is None:
        slice_locations = _default_slice(metric.grid_shape, sliced_planes)

    alpha, beta_down, gamma_down, _, _ = three_plus_one_decomposer(metric)
    ax1, ax2 = _shown_axes(sliced_planes)
    xlabel, ylabel = _AXIS_LABELS[ax1], _AXIS_LABELS[ax2]

    figures = [
        _plot_slice(
            get_slice_data(alpha, sliced_planes, slice_locations),
            r"$\alpha$",
            xlabel,
            ylabel,
        )
    ]
    for i in range(3):
        figures.append(
            _plot_slice(
                get_slice_data(beta_down[i], sliced_planes, slice_locations),
                rf"$\beta_{i + 1}$",
                xlabel,
                ylabel,
            )
        )
    for i in range(3):
        for j in range(i, 3):
            figures.append(
                _plot_slice(
                    get_slice_data(gamma_down[i, j], sliced_planes, slice_locations),
                    rf"$\gamma_{{{i + 1}{j + 1}}}$",
                    xlabel,
                    ylabel,
                )
            )
    return figures
