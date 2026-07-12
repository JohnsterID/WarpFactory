"""SI-unit conversion for grid stress-energy tensors.

The grid pipeline works in geometric units (G = c = 1) with all
coordinates in meters (the time axis is c*t), so the solver returns
T = G_munu / 8 pi with components in 1/m^2. The MATLAB original instead
returns SI components, T = (c^4 / 8 pi G) * G_munu in J/m^3. These
helpers convert between the two conventions at the API boundary so the
internal representation stays geometric.
"""

import numpy as np

from ..units import Constants
from .tensor import SpacetimeTensor

_CONSTANTS = Constants()


def si_energy_factor() -> float:
    """c^4 / G: J/m^3 per 1/m^2 of geometric stress-energy."""
    return _CONSTANTS.c**4 / _CONSTANTS.G


def stress_energy_to_si(stress_energy: SpacetimeTensor) -> SpacetimeTensor:
    """Convert a geometric-unit stress-energy tensor to SI (J/m^3).

    All components carry the same factor because every coordinate
    (including time) is in meters in the geometric convention.
    """
    if stress_energy.type.lower() != "stress-energy":
        raise ValueError(
            "stress_energy_to_si expects a stress-energy "
            f"tensor, got type '{stress_energy.type}'"
        )
    if stress_energy.params.get("units") == "si":
        return stress_energy
    params = dict(stress_energy.params)
    params["units"] = "si"
    return SpacetimeTensor(
        tensor=np.asarray(stress_energy.tensor) * si_energy_factor(),
        type=stress_energy.type,
        index=stress_energy.index,
        coords=stress_energy.coords,
        scaling=stress_energy.scaling,
        name=stress_energy.name,
        params=params,
        frame=stress_energy.frame,
    )


def stress_energy_to_geometric(stress_energy: SpacetimeTensor) -> SpacetimeTensor:
    """Convert an SI stress-energy tensor back to geometric units."""
    if stress_energy.type.lower() != "stress-energy":
        raise ValueError(
            "stress_energy_to_geometric expects a "
            "stress-energy tensor, got type "
            f"'{stress_energy.type}'"
        )
    if stress_energy.params.get("units") != "si":
        return stress_energy
    params = dict(stress_energy.params)
    params.pop("units")
    return SpacetimeTensor(
        tensor=np.asarray(stress_energy.tensor) / si_energy_factor(),
        type=stress_energy.type,
        index=stress_energy.index,
        coords=stress_energy.coords,
        scaling=stress_energy.scaling,
        name=stress_energy.name,
        params=params,
        frame=stress_energy.frame,
    )
