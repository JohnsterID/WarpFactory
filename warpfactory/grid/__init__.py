"""Full 4-D grid pipeline: metrics, EFE solver, index management, plotting.

Python port of the MATLAB WarpFactory grid workflow (metricGet_* ->
getEnergyTensor -> plotTensor) in geometric units (G = c = 1).
"""

from .energy_conditions import (
    do_frame_transfer,
    eulerian_transformation_matrix,
    even_points_on_sphere,
    generate_uniform_field,
    get_energy_conditions,
)
from .interpolation import (
    legendre_radial_interp,
    quadrilinear_interp,
    trilinear_interp,
)
from .metrics import (
    alcubierre_comoving_metric,
    alcubierre_metric,
    lentz_comoving_metric,
    lentz_metric,
    minkowski_metric,
    modified_time_comoving_metric,
    modified_time_metric,
    schwarzschild_metric,
    van_den_broeck_comoving_metric,
    van_den_broeck_metric,
)
from .plotting import get_slice_data, plot_tensor, plot_three_plus_one
from .scalars import eulerian_velocity, get_scalars
from .shape_functions import alcubierre_shape, compact_sigmoid
from .si_units import (
    si_energy_factor,
    stress_energy_to_geometric,
    stress_energy_to_si,
)
from .solver import GridSolver
from .tensor import SpacetimeTensor, change_tensor_index, verify_tensor
from .three_plus_one import (
    minkowski_three_plus_one,
    three_plus_one_builder,
    three_plus_one_decomposer,
)
from .warp_shell import (
    alpha_numeric_solver,
    sph2cart_diag,
    tov_constant_density_pressure,
    warp_shell_comoving_metric,
)

__all__ = [
    "SpacetimeTensor",
    "verify_tensor",
    "change_tensor_index",
    "GridSolver",
    "minkowski_metric",
    "alcubierre_metric",
    "alcubierre_comoving_metric",
    "lentz_metric",
    "lentz_comoving_metric",
    "van_den_broeck_metric",
    "van_den_broeck_comoving_metric",
    "modified_time_metric",
    "modified_time_comoving_metric",
    "schwarzschild_metric",
    "warp_shell_comoving_metric",
    "tov_constant_density_pressure",
    "alpha_numeric_solver",
    "sph2cart_diag",
    "minkowski_three_plus_one",
    "three_plus_one_builder",
    "three_plus_one_decomposer",
    "alcubierre_shape",
    "compact_sigmoid",
    "get_scalars",
    "eulerian_velocity",
    "si_energy_factor",
    "stress_energy_to_si",
    "stress_energy_to_geometric",
    "get_energy_conditions",
    "do_frame_transfer",
    "eulerian_transformation_matrix",
    "generate_uniform_field",
    "even_points_on_sphere",
    "trilinear_interp",
    "quadrilinear_interp",
    "legendre_radial_interp",
    "get_slice_data",
    "plot_tensor",
    "plot_three_plus_one",
]
