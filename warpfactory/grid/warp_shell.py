"""Comoving Warp Shell metric (Helmerich & Fuchs, CQG 2024).

Port of metricGet_WarpShellComoving.m plus its utilities
TOVconstDensity.m, alphaNumericSolver.m and sph2cartDiag.m, in
geometric units (G = c = 1: masses are lengths).
"""

from typing import Sequence

import numpy as np

from .interpolation import legendre_radial_interp
from .metrics import _base_metric, _require_single_time_slice, _world_coordinates
from .shape_functions import compact_sigmoid
from .tensor import SpacetimeTensor


def tov_constant_density_pressure(R: float, M: np.ndarray, rho: np.ndarray,
                                  r: np.ndarray) -> np.ndarray:
    """Interior pressure of a constant-density TOV star (TOVconstDensity).

    P(r) = rho [ (R sqrt(R - 2M) - sqrt(R^3 - 2 M r^2))
                 / (sqrt(R^3 - 2 M r^2) - 3 R sqrt(R - 2M)) ]  for r < R
    with M the total mass (geometric units).
    """
    m_total = M[-1]
    outer = R*np.sqrt(R - 2*m_total)
    inner = np.sqrt(R**3 - 2*m_total*r**2)
    return rho*((outer - inner)/(inner - 3*outer))*(r < R)


def alpha_numeric_solver(M: np.ndarray, P: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Numeric lapse exponent a(r) for a static spherical star.

    Port of alphaNumericSolver.m: integrates
    da/dr = (M + 4 pi r^3 P) / (r^2 - 2 M r)
    outward with the trapezoid rule and offsets the result so that
    a(r_max) matches the Schwarzschild exterior value
    log(1 - 2 M_total/r_max)/2.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        dalpha = (M + 4*np.pi*r**3*P)/(r**2 - 2*M*r)
    dalpha[0] = 0.0
    dalpha[~np.isfinite(dalpha)] = 0.0

    alpha = np.concatenate(
        ([0.0], np.cumsum((dalpha[1:] + dalpha[:-1])/2*np.diff(r))))
    exterior = 0.5*np.log(1 - 2*M[-1]/r[-1])
    return alpha + (exterior - alpha[-1])


def sph2cart_diag(theta: np.ndarray, phi: np.ndarray, g11_sph: np.ndarray,
                  g22_sph: np.ndarray):
    """Transform a diagonal spherical metric (g_tt, g_rr) to Cartesian.

    Port of sph2cartDiag.m: assumes unit angular metric factors
    (g_theta_theta = r^2, g_phi_phi = r^2 sin^2 theta already divided
    out), returning the Cartesian components produced by the coordinate
    rotation.
    """
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(phi), np.cos(phi)
    # Match MATLAB's exact-zero handling at +-pi/2 to avoid residues of
    # order 1e-17 flipping signs in downstream products.
    cos_t = np.where(np.isclose(np.abs(theta), np.pi/2), 0.0, cos_t)
    cos_p = np.where(np.isclose(np.abs(phi), np.pi/2), 0.0, cos_p)

    E = g22_sph
    g11_cart = g11_sph
    g22_cart = E*cos_p**2*sin_t**2 + cos_p**2*cos_t**2 + sin_p**2
    g33_cart = E*sin_p**2*sin_t**2 + cos_t**2*sin_p**2 + cos_p**2
    g44_cart = E*cos_t**2 + sin_t**2
    g23_cart = E*cos_p*sin_p*sin_t**2 + cos_p*cos_t**2*sin_p - cos_p*sin_p
    g24_cart = E*cos_p*cos_t*sin_t - cos_p*cos_t*sin_t
    g34_cart = E*cos_t*sin_p*sin_t - cos_t*sin_p*sin_t
    return g11_cart, g22_cart, g23_cart, g24_cart, g33_cart, g34_cart, g44_cart


def smooth_profile(y: np.ndarray, span: float, passes: int = 1) -> np.ndarray:
    """Centered moving-average smoothing (MATLAB smooth), repeated.

    The window is the greatest odd integer <= span (minimum 1);
    endpoints use progressively shrinking centered windows exactly like
    MATLAB's smooth.
    """
    window = max(int(span), 1)
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return y.copy()

    result = y.astype(float)
    half = window//2
    for _ in range(passes):
        smoothed = np.convolve(result, np.ones(window)/window, mode="same")
        # Rebuild the endpoint values with shrinking centered windows.
        for i in range(half):
            smoothed[i] = result[:2*i + 1].mean()
            smoothed[-(i + 1)] = result[-(2*i + 1):].mean()
        result = smoothed
    return result


def warp_shell_comoving_metric(grid_size: Sequence[int],
                               world_center: Sequence[float],
                               m: float, R1: float, R2: float,
                               Rbuff: float = 0.0, sigma: float = 0.0,
                               smooth_factor: float = 1.0,
                               v_warp: float = 0.0, do_warp: bool = False,
                               grid_scale: Sequence[float] = (1, 1, 1, 1),
                               r_sample_res: int = 10**5) -> SpacetimeTensor:
    """Comoving Warp Shell metric (metricGet_WarpShellComoving).

    Builds a constant-density matter shell between radii R1 and R2 with
    a TOV-consistent pressure profile, solves the static spherically
    symmetric field equations numerically, and (optionally) applies the
    interior shift vector that produces the warp effect.

    Parameters mirror the MATLAB original; m is the total shell mass in
    geometric units (same length units as the grid).
    """
    _require_single_time_slice(grid_size, "warp_shell_comoving_metric")

    _, x, y, z = _world_coordinates(grid_size, world_center, grid_scale)
    world_radius = np.sqrt(
        ((grid_size[1] - 1)*grid_scale[1] - world_center[1])**2
        + ((grid_size[2] - 1)*grid_scale[2] - world_center[2])**2
        + ((grid_size[3] - 1)*grid_scale[3] - world_center[3])**2)
    r_sample = np.linspace(0, world_radius*1.2, r_sample_res)

    rho = np.where((r_sample > R1) & (r_sample < R2),
                   m/(4.0/3.0*np.pi*(R2**3 - R1**3)), 0.0)

    mass_profile = _cumtrapz(4*np.pi*rho*r_sample**2, r_sample)
    pressure = tov_constant_density_pressure(R2, mass_profile, rho, r_sample)

    rho = smooth_profile(rho, 1.79*smooth_factor, passes=4)
    pressure = smooth_profile(pressure, smooth_factor, passes=4)
    mass_profile = _cumtrapz(4*np.pi*rho*r_sample**2, r_sample)

    shift_profile = smooth_profile(
        compact_sigmoid(r_sample, R1, R2, sigma, Rbuff), smooth_factor, passes=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        B = 1.0/(1 - 2*mass_profile/r_sample)
    B[0] = 1.0

    a = alpha_numeric_solver(mass_profile, pressure, r_sample)
    A = -np.exp(2*a)

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    r_b, theta_b, phi_b = np.broadcast_arrays(r, theta, phi)

    dr = r_sample[1] - r_sample[0]
    frac_idx = r_b/dr

    g11_sph = legendre_radial_interp(A, frac_idx)
    g22_sph = legendre_radial_interp(B, frac_idx)
    shift = legendre_radial_interp(shift_profile, frac_idx)

    (g11_c, g22_c, g23_c, g24_c,
     g33_c, g34_c, g44_c) = sph2cart_diag(theta_b, phi_b, g11_sph, g22_sph)

    grid_shape = tuple(int(n) for n in grid_size)
    g = np.zeros((4, 4) + grid_shape)
    g[0, 0] = g11_c
    g[1, 1] = g22_c
    g[2, 2] = g33_c
    g[3, 3] = g44_c
    g[1, 2] = g[2, 1] = g23_c
    g[1, 3] = g[3, 1] = g24_c
    g[2, 3] = g[3, 2] = g34_c

    if do_warp:
        g[0, 1] = g[0, 1] - g[0, 1]*shift - shift*v_warp
        g[1, 0] = g[0, 1]

    metric = _base_metric(
        "Comoving Warp Shell", grid_scale,
        {"grid_size": tuple(grid_size), "world_center": tuple(world_center),
         "m": m, "R1": R1, "R2": R2, "Rbuff": Rbuff, "sigma": sigma,
         "smooth_factor": smooth_factor, "v_warp": v_warp,
         "do_warp": bool(do_warp)}, g)
    metric.params["rho"] = rho
    metric.params["P"] = pressure
    metric.params["M"] = mass_profile
    metric.params["r_vec"] = r_sample
    metric.params["A"] = A
    metric.params["B"] = B
    return metric


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum((y[1:] + y[:-1])/2*np.diff(x))))
