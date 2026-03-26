# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:21:34 2026

@author: jacpe396
"""

import numpy as np
import pytest
from models.diffusion import SpectralNDE3D
from numpy.fft import fftfreq

@pytest.fixture
def config_3d_basic():
    """Basic 3D configuration for unit testing."""
    return {
        'physics': {
            'D': 1.0, 
            'rho': 0.0, 
            'L': [10.0, 10.0, 10.0], 
            'nodes': [21, 21, 21]
        },
        'initial_condition': {
            'type': 'uniform', 
            'amplitude': 1.0
        }
    }

def test_3d_laplacian_uniform(config_3d_basic):
    """The Laplacian of a constant 3D field must be zero everywhere."""
    model = SpectralNDE3D(config_3d_basic)
    N_initial = model.get_initial_condition() # Returns flattened 1.0s
    
    rhs = model.compute_rhs(0, N_initial)
    rhs_3d = rhs.reshape(model.nodes)
    
    # Internal points should be zero (rho=0, D*grad^2(1)=0)
    # We slice [1:-1] to avoid boundary condition effects
    assert np.allclose(rhs_3d[1:-1, 1:-1, 1:-1], 0.0)

def test_3d_point_source_integral(config_3d_basic):
    """The 3D volume integral of a point source should equal the amplitude."""
    config_3d_basic['initial_condition'] = {
        'type': 'point_source', 
        'amplitude': 7.5
    }
    model = SpectralNDE3D(config_3d_basic)
    N0_flat = model.get_initial_condition()
    
    total_neutrons = np.sum(N0_flat) * model.dv
    
    assert pytest.approx(total_neutrons) == 7.5

def test_3d_ic_shape(config_3d_basic):
    """Verify that the IC generator returns the correct flattened size."""
    model = SpectralNDE3D(config_3d_basic)
    N0 = model.get_initial_condition()
    
    expected_size = np.prod(config_3d_basic['physics']['nodes'])
    assert N0.size == expected_size


def test_3d_point_source_center_offset(config_3d_basic):
    """Point source should be placed near the requested offset center."""
    config = dict(config_3d_basic)
    config["initial_condition"] = {
        "type": "point_source",
        "amplitude": 3.0,
        "center_offset": 2.5,  # scalar applies to x/y/z
    }

    model = SpectralNDE3D(config)
    nodes = model.nodes.astype(int)

    # Expected grid indices using the same logic as in get_initial_condition():
    # mid = clip(round(center / dx_vec), 0, nodes-1), where center = L/2 + offset_vec.
    center = model.L / 2 + np.array([2.5, 2.5, 2.5], dtype=float)
    expected_mid = np.clip(
        np.round(center / model.dx_vec).astype(int),
        0,
        nodes - 1,
    )

    N0 = model.get_initial_condition().reshape(nodes)
    mid_from_data = np.array(np.unravel_index(np.argmax(N0), nodes))
    assert np.all(mid_from_data == expected_mid)


def test_3d_spectral_eigenmode_diffusion(config_3d_basic):
    r"""
    Spectral Laplacian eigenmode test.

    For a periodic FFT grid, the Laplacian eigenvalue relation is

    .. math::
        \nabla^2 N = -|k|^2\, N,

    so for :math:`\rho=0` the reaction term vanishes and

    .. math::
        \frac{dN}{dt} = D\nabla^2 N = -D|k|^2\,N.
    """
    config = dict(config_3d_basic)
    config["physics"] = dict(config_3d_basic["physics"])
    config["physics"]["rho"] = 0.0
    config["physics"]["D"] = 0.75
    model = SpectralNDE3D(config)

    nodes = model.nodes.astype(int)
    D = model.D

    # Build k-vectors consistently with the model:
    kx = 2 * np.pi * fftfreq(nodes[0], d=model.dx_vec[0])
    ky = 2 * np.pi * fftfreq(nodes[1], d=model.dx_vec[1])
    kz = 2 * np.pi * fftfreq(nodes[2], d=model.dx_vec[2])

    # Choose a non-zero mode in x only (y,z mode indices = 0).
    i, j, k = 1, 0, 0
    k_mode_sq = kx[i] ** 2 + ky[j] ** 2 + kz[k] ** 2
    expected_eig = -D * k_mode_sq

    # Real eigenfunction (cosine) to avoid dealing with complex arithmetic.
    N = np.cos(kx[i] * model.X + ky[j] * model.Y + kz[k] * model.Z)
    N_flat = N.flatten()

    rhs_flat = model.compute_rhs(0.0, N_flat)
    rhs = rhs_flat.reshape(nodes)

    # Avoid division by zero points where cosine is ~0.
    mask = np.abs(N) > 1e-6
    ratio = rhs[mask] / N[mask]

    assert np.allclose(ratio, expected_eig, rtol=1e-2, atol=1e-4)


def test_3d_gaussian_center_offset_peak(config_3d_basic):
    """Gaussian peak should be placed near the requested offset center."""
    config = dict(config_3d_basic)
    config["initial_condition"] = {
        "type": "gaussian",
        "amplitude": 1.0,
        "width": 0.2,  # narrower than grid spacing to make the discrete maximum robust
        "center_offset": 2.5,  # scalar applies to x/y/z
    }

    model = SpectralNDE3D(config)
    nodes = model.nodes.astype(int)

    N0 = model.get_initial_condition().reshape(nodes)
    mid_from_data = np.array(np.unravel_index(np.argmax(N0), nodes))

    center = model.L / 2 + np.array([2.5, 2.5, 2.5], dtype=float)
    expected_mid = np.clip(
        np.round(center / model.dx_vec).astype(int),
        0,
        nodes - 1,
    )

    assert np.all(mid_from_data == expected_mid)


def test_3d_spectral_eigenmode_growth(config_3d_basic):
    """
    Spectral Laplacian eigenmode test including the reaction term.

    For a periodic FFT eigenfunction ``N`` and constant coefficients:
        dN/dt = (-D|k|^2 + rho) N
    """
    config = dict(config_3d_basic)
    config["physics"] = dict(config_3d_basic["physics"])
    config["physics"]["rho"] = 0.25
    config["physics"]["D"] = 0.75
    model = SpectralNDE3D(config)

    nodes = model.nodes.astype(int)
    D = model.D
    rho = model.rho

    kx = 2 * np.pi * fftfreq(nodes[0], d=model.dx_vec[0])
    ky = 2 * np.pi * fftfreq(nodes[1], d=model.dx_vec[1])
    kz = 2 * np.pi * fftfreq(nodes[2], d=model.dx_vec[2])

    i, j, k = 1, 0, 0
    k_mode_sq = kx[i] ** 2 + ky[j] ** 2 + kz[k] ** 2
    expected_eig = -D * k_mode_sq + rho

    N = np.cos(kx[i] * model.X + ky[j] * model.Y + kz[k] * model.Z)
    N_flat = N.flatten()
    rhs_flat = model.compute_rhs(0.0, N_flat)
    rhs = rhs_flat.reshape(nodes)

    mask = np.abs(N) > 1e-6
    ratio = rhs[mask] / N[mask]

    assert np.allclose(ratio, expected_eig, rtol=1e-2, atol=1e-4)


def test_3d_radial_shell_integral_consistency(config_3d_basic):
    """Shell-weighted radial histogram should integrate to total mass (dv included)."""
    config = dict(config_3d_basic)
    config["initial_condition"] = {
        "type": "point_source",
        "amplitude": 7.0,
        "center_offset": 1.7,
    }

    model = SpectralNDE3D(config)
    nodes = model.nodes.astype(int)
    N0_3d = model.get_initial_condition().reshape(nodes)

    center_offset = model.ic_config.get("center_offset", 0.0)
    if np.isscalar(center_offset):
        offset_vec = np.array([float(center_offset)] * 3, dtype=float)
    else:
        offset_vec = np.array(center_offset, dtype=float)

    cx, cy, cz = (model.L / 2 + offset_vec)
    r = np.sqrt((model.X - cx) ** 2 + (model.Y - cy) ** 2 + (model.Z - cz) ** 2)

    r_max = np.max(r)
    r_bins = np.linspace(0, r_max, nodes[0])

    hist, _ = np.histogram(r, bins=r_bins, weights=N0_3d)
    shell_integral = float(np.sum(hist * model.dv))

    total_mass = float(np.sum(N0_3d) * model.dv)
    assert np.isclose(shell_integral, total_mass, rtol=1e-12, atol=1e-12)