# -*- coding: utf-8 -*-
import copy

import numpy as np
import pytest
from models.diffusion import SpectralNDE3D, DelayedNeutronSpectralNDE3D
from numpy.fft import fftfreq

@pytest.fixture
def config_3d_basic():
    """Basic 3D configuration for unit testing."""
    return {
        'physics': {
            'D': 1.0, 
            'rho': 0.0,
            'sigma': 0.0,
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
    config = copy.deepcopy(config_3d_basic)
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
    config = copy.deepcopy(config_3d_basic)
    config["physics"]["rho"] = 0.0
    config["physics"]["D"] = 0.75
    config["physics"]["sigma"] = 0.0
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
    config = copy.deepcopy(config_3d_basic)
    config["initial_condition"] = {
        "type": "gaussian",
        "amplitude": 1.0,
        "width": 0.2,
        "center_offset": 2.5,
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
    config = copy.deepcopy(config_3d_basic)
    config["physics"]["rho"] = 0.25
    config["physics"]["D"] = 0.75
    config["physics"]["sigma"] = 0.0
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
    config = copy.deepcopy(config_3d_basic)
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


# --- Spatial reactivity profile tests ---

def test_spatial_rho_uniform_equals_scalar(config_3d_basic):
    """Uniform spatial rho must produce the same RHS as the scalar form."""
    scalar_cfg = copy.deepcopy(config_3d_basic)
    scalar_cfg["physics"]["rho"] = 0.25
    model_scalar = SpectralNDE3D(scalar_cfg)

    spatial_cfg = copy.deepcopy(config_3d_basic)
    spatial_cfg["physics"]["rho"] = {"type": "uniform", "value": 0.25}
    model_spatial = SpectralNDE3D(spatial_cfg)

    N = model_scalar.get_initial_condition()
    rhs_scalar = model_scalar.compute_rhs(0.0, N)
    rhs_spatial = model_spatial.compute_rhs(0.0, N)

    assert np.allclose(rhs_scalar, rhs_spatial)


def test_spatial_rho_step_values():
    """Step profile: inside and outside regions must have correct rho values."""
    cfg = {
        'physics': {
            'D': 1.0, 'rho': {
                'type': 'step', 'inside': 0.05, 'outside': -0.01, 'width': 3.0
            }, 'sigma': 0.0, 'L': [10.0, 10.0, 10.0], 'nodes': [51, 21, 21],
        },
        'initial_condition': {'type': 'uniform', 'amplitude': 1.0},
    }
    model = SpectralNDE3D(cfg)
    half = int(model.nodes[0] // 2)
    half_w = int(0.15 * model.nodes[0])
    rho = model.rho_field
    assert np.allclose(rho[half - half_w:half + half_w, :, :], 0.05)
    assert np.allclose(rho[0, :, :], -0.01)
    assert np.allclose(rho[-1, :, :], -0.01)


def test_spatial_rho_gaussian_symmetry():
    """Gaussian profile must be centred and symmetric about the domain centre."""
    cfg = {
        'physics': {
            'D': 1.0, 'rho': {
                'type': 'gaussian', 'peak': 0.05, 'background': -0.01, 'width': 2.0
            }, 'sigma': 0.0, 'L': [10.0, 10.0, 10.0], 'nodes': [31, 31, 31],
        },
        'initial_condition': {'type': 'uniform', 'amplitude': 1.0},
    }
    model = SpectralNDE3D(cfg)
    mid = tuple(d // 2 for d in model.nodes)
    cx, cy, cz = 0.5 * model.L
    dx, dy, dz = model.dx_vec
    # Nearest grid point to the true centre.
    i = int(np.round(cx / dx))
    j = int(np.round(cy / dy))
    k = int(np.round(cz / dz))
    d_sq = ((i * dx - cx)**2 + (j * dy - cy)**2 + (k * dz - cz)**2)
    sigma = 2.0
    expected = -0.01 + 0.06 * np.exp(-d_sq / (2 * sigma**2))
    assert model.rho_field[i, j, k] == pytest.approx(expected, abs=1e-10)
    assert model.rho_field[0, 0, 0] == pytest.approx(-0.01, abs=1e-3)
    # The maximum must be at the grid cell closest to the domain centre.
    mid_flat = np.argmax(model.rho_field.ravel())
    mid_idx = np.unravel_index(mid_flat, model.nodes)
    assert all(abs(mid_idx[d] - 0.5 * model.nodes[d]) <= 0.5 for d in range(3))


def test_spatial_rho_sinusoidal_mean():
    """Sinusoidal profile must have the correct spatial mean."""
    cfg = {
        'physics': {
            'D': 1.0, 'rho': {
                'type': 'sinusoidal', 'mean': 0.02, 'amplitude': 0.01,
                'freq': [1, 1, 1],
            }, 'sigma': 0.0, 'L': [10.0, 10.0, 10.0], 'nodes': [31, 31, 31],
        },
        'initial_condition': {'type': 'uniform', 'amplitude': 1.0},
    }
    model = SpectralNDE3D(cfg)
    assert float(np.mean(model.rho_field)) == pytest.approx(0.02, abs=1e-2)


def test_spatial_rho_radial_shell():
    """Radial profile: inner region must be close to rho_inner."""
    cfg = {
        'physics': {
            'D': 1.0, 'rho': {
                'type': 'radial', 'inner': 0.05, 'outer': -0.01,
                'radius': 3.0, 'sharpness': 50,
            }, 'sigma': 0.0, 'L': [20.0, 20.0, 20.0], 'nodes': [41, 41, 41],
        },
        'initial_condition': {'type': 'uniform', 'amplitude': 1.0},
    }
    model = SpectralNDE3D(cfg)
    mid = tuple(d // 2 for d in model.nodes)
    assert model.rho_field[mid] == pytest.approx(0.05, abs=1e-3)
    corner = (0, 0, 0)
    assert model.rho_field[corner] == pytest.approx(-0.01, abs=1e-3)


def test_spatial_rho_eigenmode_growth():
    """
    Spectral Laplacian eigenmode with uniform spatial rho must match
    the scalar-rho analytical growth rate.
    """
    cfg = {
        'physics': {
            'D': 0.75, 'rho': {'type': 'uniform', 'value': 0.25},
            'sigma': 0.0, 'L': [10.0, 10.0, 10.0], 'nodes': [21, 21, 21],
        },
        'initial_condition': {'type': 'uniform', 'amplitude': 1.0},
    }
    model = SpectralNDE3D(cfg)
    nodes = model.nodes.astype(int)
    D = model.D
    rho_val = model.rho

    kx = 2 * np.pi * fftfreq(nodes[0], d=model.dx_vec[0])
    ky = 2 * np.pi * fftfreq(nodes[1], d=model.dx_vec[1])
    kz = 2 * np.pi * fftfreq(nodes[2], d=model.dx_vec[2])
    i, j, k = 1, 0, 0
    k_mode_sq = kx[i] ** 2 + ky[j] ** 2 + kz[k] ** 2
    expected_eig = -D * k_mode_sq + rho_val

    N = np.cos(kx[i] * model.X + ky[j] * model.Y + kz[k] * model.Z)
    N_flat = N.flatten()
    rhs_flat = model.compute_rhs(0.0, N_flat)
    rhs = rhs_flat.reshape(nodes)
    mask = np.abs(N) > 1e-6
    ratio = rhs[mask] / N[mask]

    assert np.allclose(ratio, expected_eig, rtol=1e-2, atol=1e-4)


# --- Delayed neutron tests ---------------------------------------------------

def test_delayed_state_vector_length():
    """Delayed model must have a state vector twice the grid size."""
    cfg = {
        'physics': {
            'D': 1.0, 'rho': 0.01, 'sigma': 0.0,
            'beta': 0.1, 'lambda_D': 1.0,
            'L': [10.0, 10.0, 10.0], 'nodes': [16, 16, 16],
        },
        'initial_condition': {'type': 'uniform', 'amplitude': 1.0},
    }
    model = DelayedNeutronSpectralNDE3D(cfg)
    nc = int(np.prod(model.nodes))
    y0 = model.get_initial_condition()
    assert len(y0) == 2 * nc


def test_delayed_no_beta_is_standard():
    """With beta=0, delayed model RHS must match standard model."""
    cfg = {
        'physics': {
            'D': 1.0, 'rho': 0.01, 'sigma': -0.05,
            'beta': 0.0, 'lambda_D': 0.0,
            'L': [10.0, 10.0, 10.0], 'nodes': [16, 16, 16],
        },
        'initial_condition': {'type': 'uniform', 'amplitude': 1.0},
    }
    model_d = DelayedNeutronSpectralNDE3D(cfg)
    model_s = SpectralNDE3D(cfg)
    nc = int(np.prod(model_d.nodes))

    N0 = model_s.get_initial_condition()
    M0 = np.zeros_like(N0)
    y0 = np.concatenate([N0.flatten(), M0.flatten()])

    rhs_d = model_d.compute_rhs(0.0, y0)
    rhs_s = model_s.compute_rhs(0.0, N0)

    assert np.allclose(rhs_d[:nc], rhs_s)
    assert np.allclose(rhs_d[nc:], 0.0)


def test_delayed_precursor_build_up():
    """Precursor density M must grow when beta > 0 and N > 0."""
    cfg = {
        'physics': {
            'D': 1.0, 'rho': 0.01, 'sigma': 0.0,
            'beta': 0.1, 'lambda_D': 0.5,
            'L': [10.0, 10.0, 10.0], 'nodes': [21, 21, 21],
        },
        'initial_condition': {'type': 'uniform', 'amplitude': 1.0},
    }
    model = DelayedNeutronSpectralNDE3D(cfg)
    nc = int(np.prod(model.nodes))
    y0 = model.get_initial_condition()
    rhs = model.compute_rhs(0.0, y0)
    dM_dt = rhs[nc:].reshape(model.nodes)
    N0 = y0[:nc].reshape(model.nodes)
    assert np.allclose(dM_dt, cfg['physics']['beta'] * N0)


def test_delayed_precursor_decay():
    """When N -> 0, M must decay as dM/dt = -lambda_D * M."""
    cfg = {
        'physics': {
            'D': 1.0, 'rho': -1.0, 'sigma': 0.0,
            'beta': 0.1, 'lambda_D': 2.0,
            'L': [10.0, 10.0, 10.0], 'nodes': [16, 16, 16],
        },
        'initial_condition': {'type': 'uniform', 'amplitude': 0.0,
                              'M_amplitude': 5.0},
    }
    model = DelayedNeutronSpectralNDE3D(cfg)
    nc = int(np.prod(model.nodes))
    y0 = model.get_initial_condition()
    rhs = model.compute_rhs(0.0, y0)
    dM_dt = rhs[nc:].reshape(model.nodes)
    expected = -cfg['physics']['lambda_D'] * y0[nc:].reshape(model.nodes)
    assert np.allclose(dM_dt, expected)


def test_delayed_noise_amplitude():
    """Precursor noise amplitude must scale as sqrt(2 * T2 * M / dv)."""
    cfg = {
        'physics': {
            'D': 1.0, 'rho': 0.0, 'sigma': 0.0,
            'beta': 0.01, 'lambda_D': 0.1, 'T2': 1.0,
            'L': [10.0, 10.0, 10.0], 'nodes': [16, 16, 16],
        },
        'initial_condition': {'type': 'uniform', 'amplitude': 1.0},
    }
    model = DelayedNeutronSpectralNDE3D(cfg)
    M = np.ones(model.nodes) * 4.0
    amp = model.compute_precursor_noise_amplitude(M)
    expected = np.sqrt(2.0 * model.T2 * 4.0 / model.dv)
    assert np.allclose(amp, expected)


def test_delayed_scalar_rho_works():
    """Delayed model must accept scalar rho and run a short integration."""
    cfg = {
        'physics': {
            'D': 1.0, 'rho': 0.05, 'sigma': 0.0,
            'beta': 0.05, 'lambda_D': 0.5,
            'L': [10.0, 10.0, 10.0], 'nodes': [16, 16, 16],
        },
        'initial_condition': {'type': 'uniform', 'amplitude': 1.0},
        'simulation': {'t_end': 0.5, 'n_frames': 2, 'method': 'RK45',
                       'rtol': 1e-3, 'atol': 1e-6},
    }
    from solvers.integrator import run_pde_solver
    model = DelayedNeutronSpectralNDE3D(cfg)
    sol = run_pde_solver(model, cfg)
    nc = int(np.prod(model.nodes))
    assert np.all(np.isfinite(sol.y[:nc]))
    assert np.all(sol.y[:nc] >= 0)
    assert np.all(np.isfinite(sol.y[nc:]))