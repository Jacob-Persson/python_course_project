# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:26:37 2026

@author: jacpe396
"""

import numpy as np
import pytest
from models.diffusion import SpectralNDE3D
from solvers.integrator import run_pde_solver

@pytest.fixture
def config_3d():
    """Standard 3D config for testing."""
    return {
        'physics': {
            'D': 0.5, 
            'rho': 0.1, 
            'L': [20.0, 20.0, 20.0], 
            'nodes': [31, 31, 31]
        },
        'initial_condition': {'type': 'point_source', 'amplitude': 10.0},
        'simulation': {'t_end': 2.0, 'n_frames': 2, 'method': 'RK45'}
    }

def test_3d_exponential_growth(config_3d):
    r"""
    Test: Total population grows by an exponential factor.

    For the linear model

    .. math::
        M(t) = M(0)\, e^{\rho t},

    where ``M(t)`` is the spatial integral of the neutron density and ``rho`` is
    the reactivity.
    """
    config_3d['physics']['D'] = 0.2  # Include diffusion to ensure it doesn't break mass
    model = SpectralNDE3D(config_3d)
    sol = run_pde_solver(model, config_3d)
    
    pop_start = np.sum(sol.y[:, 0]) * model.dv
    pop_end = np.sum(sol.y[:, -1]) * model.dv
    
    expected_pop = pop_start * np.exp(config_3d['physics']['rho'] * 2.0)
    
    # Check that growth matches analytical expectation
    assert pytest.approx(pop_end, rel=1e-3) == expected_pop

def test_3d_diffusion_greens_function(config_3d):
    r"""
    Test: Compare 3D point source diffusion against 3D Green's function.
    
    .. math::
        N(r,t) = \frac{A}{(4\pi D t)^{3/2}} \exp\left(-\frac{r^2}{4Dt}\right).

    """
    config_3d['physics']['rho'] = 0.0 # Pure diffusion
    t_end = config_3d['simulation']['t_end']
    D = config_3d['physics']['D']
    amp = config_3d['initial_condition']['amplitude']
    
    model = SpectralNDE3D(config_3d)
    sol = run_pde_solver(model, config_3d)
    
    # Analytical solution at center point
    # r=0 at the center where the source was placed
    # Analytical peak N(0,t) = A / (4 * pi * D * t)^1.5
    variance = 4 * np.pi * D * t_end
    analytical_peak = amp / (variance**1.5)
    
    # Numerical peak
    N_final = sol.y[:, -1].reshape(model.nodes)
    # On the periodic grid, the delta is placed into the grid cell whose
    # coordinate is closest to the physical source center at ``L/2``.
    center = model.L / 2
    nodes = model.nodes.astype(int)
    expected_mid = np.clip(
        np.round(center / model.dx_vec).astype(int),
        0,
        nodes - 1,
    )
    numerical_peak = N_final[expected_mid[0], expected_mid[1], expected_mid[2]]

    # Discretization of delta function leads to some error; check within 1%
    assert pytest.approx(numerical_peak, rel=0.01) == analytical_peak


def test_3d_pure_diffusion_mass_conservation(config_3d):
    r"""
    For pure diffusion (rho=0) on a periodic FFT grid, the total mass
    should be conserved:

    .. math::
        M(t) = \int N(\mathbf{x},t)\, dV \approx \sum_i N_i(t)\, dv.
    """
    config_3d = dict(config_3d)
    config_3d["physics"] = dict(config_3d["physics"])
    config_3d["physics"]["rho"] = 0.0

    model = SpectralNDE3D(config_3d)
    sol = run_pde_solver(model, config_3d)

    mass_start = float(np.sum(sol.y[:, 0]) * model.dv)
    mass_end = float(np.sum(sol.y[:, -1]) * model.dv)

    assert pytest.approx(mass_end, rel=1e-3) == mass_start


def test_3d_analytic_fourier_evolution_reference(config_3d):
    r"""
    Analytic reference test using Fourier-space diagonalization.

    For constant coefficients:

    .. math::
        \widehat{N}(\mathbf{k}, t) =
        e^{(-D|\mathbf{k}|^2 + \rho)\, t}\,\widehat{N}(\mathbf{k},0).
    """
    config_3d = dict(config_3d)
    config_3d["physics"] = dict(config_3d["physics"])
    config_3d["physics"]["D"] = 0.37
    config_3d["physics"]["rho"] = 0.18

    model = SpectralNDE3D(config_3d)

    # Use the model-provided initial condition and run the numerical solver.
    sol = run_pde_solver(model, config_3d)
    t_end = config_3d["simulation"]["t_end"]

    # Exact evolution in Fourier space:
    #   N_hat(t_end) = exp((-D*k_sq + rho)*t_end) * N_hat(0)
    N0_flat = model.get_initial_condition()
    N0 = N0_flat.reshape(model.nodes)
    N_hat0 = np.fft.fftn(N0)
    growth = np.exp((-model.D * model.k_sq + model.rho) * t_end)
    N_hat_exact = growth * N_hat0
    N_exact = np.real(np.fft.ifftn(N_hat_exact))

    N_num = sol.y[:, -1].reshape(model.nodes)

    # Compare fields. Tolerance accounts for numerical ODE integration error.
    denom = float(np.max(np.abs(N_exact)))
    err = float(np.max(np.abs(N_num - N_exact)))
    if denom > 0:
        assert err / denom < 1e-3
    else:
        assert err < 1e-8