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
    """Test: Total population grows by exp(rho * t) regardless of D."""
    config_3d['physics']['D'] = 0.2  # Include diffusion to ensure it doesn't break mass
    model = SpectralNDE3D(config_3d)
    sol = run_pde_solver(model, config_3d)
    
    pop_start = np.sum(sol.y[:, 0]) * model.dv
    pop_end = np.sum(sol.y[:, -1]) * model.dv
    
    expected_pop = pop_start * np.exp(config_3d['physics']['rho'] * 2.0)
    
    # Check that growth matches analytical expectation
    assert pytest.approx(pop_end, rel=1e-3) == expected_pop

def test_3d_diffusion_greens_function(config_3d):
    """
    Test: Compare 3D point source diffusion against 3D Green's function.
    N(r,t) = A / (4*pi*D*t)^1.5 * exp(-r^2 / 4Dt)
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
    mid = model.nodes // 2
    numerical_peak = N_final[mid, mid, mid]

    # Discretization of delta function leads to some error; check within 1%
    assert pytest.approx(numerical_peak, rel=0.01) == analytical_peak