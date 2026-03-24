# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:21:34 2026

@author: jacpe396
"""

import numpy as np
import pytest
from models.diffusion import LinearNDE
from solvers.integrator import run_pde_solver

@pytest.fixture
def base_config():
    """Provides a standard config dictionary for tests."""
    return {
        'physics': {'D': 1.0, 'rho': 0.0, 'L': 10.0, 'nx': 101},
        'initial_condition': {'type': 'uniform', 'amplitude': 1.0}
    }

def test_laplacian_zero(base_config):
    """The Laplacian of a constant (uniform) field should be zero."""
    model = LinearNDE(base_config)
    N = model.get_initial_condition() # All 1.0s
    rhs = model.compute_rhs(0, N)
    # With rho=0, dN/dt should be 0 everywhere except possibly boundaries
    assert np.allclose(rhs[1:-1], 0.0)

def test_point_source_integral(base_config):
    """The integral of a point source IC should equal the amplitude."""
    base_config['initial_condition'] = {'type': 'point_source', 'amplitude': 5.0}
    model = LinearNDE(base_config)
    N0 = model.get_initial_condition()
    
    total_neutrons = np.sum(N0) * model.dx
    assert pytest.approx(total_neutrons) == 5.0

def test_invalid_ic_type(base_config):
    """Ensure the model raises an error for unsupported IC types."""
    base_config['initial_condition']['type'] = "nonexistent_type"
    model = LinearNDE(base_config)
    with pytest.raises(ValueError, match="Unknown initial condition type"):
        model.get_initial_condition()

def test_mass_conservation():
    """
    Test: With rho=0 and no neutrons reaching the boundaries, 
    the total integral of N(x) must be constant (Neutron Economy).
    """
    config = {
        'physics': {'D': 0.1, 'rho': 0.0, 'L': 100.0, 'nx': 200},
        'initial_condition': {'type': 'gaussian', 'amplitude': 1.0, 'width': 2.0},
        'simulation': {'t_end': 10.0, 'n_frames': 5, 'method': 'RK45'}
    }
    model = LinearNDE(config)
    sol = run_pde_solver(model, config)
    
    # Calculate total mass at each time step: Integral approx by sum(N)*dx
    masses = [np.sum(sol.y[:, i]) * model.dx for i in range(len(sol.t))]
    
    # Check that mass at t_end is the same as mass at t_start
    # We use a tight tolerance (1e-6) because mass should be very stable
    assert np.allclose(masses, masses[0], rtol=1e-6)