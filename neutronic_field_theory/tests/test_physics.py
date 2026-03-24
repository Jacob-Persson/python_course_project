# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:26:37 2026

@author: jacpe396
"""

import numpy as np
import pytest
from models.diffusion import LinearNDE
from solvers.integrator import run_pde_solver

def test_exponential_growth():
    """
    Test: If D=0 and N is uniform, N(t) should follow N0 * exp(rho * t).
    This validates that the reaction term and the time-stepper work together.
    """
    # 1. Setup a "No-Diffusion" config
    rho = 0.1
    t_end = 5.0
    config = {
        'physics': {'D': 0.0, 'rho': rho, 'L': 10.0, 'nx': 50},
        'initial_condition': {'type': 'uniform', 'amplitude': 1.0},
        'simulation': {'t_end': t_end, 'n_frames': 2, 'method': 'RK45'}
    }
    
    # 2. Run simulation
    model = LinearNDE(config)
    sol = run_pde_solver(model, config)
    
    # 3. Calculate Analytical vs Numerical
    N_final_numerical = sol.y[:, -1]
    N_final_analytical = 1.0 * np.exp(rho * t_end)
    
    # 4. Assertion (allowing for small numerical integration error)
    assert np.allclose(N_final_numerical, N_final_analytical, rtol=1e-4)

def test_diffusion_widening():
    """
    Test: With rho=0 and a point source, the variance of the distribution 
    should increase over time (qualitative check for D > 0).
    """
    config = {
        'physics': {'D': 1.0, 'rho': 0.0, 'L': 50.0, 'nx': 200},
        'initial_condition': {'type': 'point_source', 'amplitude': 1.0},
        'simulation': {'t_end': 2.0, 'n_frames': 2, 'method': 'RK45'}
    }
    model = LinearNDE(config)
    sol = run_pde_solver(model, config)
    
    initial_peak = np.max(sol.y[:, 0])
    final_peak = np.max(sol.y[:, -1])
    
    # In pure diffusion, the peak must drop as the pulse spreads
    assert final_peak < initial_peak
    
def test_diffusion_analytical_gaussian():
    """
    Test: Compare numerical diffusion of a point source 
    against the analytical Green's function.
    """
    # 1. Setup Parameters
    D, L, t_end = 0.5, 50.0, 4.0
    amp = 10.0
    config = {
        'physics': {'D': D, 'rho': 0.0, 'L': L, 'nx': 500}, # High nx for accuracy
        'initial_condition': {'type': 'point_source', 'amplitude': amp},
        'simulation': {'t_end': t_end, 'n_frames': 2, 'method': 'RK45'}
    }
    
    model = LinearNDE(config)
    sol = run_pde_solver(model, config)
    
    # 2. Extract Numerical Result at t_end
    x = model.x
    x0 = L / 2
    N_numerical = sol.y[:, -1]
    
    # 3. Calculate Analytical Solution
    # N(x,t) = A / sqrt(4 * pi * D * t) * exp(-(x-x0)^2 / (4 * D * t))
    variance = 4 * D * t_end
    N_analytical = (amp / np.sqrt(np.pi * variance)) * np.exp(-(x - x0)**2 / variance)
    
    # 4. Compare (excluding boundaries where Dirichlet BCs cause divergence)
    inner = slice(50, -50) 
    # Use np.allclose or check the maximum error
    max_error = np.abs(N_numerical[inner] - N_analytical[inner]).max()
    
    # Tolerance is slightly higher due to discretization of the Dirac Delta
    assert max_error < 0.05 