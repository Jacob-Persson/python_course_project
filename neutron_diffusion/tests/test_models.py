# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:21:34 2026

@author: jacpe396
"""

import numpy as np
import pytest
from models.diffusion import SpectralNDE3D

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