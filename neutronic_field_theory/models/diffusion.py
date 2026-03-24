# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:21:12 2026

@author: jacpe396
"""

import numpy as np
from .base_model import BaseModel

class LinearNDE(BaseModel):
    """
    Implementation of the Linear Neutron Diffusion Equation.
    Inherits from BaseModel to ensure compatibility with solvers.
    """
    def __init__(self, config):
        # Extract physics and grid parameters from config
        self.D = config['physics']['D']
        self.rho = config['physics']['rho']
        self.L = config['physics']['L']
        self.nx = config['physics']['nx']
        self.ic_config = config['initial_condition']
        
        # Setup grid
        self.x = np.linspace(0, self.L, self.nx)
        self.dx = self.x[1] - self.x[0]

    def get_initial_condition(self):
        """Generates IC based on config file settings."""
        ic_type = self.ic_config.get('type', 'gaussian')
        amp = self.ic_config.get('amplitude', 1.0)
        
        if ic_type == "gaussian":
            sigma = self.ic_config.get('width', 1.0)
            center = self.L / 2 + self.ic_config.get('center_offset', 0.0)
            return amp * np.exp(-(self.x - center)**2 / (2 * sigma**2))
        
        elif ic_type == "uniform":
            return np.full_like(self.x, amp)

        elif ic_type == "point_source":
            # Find the index closest to the desired center
            center = self.L / 2 + self.ic_config.get('center_offset', 0.0)
            idx = np.abs(self.x - center).argmin()
            
            n0 = np.zeros_like(self.x)
            n0[idx] = amp / self.dx  # Scaled by dx so the integral equals amplitude
            return n0
            
        else:
            raise ValueError(f"Unknown initial condition type: {ic_type}")

    def compute_rhs(self, t, N):
        """
        Calculates dN/dt = D * Nabla^2(N) + rho * N.
        This method is required by the BaseModel interface.
        """
        d2N = np.zeros_like(N)
        # Finite difference: Second derivative with Dirichlet BCs (N=0)
        d2N[1:-1] = (N[2:] - 2*N[1:-1] + N[:-2]) / self.dx**2
        
        return self.D * d2N + self.rho * N