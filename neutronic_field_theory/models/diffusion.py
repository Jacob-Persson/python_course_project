# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:21:12 2026

@author: jacpe396
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
from .base_model import BaseModel

class SpectralNDE3D(BaseModel):
    def __init__(self, config):
        p = config['physics']
        self.D, self.rho = p['D'], p['rho']
        # Ensure L and nodes are floats/ints for math
        self.L = np.array(p['L'], dtype=float)
        self.nodes = np.array(p['nodes'], dtype=int)
        
        # Grid spacing and Volume Element
        self.dx_vec = self.L / (self.nodes - 1)
        self.dv = np.prod(self.dx_vec)
        
        # 1. Fourier Wave Numbers (k-vectors)
        # Note: fftfreq uses the grid spacing 'd'
        kx = 2 * np.pi * fftfreq(self.nodes[0], d=self.dx_vec[0])
        ky = 2 * np.pi * fftfreq(self.nodes[1], d=self.dx_vec[1])
        kz = 2 * np.pi * fftfreq(self.nodes[2], d=self.dx_vec[2])
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        # Laplacian operator in k-space: -|k|^2
        self.k_sq = KX**2 + KY**2 + KZ**2
        
        # 2. Coordinate Grid for ICs
        x = np.linspace(0, self.L[0], self.nodes[0])
        y = np.linspace(0, self.L[1], self.nodes[1])
        z = np.linspace(0, self.L[2], self.nodes[2])
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        self.ic_config = config['initial_condition']

    def get_initial_condition(self):
        amp = self.ic_config.get('amplitude', 1.0)
        ic_type = self.ic_config.get('type', 'gaussian')

        if ic_type == "point_source":
            n0 = np.zeros(self.nodes)
            mid = self.nodes // 2
            n0[mid[0], mid[1], mid[2]] = amp / self.dv
            return n0.flatten()
            
        elif ic_type == "gaussian":
            sigma = self.ic_config.get('width', 1.0)
            cx, cy, cz = self.L / 2
            r_sq = (self.X - cx)**2 + (self.Y - cy)**2 + (self.Z - cz)**2
            return (amp * np.exp(-r_sq / (2 * sigma**2))).flatten()
        
        return np.zeros(self.nodes).flatten()

    def compute_rhs(self, t, N_flat):
        # Reshape to 3D for FFT
        N = N_flat.reshape(self.nodes)
        
        # 3. Spectral Derivative
        # dN/dt = F^-1 [ -D * k^2 * F(N) ] + rho * N
        N_hat = fftn(N)
        laplacian = np.real(ifftn(-self.k_sq * N_hat))
        
        dN_dt = self.D * laplacian + self.rho * N
        return dN_dt.flatten()