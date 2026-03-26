# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:21:12 2026

@author: jacpe396
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
from .base_model import BaseModel
from typing import Dict, Any

class SpectralNDE3D(BaseModel):
    r"""
    3D neutron diffusion / reaction-diffusion equation using spectral (FFT) methods.

    PDE:

    .. math::
        \partial_t N(\mathbf{x}, t) = D \nabla^2 N(\mathbf{x}, t) + \rho\, N(\mathbf{x}, t)

    Here, ``N`` is the neutron density field, ``D`` is the diffusion coefficient
    (CGS: cm^2/s), and ``rho`` is the reactivity (CGS: 1/s).

    Spectral Laplacian:
    Using the Fourier-space identity

    .. math::
        \mathcal{F}[\nabla^2 N](\mathbf{k}, t) = -|\mathbf{k}|^2 \widehat{N}(\mathbf{k}, t)

    Therefore the Laplacian term becomes a multiplication by ``-|k|^2`` in
    Fourier space.

    Numerics summary:
    The solver integrates an ODE system in a flattened state vector
    ``N_flat(t)``. Internally, we reshape to ``N[x,y,z]`` to apply FFTs, then
    flatten the RHS back to ``N_flat``.

    Attributes:
        D: Diffusion coefficient (cm^2/s).
        rho: Reactivity (1/s).
        k_sq: Pre-computed squared wave numbers ``|k|^2`` on the FFT grid.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the spectral model with configuration parameters.

        Args:
            config: Dictionary containing 'physics' and 'initial_condition' keys.
        """
        p = config['physics']
        self.D, self.rho = p['D'], p['rho']
        # Domain size and grid resolution.
        self.L = np.array(p['L'], dtype=float)
        self.nodes = np.array(p['nodes'], dtype=int)
        
        # Grid spacing and cell volume element ``dv``.
        #
        # Note (implementation detail):
        # We currently use dx = L / (nodes - 1) and construct coordinates with
        # linspace(0, L, nodes), which includes both endpoints. FFT-based spectral
        # differentiation is naturally periodic; for strict periodic consistency one
        # would typically use dx = L / nodes (avoiding duplication of x=0 and x=L).
        self.dx_vec = self.L / (self.nodes - 1)
        self.dv = float(np.prod(self.dx_vec))
        
        # Fourier wave numbers (k-vectors).
        # ``fftfreq(n, d)`` returns frequencies for a grid spacing ``d``.
        kx = 2 * np.pi * fftfreq(self.nodes[0], d=self.dx_vec[0])
        ky = 2 * np.pi * fftfreq(self.nodes[1], d=self.dx_vec[1])
        kz = 2 * np.pi * fftfreq(self.nodes[2], d=self.dx_vec[2])
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        # Laplacian operator factor in k-space: |k|^2 (we add the minus sign later).
        self.k_sq = KX**2 + KY**2 + KZ**2
        
        # Coordinate grid (used to build initial conditions and for plotting).
        x = np.linspace(0, self.L[0], self.nodes[0])
        y = np.linspace(0, self.L[1], self.nodes[1])
        z = np.linspace(0, self.L[2], self.nodes[2])
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        self.ic_config = config['initial_condition']

    def get_initial_condition(self):
        r"""
        Build the initial neutron density ``N(\mathbf{x}, 0)`` on the model grid.

        Supported options (from ``config['initial_condition']``) are:

        * ``type: "gaussian"``:

          .. math::
              N(\mathbf{x}, 0) = A \exp\left(-\frac{|\mathbf{x}-\mathbf{x}_c|^2}{2\sigma^2}\right)

          where ``A`` is ``amplitude`` and ``sigma`` is ``width``.

        * ``type: "point_source"``:
          we approximate ``A \delta(\mathbf{x}-\mathbf{x}_c)`` by placing the value
          ``A / dv`` into the nearest grid cell, so that
          :math:`\int N\, dV \approx \sum_i N_i\, dv = A`.

        * ``center_offset`` (optional):
          shift the source center relative to the domain center. If it is a scalar,
          it is applied equally to x/y/z; if it is a length-3 array, it is applied
          component-wise.
        """
        amp = self.ic_config.get('amplitude', 1.0)
        ic_type = self.ic_config.get('type', 'gaussian')

        # Source center is the domain center plus an optional offset.
        center_offset = self.ic_config.get('center_offset', 0.0)
        if np.isscalar(center_offset):
            offset_vec = np.array([float(center_offset)] * 3, dtype=float)
        else:
            offset_vec = np.array(center_offset, dtype=float)
            if offset_vec.size != 3:
                raise ValueError("center_offset must be a scalar or length-3 array.")
        center = self.L / 2 + offset_vec

        if ic_type == "point_source":
            n0 = np.zeros(self.nodes)
            # Choose the grid cell whose coordinate is closest to ``center``.
            mid = np.clip(
                np.round(center / self.dx_vec).astype(int),
                0,
                self.nodes - 1,
            )
            n0[mid[0], mid[1], mid[2]] = amp / self.dv  # Discrete delta normalization.
            return n0.flatten()
            
        elif ic_type == "gaussian":
            sigma = self.ic_config.get('width', 1.0)
            cx, cy, cz = center  # Source center.
            r_sq = (self.X - cx)**2 + (self.Y - cy)**2 + (self.Z - cz)**2
            return (amp * np.exp(-r_sq / (2 * sigma**2))).flatten()
        
        return np.zeros(self.nodes).flatten()

    def compute_rhs(self, t: float, N_flat: np.ndarray) -> np.ndarray:
        r"""
        Compute the RHS ``dN/dt`` for the discretized PDE at time ``t``.

        PDE (linear reaction-diffusion / neutron diffusion):

        .. math::
            \partial_t N(\mathbf{x}, t) = D\nabla^2 N(\mathbf{x}, t) + \rho\, N(\mathbf{x}, t).

        Operator mapping to Fourier space:
        Using the spectral identity

        .. math::
            \mathcal{F}[\nabla^2 N] = -|\mathbf{k}|^2 \widehat{N},

        we obtain

        .. math::
            \widehat{\nabla^2 N}(\mathbf{k}, t) = -k^2 \widehat{N}(\mathbf{k}, t).

        The code implements:
        ``N_hat = FFT[N]``; the Laplacian is ``laplacian = IFFT[-k_sq * N_hat]`` with
        ``k_sq = |k|^2``; and finally
        ``dN_dt = D * laplacian + rho * N``.

        Notes:
        For a consistent model interface, ``t`` is accepted as an argument.
        For this constant-coefficient linear model, the RHS does not depend explicitly on
        ``t``.

        Args:
            t: Current simulation time (s).
            N_flat: Flattened 3D array of neutron densities.

        Returns:
            np.ndarray: Flattened array of dN/dt values.
        """
        # Reshape to 3D so we can apply FFTs along each axis.
        N = N_flat.reshape(self.nodes)
        
        # Spectral Laplacian:
        #   laplacian(x) = FFT^{-1}[ -|k|^2 FFT[N(x)] ].
        # Then include the prefactor ``D`` and the reaction term ``rho * N``.
        N_hat = fftn(N)
        # Round-off can introduce small imaginary components; the physical
        # Laplacian of a real field is real, so we take the real part.
        laplacian = np.real(ifftn(-self.k_sq * N_hat))
        
        dN_dt = self.D * laplacian + self.rho * N
        return dN_dt.flatten()