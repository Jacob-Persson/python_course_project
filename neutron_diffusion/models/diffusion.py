# -*- coding: utf-8 -*-
import numpy as np
from numpy.fft import fftfreq
from .base_model import BaseModel
from typing import Dict, Any
from utils.timer import tic, toc

# Use pyfftw if available (2–4× faster FFTs); fall back to numpy.fft.
try:
    from pyfftw.interfaces.numpy_fft import fftn, ifftn
    from pyfftw.interfaces import cache as _cache
    _cache.enable()
except ImportError:
    from numpy.fft import fftn, ifftn

class SpectralNDE3D(BaseModel):
    r"""
    3D neutron diffusion / reaction-diffusion equation using spectral (FFT) methods.

    PDE:

    .. math::
        \partial_t N(\mathbf{x}, t) = D \nabla^2 N(\mathbf{x}, t)
        + \rho\, N(\mathbf{x}, t) + \sigma\, N(\mathbf{x}, t)^2

    Here, ``N`` is the neutron density field, ``D`` is the diffusion coefficient
    (CGS: cm^2/s), ``rho`` is the reactivity (CGS: 1/s), and ``sigma`` is the
    quadratic feedback coefficient (CGS: 1/(cm^3 s)).

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
        sigma: Quadratic feedback coefficient (1/(cm^3 s)); defaults to 0.
        T1: Noise amplitude (cm^3/s); 0 = deterministic.
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
        self.sigma = float(p.get('sigma', 0.0))
        self.T1 = float(p.get('T1', 0.0))
        # Domain size and grid resolution.
        self.L = np.array(p['L'], dtype=float)
        self.nodes = np.array(p['nodes'], dtype=int)
        
        # Grid spacing and cell volume element ``dv``.
        #
        # FFTs assume a periodic grid where x=0 and x=L are the same point.
        # To be strictly consistent with that picture we use a periodic grid:
        #   x_j = j * dx,  j=0..nodes-1,  with dx = L / nodes  (endpoint=False).
        self.dx_vec = self.L / self.nodes
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
        # Strictly periodic coordinate grid: exclude endpoint x=L to avoid duplicating x=0.
        x = np.linspace(0, self.L[0], self.nodes[0], endpoint=False)
        y = np.linspace(0, self.L[1], self.nodes[1], endpoint=False)
        z = np.linspace(0, self.L[2], self.nodes[2], endpoint=False)
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

        * ``type: "uniform"``:
          constant field :math:`N(\mathbf{x}, 0) = A` everywhere, where ``A`` is
          ``amplitude``.

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

        elif ic_type == "uniform":
            return np.full(np.prod(self.nodes), amp, dtype=float)
        
        return np.zeros(self.nodes).flatten()

    def compute_rhs(self, t: float, N_flat: np.ndarray) -> np.ndarray:
        r"""
        Compute the RHS ``dN/dt`` for the discretized PDE at time ``t``.

        PDE (reaction-diffusion with optional quadratic feedback):

        .. math::
            \partial_t N = D\nabla^2 N + \rho N + \sigma N^2.

        The Laplacian is evaluated spectrally; the local terms ``\rho N`` and
        ``\sigma N^2`` are applied in physical space after reshaping.

        Operator mapping to Fourier space (diffusion term only):
        Using the spectral identity

        .. math::
            \mathcal{F}[\nabla^2 N] = -|\mathbf{k}|^2 \widehat{N},

        we obtain

        .. math::
            \widehat{\nabla^2 N}(\mathbf{k}, t) = -k^2 \widehat{N}(\mathbf{k}, t).

        The code implements:
        ``N_hat = FFT[N]``; ``laplacian = IFFT[-k_sq * N_hat]`` with ``k_sq = |k|^2``;
        and finally ``dN_dt = D * laplacian + rho * N + sigma * N**2``.

        Notes:
        For a consistent model interface, ``t`` is accepted as an argument.
        For constant coefficients, the RHS does not depend explicitly on ``t``.
        When ``sigma = 0``, the model reduces to the linear NDE.

        Args:
            t: Current simulation time (s).
            N_flat: Flattened 3D array of neutron densities.

        Returns:
            np.ndarray: Flattened array of dN/dt values.
        """
        tic('rhs.fft')
        N = N_flat.reshape(self.nodes)
        N_hat = fftn(N)
        laplacian = np.real(ifftn(-self.k_sq * N_hat))
        toc('rhs.fft')

        tic('rhs.local')
        dN_dt = self.D * laplacian + self.rho * N + self.sigma * N**2
        toc('rhs.local')
        return dN_dt.flatten()

    def compute_noise_amplitude(self, N: np.ndarray) -> np.ndarray:
        r"""
        Multiplicative noise amplitude consistent with the Ito-interpretation SPDE

        .. math::
            \partial_t N = D\nabla^2 N + \rho N + \sigma N^2 + \eta,
            
            
            \langle\eta(\mathbf{x},t)\,\eta(\mathbf{x}',t')\rangle
            = 2 T_1\, N(\mathbf{x},t)\,\delta(\mathbf{x}-\mathbf{x}')
              \,\delta(t-t').

        After spatial discretisation the cell-volume element ``dv`` replaces
        :math:`\delta(\mathbf{x}-\mathbf{x}')`, giving the per-cell amplitude

        .. math::
            g_i(N) = \sqrt{\frac{2 T_1 N_i}{dv}}.

    Values are clipped to ``N_i \ge 0`` (via ``np.maximum``) so that the
    square root is always real.

        Args:
            N: 3-D array of the neutron density field on the full grid.

        Returns:
            3-D array of noise amplitudes (same shape as ``N``).
        """
        if self.T1 == 0.0:
            return np.zeros_like(N)
        return np.sqrt(2.0 * self.T1 * np.maximum(N, 0.0) / self.dv)


class MomentClosureNDE3D(SpectralNDE3D):
    r"""
    Gaussian moment closure for the stochastic neutron diffusion equation.

    Inherits the spectral (FFT) infrastructure from :class:`SpectralNDE3D`.

    Instead of drawing individual noise realisations we evolve the first two
    moments of the density field directly.  The state vector contains the
    ensemble-mean field :math:`\langle N \rangle` and the variance
    :math:`\langle n^2 \rangle` (where :math:`n = N - \langle N \rangle`).
    The evolution equations are derived from the SPDE

    .. math::

        \partial_t N = D\nabla^2 N + \rho N + \sigma N^2
                      + \eta(\mathbf{x},t)

        \langle\eta(\mathbf{x},t)\,\eta(\mathbf{x}',t')\rangle
        = 2 T_1\, N(\mathbf{x},t)\,\delta(\mathbf{x}-\mathbf{x}')
          \delta(t-t'),

    together with a Gaussian (zero-third-cumulant) closure :math:`\langle
    n^3 \rangle = 0` and a gradient-diffusion model for the dissipative term

    .. math::

        \langle |\nabla n|^2 \rangle = k_0^2\,\langle n^2 \rangle,
        \qquad k_0 = \frac{\pi}{\Delta x_{\rm avg}},

    which accounts for the damping of sub-grid fluctuations by diffusion.

    The resulting deterministic system for the two moments is

    .. math::

        \partial_t \langle N \rangle
          = D\nabla^2 \langle N \rangle
            + \rho \langle N \rangle
            + \sigma \langle N \rangle^2
            + \sigma \langle n^2 \rangle,

        \partial_t \langle n^2 \rangle
          = D\nabla^2 \langle n^2 \rangle
            + 2\rho \langle n^2 \rangle
            + 4\sigma \langle N \rangle \langle n^2 \rangle
            - 2 D k_0^2 \langle n^2 \rangle
            + \frac{2 T_1 \langle N \rangle}{dv}.

    Because the right-hand side is purely deterministic, the system is
    integrated with ``run_pde_solver`` (the same SciPy-based ODE solver used
    for the deterministic NDE).  A restoring term
    :math:`-\min(\langle n^2 \rangle_{\rm raw}, 0) \times 10^4` is added to
    the variance RHS to correct solver-induced negative values (see
    ``compute_rhs``).  The state vector is twice as large:

        ``y = [⟨N⟩(x,y,z) flattened;  ⟨n²⟩(x,y,z) flattened]``.

    When :math:`T_1 = 0` the variance equation has no source so
    :math:`\langle n^2 \rangle \equiv 0`, and the mean reduces to the
    deterministic nonlinear NDE.

    Attributes:
        n_state_half: Number of degrees of freedom for a single field
            (``= prod(nodes)``); the full state length is ``2 * n_state_half``.
        k0_sq: Squared characteristic wavenumber used in the dissipative
            closure, :math:`k_0^2 = (\pi / \Delta x_{\rm avg})^2`.
    """

    def __init__(self, config):
        super().__init__(config)
        self.n_state_half = int(np.prod(self.nodes))
        # Characteristic wavenumber for the dissipative term.
        dx_avg = float(np.mean(self.dx_vec))
        self.k0_sq = (np.pi / dx_avg) ** 2

    def get_initial_condition(self):
        r"""
        Stack the initial mean (identical to the deterministic initial
        condition) and the initial variance (identically zero).

        Returns:
            Flattened array ``[⟨N⟩_flat;  ⟨n²⟩_flat]`` of length
            ``2 * prod(nodes)``.
        """
        n0 = super().get_initial_condition()
        var0 = np.zeros_like(n0)
        return np.concatenate([n0, var0])

    def compute_rhs(self, t: float, y_flat: np.ndarray) -> np.ndarray:
        r"""
        Right-hand side for the two-moment system.

        The first half of ``y_flat`` is :math:`\langle N \rangle` and the
        second half is :math:`\langle n^2 \rangle`.

        The raw (unclipped) variance is used to detect any solver-induced
        negative values; a restoring term
        :math:`-\min(\langle n^2 \rangle_{\rm raw}, 0) \times 10^4` rapidly
        pushes them back to zero.

        Args:
            t: Current simulation time (s).
            y_flat: Flattened state vector of length ``2 * prod(nodes)``.

        Returns:
            Flattened RHS vector of the same length.
        """
        n = y_flat[:self.n_state_half].reshape(self.nodes)
        var_raw = y_flat[self.n_state_half:].reshape(self.nodes)
        var = np.maximum(var_raw, 0.0)

        tic('rhs_mc.fft')
        n_hat = fftn(n)
        var_hat = fftn(var)
        lap_n = np.real(ifftn(-self.k_sq * n_hat))
        lap_var = np.real(ifftn(-self.k_sq * var_hat))
        toc('rhs_mc.fft')

        tic('rhs_mc.local')
        dn_dt = (self.D * lap_n
                 + self.rho * n
                 + self.sigma * n ** 2
                 + self.sigma * var)

        dvar_dt = (self.D * lap_var
                   + 2.0 * self.rho * var
                   + 4.0 * self.sigma * n * var
                   - 2.0 * self.D * self.k0_sq * var
                   + 2.0 * self.T1 * n / self.dv)

        # Restoring force for unphysical negative variance.
        # The ODE solver can overshoot below zero at cells where both N and
        # ⟨n²⟩ are near zero.  This correction brings them back with a
        # sub-millisecond time constant, much faster than any physical scale.
        negativity = np.minimum(var_raw, 0.0)
        dvar_dt += -negativity * 1e4
        toc('rhs_mc.local')

        return np.concatenate([dn_dt.flatten(), dvar_dt.flatten()])