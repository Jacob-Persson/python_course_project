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
        self.D = p['D']
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

        # Parse reactivity — scalar or spatial profile.
        rho_config = p['rho']
        if isinstance(rho_config, dict):
            self.rho_field = self._build_rho_field(rho_config)
            self.rho = float(self.rho_field.flat[0])
        else:
            self.rho = float(rho_config)
            self.rho_field = self.rho * np.ones(self.nodes, dtype=float)

    def _build_rho_field(self, cfg: dict) -> np.ndarray:
        r"""Build a 3-D reactivity field from a profile configuration.

        Accepted ``cfg['type']`` values:

        * ``uniform`` — constant ``value`` everywhere.
        * ``step`` — ``inside`` / ``outside`` values separated by a planar
          interface of total ``width`` (cm).
        * ``gaussian`` — ``peak`` at center, ``background`` at infinity,
          ``width`` :math:`\sigma` in cm.
        * ``sinusoidal`` — ``mean`` value modulated by ``amplitude`` with
          ``freq`` waves across the domain in each dimension.
        * ``radial`` — ``inner`` / ``outer`` values joined by a sigmoid
          at ``radius`` (cm).
        """
        rho_type = cfg.get('type', 'uniform')
        builders = {
            'uniform': self._build_rho_uniform,
            'step': self._build_rho_step,
            'gaussian': self._build_rho_gaussian,
            'sinusoidal': self._build_rho_sinusoidal,
            'radial': self._build_rho_radial,
        }
        builder = builders.get(rho_type)
        if builder is None:
            raise ValueError(
                f"Unknown rho profile type '{rho_type}'. "
                f"Supported: {list(builders)}"
            )
        return builder(cfg)

    def _build_rho_uniform(self, cfg: dict) -> np.ndarray:
        value = float(cfg.get('value', 0.0))
        return np.full(self.nodes, value, dtype=float)

    def _build_rho_step(self, cfg: dict) -> np.ndarray:
        inside = float(cfg.get('inside', 0.0))
        outside = float(cfg.get('outside', 0.0))
        width = float(cfg.get('width', 0.5))
        axis = cfg.get('axis', 'x')
        axes = {'x': 0, 'y': 1, 'z': 2}
        ax = axes.get(axis, 0)
        L_ax = self.L[ax]
        n_ax = self.nodes[ax]
        dx = self.dx_vec[ax]
        half_width = 0.5 * width
        center = 0.5 * L_ax
        rho = np.full(self.nodes, outside, dtype=float)
        lo = int(np.clip((center - half_width) / dx, 0, n_ax - 1))
        hi = int(np.clip((center + half_width) / dx + 1, 0, n_ax))
        if ax == 0:
            rho[lo:hi, :, :] = inside
        elif ax == 1:
            rho[:, lo:hi, :] = inside
        else:
            rho[:, :, lo:hi] = inside
        return rho

    def _build_rho_gaussian(self, cfg: dict) -> np.ndarray:
        peak = float(cfg.get('peak', 0.0))
        bg = float(cfg.get('background', 0.0))
        sigma = float(cfg.get('width', 0.2))
        cx, cy, cz = 0.5 * self.L
        r_sq = (self.X - cx)**2 + (self.Y - cy)**2 + (self.Z - cz)**2
        envelope = np.exp(-r_sq / (2 * sigma**2))
        return bg + (peak - bg) * envelope

    def _build_rho_sinusoidal(self, cfg: dict) -> np.ndarray:
        mean = float(cfg.get('mean', 0.0))
        amp = float(cfg.get('amplitude', 0.0))
        freq = cfg.get('freq', [1, 1, 1])
        if np.isscalar(freq):
            freq = [float(freq)] * 3
        fx, fy, fz = float(freq[0]), float(freq[1]), float(freq[2])
        Lx, Ly, Lz = self.L
        phase = (2 * np.pi * fx * self.X / Lx
                 + 2 * np.pi * fy * self.Y / Ly
                 + 2 * np.pi * fz * self.Z / Lz)
        return mean + amp * np.cos(phase)

    def _build_rho_radial(self, cfg: dict) -> np.ndarray:
        inner = float(cfg.get('inner', 0.0))
        outer = float(cfg.get('outer', 0.0))
        r0 = float(cfg.get('radius', 3.0))
        sharpness = float(cfg.get('sharpness', 10.0))
        center_offset = cfg.get('center', [0.0, 0.0, 0.0])
        if np.isscalar(center_offset):
            center_offset = [float(center_offset)] * 3
        cx = 0.5 * self.L[0] + float(center_offset[0])
        cy = 0.5 * self.L[1] + float(center_offset[1])
        cz = 0.5 * self.L[2] + float(center_offset[2])
        r = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2 + (self.Z - cz)**2)
            # Guard against division by zero and clamp the tanh
            # argument to avoid overflow with very small r0.
        r0 = max(r0, 1e-16)
        return outer + (inner - outer) * 0.5 * (1 - np.tanh(sharpness * (r - r0) / r0))

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
        dN_dt = self.D * laplacian + self.rho_field * N + self.sigma * N**2
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
        n_state_block: Number of degrees of freedom for a single field
            (``= prod(nodes)``); the full state length is ``2 * n_state_block``.
        k0_sq: Squared characteristic wavenumber used in the dissipative
            closure, :math:`k_0^2 = (\pi / \Delta x_{\rm avg})^2`.
    """

    def __init__(self, config):
        super().__init__(config)
        self.n_state_block = int(np.prod(self.nodes))
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
        negative values; a restoring term (linear for small negativity,
        capped at 10) rapidly pushes them back to zero.

        Args:
            t: Current simulation time (s).
            y_flat: Flattened state vector of length ``2 * prod(nodes)``.

        Returns:
            Flattened RHS vector of the same length.
        """
        n = y_flat[:self.n_state_block].reshape(self.nodes)
        var_raw = y_flat[self.n_state_block:].reshape(self.nodes)
        var = np.maximum(var_raw, 0.0)

        tic('rhs_mc.fft')
        n_hat = fftn(n)
        var_hat = fftn(var)
        lap_n = np.real(ifftn(-self.k_sq * n_hat))
        lap_var = np.real(ifftn(-self.k_sq * var_hat))
        toc('rhs_mc.fft')

        tic('rhs_mc.local')
        dn_dt = (self.D * lap_n
                 + self.rho_field * n
                 + self.sigma * n ** 2
                 + self.sigma * var)

        dvar_dt = (self.D * lap_var
                   + 2.0 * self.rho_field * var
                   + 4.0 * self.sigma * n * var
                   - 2.0 * self.D * self.k0_sq * var
                   + 2.0 * self.T1 * n / self.dv)

        # Restoring force for unphysical negative variance.
        # The ODE solver can overshoot below zero at cells where both N and
        # ⟨n²⟩ are near zero.  Linear correction for small negativity;
        # capped at 10 to avoid stiff terms when overshoot is large.
        negativity = np.minimum(var_raw, 0.0)
        correction = np.clip(-negativity * 1e4, 0.0, 10.0)
        dvar_dt += correction
        toc('rhs_mc.local')

        return np.concatenate([dn_dt.flatten(), dvar_dt.flatten()])


class ThirdOrderMomentClosureNDE3D(SpectralNDE3D):
    r"""
    Third-order moment closure for the stochastic neutron diffusion equation.

    Extends the Gaussian closure (:class:`MomentClosureNDE3D`) by promoting
    the third cumulant :math:`\langle n^3 \rangle` to a dynamical variable
    instead of discarding it.  The state vector contains three fields:
    the ensemble-mean density :math:`\langle N \rangle`, the variance
    :math:`\langle n^2 \rangle`, and the third cumulant
    :math:`\langle n^3 \rangle`, where :math:`n = N - \langle N \rangle`.

    The evolution equations are derived from the SPDE

    .. math::

        \partial_t N = D\nabla^2 N + \rho N + \sigma N^2
                      + \eta(\mathbf{x},t)

        \langle\eta(\mathbf{x},t)\,\eta(\mathbf{x}',t')\rangle
        = 2 T_1\, N(\mathbf{x},t)\,\delta(\mathbf{x}-\mathbf{x}')
          \delta(t-t'),

    keeping :math:`\langle n^3 \rangle` as a dynamical variable and closing
    at the fourth-cumulant level (:math:`\langle n^4 \rangle_c = 0`).
    The dissipative sub-grid terms are modelled by

    .. math::

        \langle |\nabla n|^2 \rangle = k_0^2\,\langle n^2 \rangle,
        \qquad
        \langle n|\nabla n|^2 \rangle = k_1^2\,\langle n^3 \rangle,
        \qquad k_0 = k_1 = \frac{\pi}{\Delta x_{\rm avg}}.

    The resulting deterministic system for the three moments is

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
            + 2\sigma \langle n^3 \rangle
            - 2 D k_0^2 \langle n^2 \rangle
            + \frac{2 T_1 \langle N \rangle}{dv},

        \partial_t \langle n^3 \rangle
          = D\nabla^2 \langle n^3 \rangle
            + 3\rho \langle n^3 \rangle
            + 6\sigma \langle N \rangle \langle n^3 \rangle
            + 6\sigma \langle n^2 \rangle^2
            - 6 D k_1^2 \langle n^3 \rangle
            + \frac{6 T_1 \langle n^2 \rangle}{dv}.

    The system is deterministic and integrated with ``run_pde_solver``.
    The state vector is three times the single-field size:

        ``y = [⟨N⟩ flattened; ⟨n²⟩ flattened; ⟨n³⟩ flattened]``.

    When :math:`\langle n^3 \rangle = 0` the first two equations reduce
    exactly to the Gaussian closure (:class:`MomentClosureNDE3D`).

    Attributes:
        n_state_block: Number of degrees of freedom for a single field
            (``= prod(nodes)``); the full state length is
            ``3 * n_state_block``.
        k0_sq: Squared characteristic wavenumber for the variance
            dissipation, :math:`k_0^2 = (\pi / \Delta x_{\rm avg})^2`.
        k1_sq: Squared characteristic wavenumber for the third-moment
            dissipation, :math:`k_1^2 = (\pi / \Delta x_{\rm avg})^2`.
    """

    def __init__(self, config):
        super().__init__(config)
        self.n_state_block = int(np.prod(self.nodes))
        dx_avg = float(np.mean(self.dx_vec))
        self.k0_sq = (np.pi / dx_avg) ** 2
        self.k1_sq = (np.pi / dx_avg) ** 2

    def get_initial_condition(self):
        r"""
        Stack the initial mean (identical to the deterministic initial
        condition), the initial variance (identically zero), and the
        initial third cumulant (identically zero).

        Returns:
            Flattened array ``[⟨N⟩_flat; ⟨n²⟩_flat; ⟨n³⟩_flat]`` of
            length ``3 * prod(nodes)``.
        """
        n0 = super().get_initial_condition()
        var0 = np.zeros_like(n0)
        third0 = np.zeros_like(n0)
        return np.concatenate([n0, var0, third0])

    def compute_rhs(self, t: float, y_flat: np.ndarray) -> np.ndarray:
        r"""
        Right-hand side for the three-moment system.

        The three blocks of ``y_flat`` are, in order,
        :math:`\langle N \rangle`, :math:`\langle n^2 \rangle`, and
        :math:`\langle n^3 \rangle`.

        A restoring term (linear for small negativity, capped at 10)
        is added to the variance RHS to correct solver-induced negative
        values.  The third cumulant is not restored — it may be positive
        or negative by physics.

        Args:
            t: Current simulation time (s).
            y_flat: Flattened state vector of length
                ``3 * prod(nodes)``.

        Returns:
            Flattened RHS vector of the same length.
        """
        nb = self.n_state_block
        n = y_flat[:nb].reshape(self.nodes)
        var_raw = y_flat[nb:2*nb].reshape(self.nodes)
        third_raw = y_flat[2*nb:].reshape(self.nodes)

        var = np.maximum(var_raw, 0.0)

        tic('rhs_mc3.fft')
        n_hat = fftn(n)
        var_hat = fftn(var)
        third_hat = fftn(third_raw)
        lap_n = np.real(ifftn(-self.k_sq * n_hat))
        lap_var = np.real(ifftn(-self.k_sq * var_hat))
        lap_third = np.real(ifftn(-self.k_sq * third_hat))
        toc('rhs_mc3.fft')

        tic('rhs_mc3.local')
        dn_dt = (self.D * lap_n
                 + self.rho_field * n
                 + self.sigma * n ** 2
                 + self.sigma * var)

        dvar_dt = (self.D * lap_var
                   + 2.0 * self.rho_field * var
                   + 4.0 * self.sigma * n * var
                   + 2.0 * self.sigma * third_raw
                   - 2.0 * self.D * self.k0_sq * var
                   + 2.0 * self.T1 * n / self.dv)

        dthird_dt = (self.D * lap_third
                     + 3.0 * self.rho_field * third_raw
                     + 6.0 * self.sigma * n * third_raw
                     + 6.0 * self.sigma * var ** 2
                     - 6.0 * self.D * self.k1_sq * third_raw
                     + 6.0 * self.T1 * var / self.dv)

        # Restoring force for unphysical negative variance.
        negativity = np.minimum(var_raw, 0.0)
        correction = np.clip(-negativity * 1e4, 0.0, 10.0)
        dvar_dt += correction
        toc('rhs_mc3.local')

        return np.concatenate([dn_dt.flatten(),
                               dvar_dt.flatten(),
                               dthird_dt.flatten()])


class DelayedNeutronSpectralNDE3D(SpectralNDE3D):
    r"""
    3D neutron diffusion with delayed neutron precursors.

    Coupled PDE system (Ito interpretation):

    .. math::

        \partial_t N &= D\nabla^2 N + (\rho - \beta)N + \sigma N^2
                       + \lambda_D M + \eta_1, \\[4pt]
        \partial_t M &= \beta N - \lambda_D M + \eta_2,

    where :math:`N` is the neutron density, :math:`M` the precursor density,
    :math:`\beta` the delayed fraction, :math:`\lambda_D` the precursor decay
    constant, and :math:`\eta_1`, :math:`\eta_2` are independent Gaussian
    noises with

    .. math::

        \langle\eta_i(\mathbf{x},t)\,\eta_i(\mathbf{x}',t')\rangle
        = 2 T_i\, N_i(\mathbf{x},t)\,\delta(\mathbf{x}-\mathbf{x}')
          \,\delta(t-t').

    The state vector is :math:`[N_{\text{flat}};\, M_{\text{flat}}]`, i.e.
    twice the length of the single-variable model.
    """

    def __init__(self, config):
        super().__init__(config)
        p = config['physics']
        self.beta = float(p.get('beta', 0.0))
        self.lambda_D = float(p.get('lambda_D', 0.0))
        self.T2 = float(p.get('T2', 0.0))

    @property
    def is_delayed(self):
        return True

    def get_initial_condition(self):
        N0 = super().get_initial_condition()
        ic = self.ic_config
        M0_amp = float(ic.get('M_amplitude', 0.0))
        M0 = np.full_like(N0, M0_amp)
        return np.concatenate([N0.flatten(), M0.flatten()])

    def compute_rhs(self, t, y_flat):
        nc = int(np.prod(self.nodes))
        N = y_flat[:nc].reshape(self.nodes)
        M = y_flat[nc:].reshape(self.nodes)

        tic('rhs_delayed.fft')
        N_hat = fftn(N)
        laplacian = np.real(ifftn(-self.k_sq * N_hat))
        toc('rhs_delayed.fft')

        tic('rhs_delayed.local')
        dN_dt = (self.D * laplacian
                 + (self.rho_field - self.beta) * N
                 + self.sigma * N**2
                 + self.lambda_D * M)
        dM_dt = self.beta * N - self.lambda_D * M
        toc('rhs_delayed.local')

        return np.concatenate([dN_dt.flatten(), dM_dt.flatten()])

    def compute_precursor_noise_amplitude(self, M):
        if self.T2 == 0.0:
            return np.zeros_like(M)
        return np.sqrt(2.0 * self.T2 * np.maximum(M, 0.0) / self.dv)