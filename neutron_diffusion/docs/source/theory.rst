Theory of Neutron Diffusion
===========================

The core of this project is the Neutron Diffusion Equation (NDE) in 3D with
optional quadratic feedback:

.. math::

   \frac{\partial N}{\partial t} = D \nabla^2 N + \rho N + \sigma N^2

Where:
* :math:`D` is the diffusion coefficient.
* :math:`\rho` is the reactivity (linear growth/decay).
* :math:`\sigma` is the quadratic feedback coefficient (set ``sigma: 0`` for the linear model).

Noise (stochastic forcing)
--------------------------
Fluctuations from the discrete nature of the fission chain are modelled by a
Gaussian multiplicative noise term:

.. math::

   \partial_t N = D \nabla^2 N + \rho N + \sigma N^2
                 + \eta(\mathbf{x},t),

   \langle \eta(\mathbf{x},t)\,\eta(\mathbf{x}',t') \rangle
   = 2 T_1\, N(\mathbf{x},t)\,\delta(\mathbf{x}-\mathbf{x}')\,
     \delta(t-t').

The noise is:

* **Multiplicative**: the amplitude scales as :math:`\sqrt{N}` because the
  variance of a Poisson birth-death process is proportional to the mean
  population.
* **White in space**: fluctuations at different grid cells are independent
  (the microscopic fission events are uncorrelated).
* **White in time**: the noise has no memory on the macroscopic timescale.

The parameter :math:`T_1` (set ``T1`` in the config) controls the fluctuation
strength.  Setting ``T1: 0`` recovers the deterministic NDE.

Gaussian moment closure
-----------------------
To obtain the ensemble-averaged density :math:`\langle N \rangle` without
running many individual noise realisations, we evolve the first two moments of
the field.  Write :math:`N = \langle N \rangle + n` with :math:`\langle n
\rangle = 0`.  Averaging the SPDE gives the exact equation

.. math::

   \partial_t \langle N \rangle
   = D \nabla^2 \langle N \rangle
     + \rho \langle N \rangle
     + \sigma \bigl( \langle N \rangle^2 + \langle n^2 \rangle \bigr),

so the mean is driven by both the deterministic dynamics and the variance
:math:`\langle n^2 \rangle` (the noise-induced shift).

An equation for the variance is obtained from the Ito formula for
:math:`\langle n^2 \rangle`.  Discarding the third cumulant (Gaussian
closure, :math:`\langle n^3 \rangle = 0`) and modelling the sub-grid
dissipation as :math:`\langle |\nabla n|^2 \rangle = k_0^2 \langle n^2
\rangle` with :math:`k_0 = \pi / \Delta x_{\rm avg}` yields:

.. math::

   \partial_t \langle n^2 \rangle
   = D \nabla^2 \langle n^2 \rangle
     + 2\rho \langle n^2 \rangle
     + 4\sigma \langle N \rangle \langle n^2 \rangle
     - 2 D k_0^2 \langle n^2 \rangle
     + \frac{2 T_1 \langle N \rangle}{dv}.

The term :math:`2 T_1 \langle N \rangle / dv` is the noise injection from the
covariance above, and :math:`-2 D k_0^2 \langle n^2 \rangle` represents the
smoothing of small-scale fluctuations by diffusion.

The two equations form a deterministic PDE system that is integrated with the
same spectral ODE solver used for the deterministic NDE.  The state vector
is the concatenation :math:`[\langle N \rangle; \langle n^2 \rangle]`.

A restoring term :math:`-\min(\langle n^2 \rangle_{\rm raw}, 0) \times 10^4`
is added to the variance RHS to rapidly push any solver-induced negative
variance back to zero — the adaptive Runge–Kutta integrator can otherwise
overshoot below zero at cells where both :math:`\langle N \rangle` and
:math:`\langle n^2 \rangle` are near zero.

Numerical Implementation
------------------------
To achieve high precision, we use a **Spectral Method**. By taking the
Fourier Transform of both sides, the Laplacian becomes a simple multiplication
by :math:`-k^2`:

.. math::

   \mathcal{F}\{\nabla^2 N\} = -|k|^2 \hat{N}
