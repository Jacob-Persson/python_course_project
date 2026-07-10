Theory of Neutron Diffusion
===========================

The core of this project is the Neutron Diffusion Equation (NDE) in 3D with
optional quadratic feedback and multiplicative noise:

.. math::

   \frac{\partial N}{\partial t} = D \nabla^2 N + \rho N + \sigma N^2

Where:
* :math:`D` is the diffusion coefficient.
* :math:`\rho` is the reactivity (linear growth/decay).  May be a scalar or a 3-D field :math:`\rho(\mathbf{x})` — see :doc:`usage`.
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

In this code :math:`T_1` is treated as a phenomenological noise amplitude.
A direct mapping to nuclear-engineering constants (e.g. the delayed-neutron
fraction :math:`\beta_{\rm eff}` or the prompt-neutron generation time
:math:`\Lambda`) would require an additional coarse-graining step that is
beyond the scope of the present Directed-Percolation study.

Delayed neutron precursors
--------------------------

When the delayed-neutron fraction :math:`\beta` is non-zero the model is
extended to a two-variable system:

.. math::

   \partial_t N &= D\nabla^2 N + (\rho - \beta) N + \sigma N^2
                  + \lambda_D M + \eta_1, \\[4pt]
   \partial_t M &= \beta N - \lambda_D M + \eta_2,

where :math:`M` is the precursor density, :math:`\beta` is the delayed
fraction (dimensionless), and :math:`\lambda_D` the precursor decay
constant.  Two independent noises drive the system:

.. math::

   \langle\eta_i(\mathbf{x},t)\,\eta_i(\mathbf{x}',t')\rangle
   = 2 T_i\, N_i(\mathbf{x},t)\,\delta(\mathbf{x}-\mathbf{x}')\,
     \delta(t-t'), \qquad \langle\eta_1\eta_2\rangle = 0.

The state vector is :math:`[N_{\text{flat}};\, M_{\text{flat}}]`,
twice the length of the single-variable model.  The standard model is
recovered for :math:`\beta = \lambda_D = 0`.  Moment-closure variants
for the extended system are not yet implemented.

MSR field theory
----------------

The Martin–Siggia–Rose (MSR) formalism maps the stochastic PDE onto a
field-theoretic action, making the scaling properties of the system
accessible to renormalisation-group (RG) analysis.

Introduce a real response field :math:`\widetilde N(\mathbf{x},t)` and use the
identity :math:`\langle \det(\partial_t - D\nabla^2 - \rho - \sigma N) \rangle
= 1` for the Jacobian of the transformation from noise :math:`\eta` to
density :math:`N`.  For our Ito-interpreted SPDE the MSR action reads

.. math::

   S[N,\widetilde N] = \int d^3x\,dt\;
   \widetilde N\Bigl(\partial_t N - D\nabla^2 N - \rho N - \sigma N^2\Bigr)
   - T_1 \int d^3x\,dt\; \widetilde N^{\,2} N.

The quadratic term in :math:`\widetilde N` comes from the Gaussian noise
covariance :math:`\langle\eta\eta\rangle = 2 T_1 N`.

Correlation and response functions are generated as usual by
:math:`\langle \mathcal O \rangle = \int \mathcal D N \mathcal D\widetilde N\;
\mathcal O \, e^{-S}`.  The bare propagator (inverse of the Gaussian part of
:math:`S`) in Fourier space is

.. math::

   G_0(k,\omega) = \frac{1}{-i\omega + D k^2 - \rho},

with the causal retarded response corresponding to forward-in-time evolution
(the Ito convention picks the retarded branch).

Correlation length and the role of :math:`\rho`
-----------------------------------------------

Near the critical point :math:`\rho = \rho_c = 0` the zero-frequency response
is

.. math::

   G_0(k,0) = \frac{1}{D k^2 - \rho}
            = \frac{1}{D}\, \frac{1}{k^2 + \kappa^2},
   \qquad \kappa^2 \equiv -\frac{\rho}{D},

so the spatial correlation of the linear theory decays as
:math:`e^{-r/\xi}` with

.. math::

   \xi = \frac{1}{\kappa} = \sqrt{\frac{D}{|\rho|}}.

This defines the **correlation-length exponent** :math:`\nu` through
:math:`\xi \sim |\rho - \rho_c|^{-\nu}`, giving :math:`\nu = 1/2` at the
Gaussian (mean-field) level.

The deterministic nonlinearity :math:`\sigma N^2` and the noise vertex
:math:`T_1 \widetilde N^2 N` generate loop corrections to the propagator.
These shift the critical exponents away from their mean-field values (see, e.g.,
the directed-percolation universality class, where :math:`\nu \approx 0.58` in
:math:`d=3`).  The Gaussian moment closure described below discards the
non-Gaussian cumulants, which is equivalent to working at tree-level in the
MSR field theory, and therefore reproduces the mean-field exponents.

Dynamic exponent
----------------

The characteristic relaxation rate of a Fourier mode with wavenumber
:math:`k` is read from the pole of the propagator:

.. math::

   \omega(k) = -i D k^2 \quad \Longrightarrow \quad
   \Gamma(k) \equiv \operatorname{Im} \omega(k) = D k^{2},
   \qquad z = 2.

At the Gaussian fixed point the dynamic exponent is therefore :math:`z = 2`,
i.e. :math:`\Gamma(k) \sim k^z`.  Non-Gaussian corrections renormalise this
to :math:`z \approx 1.90` in the 3D directed-percolation universality class,
but the moment closure keeps :math:`z = 2`.

Anomalous dimension
-------------------

The anomalous dimension :math:`\eta` measures the deviation of the
equal-time correlation function from the Ornstein–Zernicke form
:math:`C(k) \sim 1/(k^{2-\eta})`.  At the Gaussian fixed point
:math:`\eta = 0`.  A non-zero :math:`\eta` requires multi-loop
corrections beyond the Gaussian closure.

Summary of mean-field exponents
--------------------------------

.. list-table::
   :header-rows: 1

   * - Exponent
     - Definition
     - Mean-field value
   * - :math:`\nu`
     - :math:`\xi \sim |\rho - \rho_c|^{-\nu}`
     - :math:`1/2`
   * - :math:`z`
     - :math:`\Gamma(k) \sim k^{z}`
     - :math:`2`
   * - :math:`\eta`
     - :math:`C(k) \sim k^{-2+\eta}`
     - :math:`0`

These are the same exponents as the Gaussian fixed point of the
:math:`\phi^4` theory (the Ising universality class in its upper critical
dimension :math:`d=4`), because the MSR action for the linear SPDE is
equivalent to a free scalar field theory for the response field.

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

The :math:`10^4` multiplier is deliberately stiff: it activates only when the
variance has already become negative (typically at near-zero cells, not in
the bulk) and brings it back with a sub-millisecond time constant.  Because
scipy's adaptive solvers expose no hook to floor internal stages after each
sub-step, the stiff penalty is the least-bad way to keep the state
physically admissible without forgoing adaptive stepping entirely.

Third-order moment closure
---------------------------
The Gaussian closure sets :math:`\langle n^3 \rangle = 0`, discarding the
third cumulant entirely.  Going one step further we promote
:math:`\langle n^3 \rangle` to a dynamical variable and close at the
*fourth*-cumulant level, which yields the first non-Gaussian correction
to the moment hierarchy.

The exact equation for the third cumulant is obtained from the Ito formula
for :math:`n^3`:

.. math::

   \partial_t \langle n^3 \rangle
   = 3 D \langle n^2 \nabla^2 n \rangle
     + 3 \rho \langle n^3 \rangle
     + 6 \sigma \langle N \rangle \langle n^3 \rangle
     + 3 \sigma \langle n^4 \rangle
     - 3 \sigma \langle n^2 \rangle^2
     + \frac{6 T_1 \langle n^2 \rangle}{dv}.

The unresolved terms are handled with two closure assumptions:

1. **Vanishing fourth cumulant** (Gaussian on the fourth):
   :math:`\langle n^4 \rangle_c = \langle n^4 \rangle - 3 \langle n^2
   \rangle^2 = 0`.

2. **Sub-grid dissipation for the third moment**:
   :math:`\langle n |\nabla n|^2 \rangle = k_1^2 \langle n^3 \rangle`
   with :math:`k_1 = \pi / \Delta x_{\rm avg} = k_0`.  Using the identity

   .. math::

      n^2 \nabla^2 n = \tfrac13 \nabla^2 (n^3) - 2 n |\nabla n|^2,

   the gradient contribution becomes

   .. math::

      \langle n^2 \nabla^2 n \rangle
      = \tfrac13 \nabla^2 \langle n^3 \rangle
        - 2 k_1^2 \langle n^3 \rangle.

Putting everything together yields the closed three-field system

.. math::

   \partial_t \langle N \rangle
     = D \nabla^2 \langle N \rangle
       + \rho \langle N \rangle
       + \sigma \bigl( \langle N \rangle^2 + \langle n^2 \rangle \bigr),

   \partial_t \langle n^2 \rangle
     = D \nabla^2 \langle n^2 \rangle
       + 2 \rho \langle n^2 \rangle
       + 4 \sigma \langle N \rangle \langle n^2 \rangle
       + 2 \sigma \langle n^3 \rangle
       - 2 D k_0^2 \langle n^2 \rangle
       + \frac{2 T_1 \langle N \rangle}{dv},

   \partial_t \langle n^3 \rangle
     = D \nabla^2 \langle n^3 \rangle
       + 3 \rho \langle n^3 \rangle
       + 6 \sigma \langle N \rangle \langle n^3 \rangle
       + 6 \sigma \langle n^2 \rangle^2
       - 6 D k_1^2 \langle n^3 \rangle
       + \frac{6 T_1 \langle n^2 \rangle}{dv}.

The state vector is three fields long:
:math:`[\langle N \rangle; \langle n^2 \rangle; \langle n^3 \rangle]`.
The restoring term for negative :math:`\langle n^2 \rangle` is applied as
before; the third cumulant is *not* restored because it can be
physically positive or negative.

When :math:`\langle n^3 \rangle = 0` the first two equations reduce
identically to the Gaussian closure.  The key new physical effect is the
coupling :math:`6 T_1 \langle n^2 \rangle / dv` in the third-moment
equation: even with a purely linear drift (:math:`\sigma = 0`), the
multiplicative noise alone generates non-Gaussian statistics through
this source term.  This is a genuine one-loop effect in the MSR field
theory that the Gaussian closure misses.

Numerical Implementation (PDE solver)
--------------------------------------
To achieve high precision, we use a **Spectral Method**. By taking the
Fourier Transform of both sides, the Laplacian becomes a simple multiplication
by :math:`-k^2`:

.. math::

   \mathcal{F}\{\nabla^2 N\} = -|k|^2 \hat{N},
   \qquad |k|^2 = k_x^2 + k_y^2 + k_z^2.

Given a state vector :math:`N(t)` flattened over the 3D grid, each RHS
evaluation performs:

#. Reshape to :math:`N_x \times N_y \times N_z`.
#. Forward FFT: :math:`\hat N = \texttt{FFT3}(N)`.
#. Spectral Laplacian: :math:`\widehat{\nabla^2 N} = -k^2 \hat N`.
#. Inverse FFT: :math:`\nabla^2 N = \texttt{IFFT3}(-k^2 \hat N)`.
#. Local terms in physical space:
   :math:`\partial_t N = D \nabla^2 N + \rho N + \sigma N^2`.

The result is flattened and returned to the ODE integrator.

For the moment-closure systems, steps 1–4 are repeated for each field.
The Gaussian closure (:math:`2 \times N_x N_y N_z` variables) applies the
Laplacian to :math:`\langle N \rangle` and :math:`\langle n^2 \rangle`.
The third-order closure (:math:`3 \times N_x N_y N_z` variables) adds a
third FFT/IFFT pair for :math:`\langle n^3 \rangle` and assembles the
additional local terms :math:`6 \sigma \langle n^2 \rangle^2`,
:math:`6 T_1 \langle n^2 \rangle / dv`, and the :math:`-6 D k_1^2 \langle n^3 \rangle`
dissipation.

For stochastic runs the solver uses a fixed-step Ito–Euler–Maruyama
scheme: each time step consists of a deterministic RK4 sub-step followed
by a fluctuation increment
:math:`\sqrt{2 T_1 N_i / dv}\; \mathcal N(0, 1)` per cell.
Adaptive step-size control is disabled in this mode because the
:math:`\sqrt{dt}` scaling of the noise increment is incompatible with
the :math:`\mathcal O(dt)` error estimates used by adaptive Runge–Kutta
controllers.  In contrast, the moment-closure system is purely
deterministic, so **no** stochastic splitting is used; its time
integration uses the same adaptive RK45 solver
(:func:`scipy.integrate.solve_ivp`) as the deterministic NDE.

Extraction of critical exponents from the numerical solution
-------------------------------------------------------------

The :mod:`utils.critical_exponents` module extracts :math:`\nu`, :math:`z`,
and :math:`\eta` by probing the linearised response of the solver, without
running a full time integration.

Correlation-length exponent :math:`\nu`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We sweep the reactivity :math:`\rho` across :math:`10` values between
:math:`-0.5` and :math:`-0.01` (keeping :math:`\sigma = 0`, :math:`T_1 = 0`)
and measure the decay rate of a single Fourier mode:

#. Initialise the field to the pure cosine mode
   :math:`N(\mathbf{x}) = \cos(k_x x)` with the smallest non-zero
   wavenumber on the grid.
#. Evaluate the RHS in one step:
   :math:`\texttt{rhs} = D \nabla^2 N + \rho N`.
#. The decay rate of that mode is
   :math:`\Gamma = -\texttt{rhs}(0) / N(0) = D k_{\rm test}^2 - \rho`.
#. From the propagator pole at :math:`D(k^2 + \kappa^2)`, the inverse
   correlation length follows as

   .. math::

      \kappa^2 = \frac{\Gamma}{D} - k_{\rm test}^2 = -\frac{\rho}{D},
      \qquad \xi = \frac{1}{\kappa}.

#. Fit :math:`\log \xi = -\nu \log|\rho - \rho_c| + \text{const}` with
   :math:`\rho_c = 0` to obtain :math:`\nu`.

The result is :math:`\nu = 0.5000`, consistent with the mean-field prediction.

Dynamic exponent :math:`z`
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The relaxation rate :math:`\Gamma(k) = D k^2` is evaluated on the spectral
grid.  A linear least-squares fit of :math:`\log \Gamma` versus
:math:`\log k` over the five smallest non-zero wavenumber shells gives

.. math::

   \log \Gamma = \log D + z \log k.

The measured value is :math:`z = 2.0000`.

Anomalous dimension :math:`\eta`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The current implementation prints :math:`\eta = 0` as the mean-field value.
A non-zero :math:`\eta` would require measuring the equal-time correlation
function :math:`C(k) = \langle |\hat N(\mathbf{k},t)|^2 \rangle` from an
explicit ensemble of stochastic realisations or from a loop-corrected
self-energy — both beyond the scope of the moment-closure approach.


