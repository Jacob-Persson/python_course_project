Testing
=======

The repository includes a small ``pytest`` suite to verify the deterministic physics
implemented in :class:`models.diffusion.SpectralNDE3D`.

Run the tests from the repository root:

.. code-block:: bash

   python -m pytest -q

What is tested
---------------

The tests cover:

* Correctness of the spectral Laplacian operator (via eigenmode relations).
* Conservation of total mass for pure diffusion (``rho=0``).
* Agreement with the known analytic Green's function peak for diffusion.
* Correct handling of initial conditions, including ``initial_condition.center_offset``.
* Persistence/serialization in ``utils.diagnostics.save_simulation_state``.

Equations tested
-----------------

1) Deterministic neutron diffusion / reaction-diffusion PDE

.. math::

   \frac{\partial N(\mathbf{x},t)}{\partial t} = D\nabla^2 N(\mathbf{x},t) + \rho\,N(\mathbf{x},t).

2) Spectral Laplacian eigenmode relation (used by eigenmode tests)

For a Fourier/cosine mode on the periodic FFT grid, the Laplacian satisfies

.. math::

   \nabla^2 N = -|\mathbf{k}|^2 N,

so for constant coefficients the PDE implies

.. math::

   \frac{dN}{dt} = \left(-D|\mathbf{k}|^2 + \rho\right)N.

3) Pure diffusion mass conservation (used by ``rho=0`` test)

.. math::

   M(t) = \int N(\mathbf{x},t)\,dV \approx \sum_i N_i(t)\,dv,

and for ``rho = 0`` we expect

.. math::

   M(t_{\mathrm{end}}) \approx M(0).

4) Exponential growth of total population (used by exponential-growth test)

The tests check

.. math::

   M(t) = M(0)\,e^{\rho t}.

5) Analytic 3D diffusion Green's function peak at the source (used by Green's-function test)

For pure diffusion (``rho=0``), the expected peak at radius ``r=0`` is

.. math::

   N(0,t) = \frac{A}{(4\pi D t)^{3/2}}.

6) Fourier-space reference evolution (used by analytic Fourier reference test)

Because the PDE is linear with constant coefficients, each Fourier mode evolves as

.. math::

   \widehat{N}(\mathbf{k}, t_{\mathrm{end}}) =
   e^{\left(-D|\mathbf{k}|^2 + \rho\right)t_{\mathrm{end}}}\,\widehat{N}(\mathbf{k},0).

The test computes this exactly in Fourier space (via FFT/IFTT) and compares to the numerical
``solve_ivp`` result at ``t_end``.

Representative parameter values used in tests
----------------------------------------------

Model/grid defaults from ``tests/test_physics.py::config_3d``:

* ``L = [20, 20, 20]`` (cm)
* ``nodes = [31, 31, 31]``
* initial condition: ``type="point_source"``, amplitude ``A = 10.0``
* ``t_end = 2.0`` and ``n_frames = 2``
* integration method: ``RK45``

Physics-specific overrides in the tests:

* ``test_3d_exponential_growth``:
  * ``rho = 0.1`` (from fixture)
  * sets ``D = 0.2``
  * checks ``M(t_end) \approx M(0) e^{rho t_end}`` with relative tolerance ``1e-3``.

* ``test_3d_diffusion_greens_function``:
  * sets ``rho = 0`` (pure diffusion)
  * uses fixture ``D = 0.5`` and ``A = 10.0`` at ``t_end = 2.0``
  * checks numerical peak at the source cell vs ``N(0,t) = A / (4\pi D t)^{3/2}`` with relative tolerance ``1e-2``.

* ``test_3d_pure_diffusion_mass_conservation``:
  * sets ``rho = 0`` and uses fixture ``D = 0.5``
  * checks ``M(t_end) \approx M(0)`` with relative tolerance ``1e-3``.

* ``test_3d_analytic_fourier_evolution_reference``:
  * sets ``D = 0.37`` and ``rho = 0.18``
  * checks that the field matches the exact Fourier-space evolution at ``t_end`` with max relative error ``< 1e-3``.

Model/grid defaults from ``tests/test_models.py::config_3d_basic``:

* ``L = [10, 10, 10]``
* ``nodes = [21, 21, 21]``
* default IC: ``type="uniform"`` with amplitude ``1.0``

Physics/IC-specific overrides in the model tests:

* ``test_3d_point_source_integral``:
  * point source amplitude ``A = 7.5``
  * checks ``sum(N_0)\,dv \approx A``.

* ``test_3d_point_source_center_offset``:
  * amplitude ``A = 3.0`` and ``center_offset = 2.5`` (scalar applied to x/y/z)
  * checks that the maximal grid cell matches the expected shifted index.

* ``test_3d_gaussian_center_offset_peak``:
  * Gaussian amplitude ``A = 1.0`` and ``width = 0.2`` with ``center_offset = 2.5``
  * checks that the discrete maximum occurs at the expected shifted cell.

* ``test_3d_spectral_eigenmode_diffusion``:
  * sets ``rho = 0`` and ``D = 0.75``
  * uses a cosine eigenfunction with mode indices ``(i,j,k) = (1,0,0)``
  * checks that ``rhs / N`` matches ``-D|k|^2``.

* ``test_3d_spectral_eigenmode_growth``:
  * sets ``rho = 0.25`` and ``D = 0.75``
  * uses the same cosine eigenfunction
  * checks that ``rhs / N`` matches ``-D|k|^2 + rho``.

* ``test_3d_radial_shell_integral_consistency``:
  * point source amplitude ``A = 7.0`` and ``center_offset = 1.7``
  * checks that the dv-weighted radial shell sum equals total mass at ``t=0``.

Diagnostics test parameter values:

* ``tests/test_diagnostics.py`` uses ``nodes = [5,5,5]``, ``L=[10,10,10]`` and a synthetic 2-time solution
  (``t=[0,1]``) built from the model initial condition.
* It checks that ``save_simulation_state`` writes arrays of shape ``(n_x, n_y, n_z, n_times)`` and that
  ``metadata.json`` contains:

  * ``peak_density = max(N_final)``
  * ``final_mass = sum(N_final) * dv``

