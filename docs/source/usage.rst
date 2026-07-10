Usage Guide
===========

This page explains how to configure and run the 3D Neutron Diffusion simulation.

Installation
------------

Ensure you have the required dependencies installed:

.. code-block:: bash

    pip install numpy scipy pyyaml matplotlib sphinx_rtd_theme

Configuration
-------------

The simulation is controlled via the ``config.yaml`` file. This allows you to modify physics and numerical parameters without touching the source code.

.. code-block:: yaml

    physics:
      D: 0.2                  # Diffusion coefficient (cm^2/s)
      rho: -1.0e-4            # Reactivity (1/s); scalar or spatial profile (see below)
      sigma: -0.05            # Quadratic feedback (0 = linear NDE)
      T1: 1.0e-5              # Noise amplitude (cm^3/s); 0 = deterministic
      beta: 0.0               # Delayed neutron fraction (0 = no precursors)
      lambda_D: 0.0           # Precursor decay constant (1/s)
      T2: 0.0                 # Precursor noise amplitude (cm^3/s)
      L: [20.0, 20.0, 20.0]   # Box dimensions (cm)
      nodes: [24, 24, 24]     # Grid points per axis [nx, ny, nz]

    initial_condition:
      type: "gaussian"        # Options: gaussian, point_source, uniform
      amplitude: 1.0
      width: 1.0
      center_offset: 0.0      # Optional shift: scalar or [dx, dy, dz] (cm)

    moment_closure:
      order: 3                # 2 = Gaussian closure, 3 = third-order closure

    simulation:
      t_end: 30.0             # Total integration time (s)
      n_frames: 6             # Number of output snapshots
      method: "RK45"          # ODE integrator (RK45, Radau, BDF, LSODA)
      rtol: 5.0e-3            # Relative tolerance
      atol: 1.0e-6            # Absolute tolerance

    debug:
      enable_timer: true      # Profile execution (no effect if false)

    output:
      save_plots: false       # Save figures to disk
      save_data: false        # Save full simulation state
      save_dir: "results"     # Output folder
      format: "png"           # Figure format: png, pdf, svg, jpg
      dpi: 300                # Figure resolution

Running a Simulation
--------------------

To execute the solver and generate the plots, run the entry point from the project root:

.. code-block:: bash

    python main.py

The script will:

1. Load parameters from ``config.yaml``.
2. Build the **Spectral (FFT)** 3D model.
3. Solve the NDE — chooses the solver based on ``physics.T1``:

   * ``T1 = 0`` → deterministic Runge–Kutta integrator (``run_pde_solver``).
   * ``T1 > 0`` → explicit Euler–Maruyama integrator (``run_stochastic_pde_solver``).

4. Solve reference solutions for comparison:

   * **Linear** (``sigma=0, T1=0``)
   * **Deterministic nonlinear** (same ``sigma``, ``T1=0``)
   * **Moment closure** (only when ``T1>0``) — the moment-closure
     system for :math:`\langle N \rangle` and :math:`\langle n^2
     \rangle` (Gaussian closure, ``order=2``) or the three-field
     system including :math:`\langle n^3 \rangle` (third-order closure,
     ``order=3``).

5. Optionally extract the mean-field critical exponents :math:`\nu`,
   :math:`z`, and :math:`\eta` from the linearised propagator.

6. Generate plots — the 1D spatial distribution overlays the main solution,
   linear reference, deterministic nonlinear reference, and moment-closure
   mean (when available).  A dedicated "Mean Shift" plot shows
   :math:`\Delta N = \langle N \rangle - N_{\rm det}`.

6. Optionally save figures and data, then display all plots.

Spatial reactivity
------------------

By default ``rho`` is a constant scalar.  To specify a space-dependent
reactivity :math:`\rho(\mathbf{x})`, pass a dictionary instead:

.. code-block:: yaml

    rho:
      type: step           # uniform | step | gaussian | sinusoidal | radial
      inside: 0.01         # reactivity inside the step region
      outside: -0.01       # reactivity outside
      width: 3.0           # step width (cm)
      axis: x              # axis along which the step varies (x, y, or z)

Each profile type uses its own set of keys:

* **uniform**: ``value`` — constant value everywhere (equivalent to the scalar form).
* **step**: ``inside``, ``outside``, ``width`` (cm), ``axis``.
* **gaussian**: ``peak`` (value at centre), ``background`` (value at infinity), ``width`` (:math:`\sigma`, cm).
* **sinusoidal**: ``mean``, ``amplitude``, ``freq`` (list of waves per axis, e.g. ``[1, 1, 1]``).
* **radial**: ``inner``, ``outer``, ``radius`` (cm), ``sharpness`` (sigmoid steepness), ``center`` (offset, list of three floats).

All three solution modes (deterministic NDE, Gaussian closure, third-order
closure) accept spatial reactivity without modification.

Extending the Model
-------------------

The project is designed using an **Object-Oriented** approach. To add a new
physics model (e.g., a new non-linearity or a different closure scheme),
create a new class in ``models/`` that inherits from ``BaseModel`` (or from
``SpectralNDE3D`` to reuse the FFT infrastructure) and implement the
``compute_rhs`` method.