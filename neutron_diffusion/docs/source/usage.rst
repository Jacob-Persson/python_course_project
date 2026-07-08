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
      rho: 0.05               # Reactivity (1/s)
      sigma: -0.1             # Quadratic feedback (0 = linear NDE)
      T1: 1.0e-4              # Noise amplitude (cm^3/s); 0 = deterministic
      L: [20.0, 20.0, 20.0]   # Box dimensions (cm)
      nodes: [32, 32, 32]     # Grid points per axis [nx, ny, nz]

    initial_condition:
      type: "gaussian"        # Options: gaussian, point_source, uniform
      amplitude: 1.0
      width: 1.0
      center_offset: 0.0      # Optional shift: scalar or [dx, dy, dz] (cm)

    simulation:
      t_end: 30.0             # Total integration time (s)
      n_frames: 6             # Number of output snapshots
      method: "RK45"          # ODE integrator (RK45, Radau, BDF, LSODA)
      rtol: 5.0e-3            # Relative tolerance
      atol: 1.0e-6            # Absolute tolerance

    debug:
      enable_timer: false     # Profile execution (no code changes needed)

    output:
      save_plots: false       # Save figures to disk
      save_data: false        # Save full simulation state
      save_dir: "results"     # Output folder under neutron_diffusion/
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

4. Solve three reference solutions for comparison:

   * **Linear** (``sigma=0, T1=0``)
   * **Deterministic nonlinear** (same ``sigma``, ``T1=0``)
   * **Moment closure** (only when ``T1>0``) — the Gaussian moment-closure
     system for :math:`\langle N \rangle` and :math:`\langle n^2 \rangle`.

5. Generate plots — the 1D spatial distribution overlays the main solution,
   linear reference, deterministic nonlinear reference, and moment-closure
   mean (when available).  A dedicated "Mean Shift" plot shows
   :math:`\Delta N = \langle N \rangle - N_{\rm det}`.

6. Optionally save figures and data, then display all plots.

Extending the Model
-------------------

The project is designed using an **Object-Oriented** approach. To add a new
physics model (e.g., a new non-linearity or a different closure scheme),
create a new class in ``models/`` that inherits from ``BaseModel`` (or from
``SpectralNDE3D`` to reuse the FFT infrastructure) and implement the
``compute_rhs`` method.