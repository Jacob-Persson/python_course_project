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
      D: 0.5            # Diffusion coefficient
      rho: 0.02         # Reactivity
      L: [10.0, 10.0, 10.0]  # Box dimensions (cm)
      nodes: 21         # Number of grid points per axis

    initial_condition:
      type: "gaussian"  # Options: gaussian, point_source
      amplitude: 1.0
      width: 1.5
      center_offset: 0.0  # Optional shift: scalar or [dx, dy, dz] (cm)

    output:
      save_plots: true      # Save figures to disk
      save_data: true       # Save full simulation state via utils.diagnostics
      save_dir: "results"  # Output folder under neutron_diffusion/
      format: "png"         # Figure format: png, pdf, svg, jpg
      dpi: 300              # Figure resolution

Running a Simulation
--------------------

To execute the solver and generate the plots, run the main entry point from the project root:

.. code-block:: bash

    python main.py

The script will:
1. Load the parameters from ``config.yaml``.
2. Initialize the **Spectral (FFT)** 3D model.
3. Solve the NDE using an adaptive Runge-Kutta method (RK45).
4. Display plots.

Extending the Model
-------------------

The project is designed using an **Object-Oriented** approach. To add a new physics model (e.g., a non-linear term), create a new class in ``models/`` that inherits from ``BaseModel`` and implement the ``compute_rhs`` method.