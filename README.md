This code solves the 3D SPDE

$$
\partial_t N(\bold{x},t) = D\nabla^2 N(\bold{x},t) + \rho N(\bold{x},t) + \sigma N(\bold{x},t)^2 + \eta(\bold{x},t),
$$

with

$$
\langle\eta (\bold{x},t),\eta (\bold{x}',t')\rangle = 2 T_1 N(\bold{x},t)\delta(\bold{x}-\bold{x}')\,\delta(t-t') ,
$$

using a spectral (FFT) method for the Laplacian and an Ito Euler–Maruyama
scheme for the noise.  Two solution modes are available:

1. **Direct stochastic integration** — single noise realisations via
   `run_stochastic_pde_solver`.
2. **Gaussian moment closure** (`MomentClosureNDE3D`) — evolves the coupled
   system for $\langle N \rangle$ and $\langle n^2 \rangle$ with a
   zero-third-cumulant closure, giving deterministic access to the ensemble
   mean and variance without averaging over trajectories.

The moment-closure system is integrated with the same adaptive Runge–Kutta
solver used for the deterministic NDE, and all plotting/diagnostics work
identically for both modes results.

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run the simulation
From the repository root, run:

```bash
cd neutron_diffusion
python main.py
```

This reads `config.yaml`, solves the PDE, displays plots, and (optionally) saves figures and the full simulation state depending on `output.*` in the config.

## Run tests
From the repository root:

```bash
python -m pytest -q
```

## Build documentation
From `neutron_diffusion/docs/`:

```bash
python -m sphinx.cmd.build -b html source build
```

## Project structure

| Directory / file | Purpose |
|---|---|
| `models/` | Physics models (`SpectralNDE3D`, `MomentClosureNDE3D`, `BaseModel`) |
| `solvers/` | ODE integrators (`run_pde_solver`, `run_stochastic_pde_solver`) |
| `utils/` | Plotting, diagnostics, performance timer |
| `config.yaml` | All run-time parameters (YAML) |
| `main.py` | Entry point — runs solver + plots |
| `tests/` | Pytest suite |
| `docs/` | Sphinx documentation source |
