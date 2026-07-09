# Neutron diffusion

Spectral 3D solver for the stochastic neutron diffusion equation

$$
\partial_t N(\mathbf{x},t) = D\nabla^2 N(\mathbf{x},t) + \rho N(\mathbf{x},t) + \sigma N(\mathbf{x},t)^2 + \eta(\mathbf{x},t),
$$

with

$$
\langle\eta (\mathbf{x},t),\eta (\mathbf{x}',t')\rangle = 2 T_1 N(\mathbf{x},t)\delta(\mathbf{x}-\mathbf{x}')\,\delta(t-t') ,
$$

using FFT-based Laplacian and Ito Euler–Maruyama noise integration.

## Solution modes

| Mode | Description |
|---|---|
| **Deterministic NDE** ($T_1=0$, $\sigma=0$) | Linear diffusion + growth |
| **Nonlinear deterministic** ($T_1=0$, $\sigma\neq0$) | Quadratic feedback |
| **Direct stochastic** ($T_1>0$) | Single noise realisations with CFL sub-stepping |
| **Gaussian moment closure** (`order: 2`) | Evolves $\langle N\rangle$, $\langle n^2\rangle$; $\langle n^3\rangle=0$ |
| **Third-order closure** (`order: 3`) | Evolves $\langle N\rangle$, $\langle n^2\rangle$, $\langle n^3\rangle$; $\langle n^4\rangle=3\langle n^2\rangle^2$ |

Both closures use adaptive RK45 (deterministic) and include sub-grid scale dissipation. Critical exponents ($\nu=1/2$, $z=2$, $\eta=0$) are extracted from spectral data.

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python main.py
```

Tests:
```bash
python -m pytest -q
```

Docs:
```bash
cd docs
python -m sphinx.cmd.build -b html source build
```

## Project structure

| Path | Purpose |
|---|---|
| `main.py` | Entry point |
| `config.yaml` | Run-time parameters |
| `models/` | `SpectralNDE3D`, `MomentClosureNDE3D`, `ThirdOrderMomentClosureNDE3D` |
| `solvers/` | `run_pde_solver`, `run_stochastic_pde_solver` |
| `utils/` | Plotting, diagnostics, timer, critical exponents |
| `tests/` | Pytest suite |
| `docs/` | Sphinx documentation |
