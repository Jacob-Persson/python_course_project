# Neutron diffusion

Spectral 3D solver for the stochastic neutron diffusion equation.

$$
\partial_t N(\mathbf{x},t) = D\nabla^2 N(\mathbf{x},t) + \rho N(\mathbf{x},t) + \sigma N(\mathbf{x},t)^2 + \eta(\mathbf{x},t),
$$

with

$$
\langle\eta (\mathbf{x},t),\eta (\mathbf{x}',t')\rangle = 2 T_1 N(\mathbf{x},t)\delta(\mathbf{x}-\mathbf{x}')\,\delta(t-t') ,
$$

Uses an FFT-based spectral Laplacian (periodic boundary conditions) and
Ito Euler--Maruyama integration for noise.

## Requirements

```bash
pip install -r requirements.txt
```

For 2--4$\times$ faster FFTs, also install pyFFTW:

```bash
pip install pyfftw
```

The code falls back to `numpy.fft` if pyFFTW is not available.

## Quick start

```bash
python main.py
```

Tests:

```bash
python -m pytest -q
```

Documentation:

```bash
cd docs
python -m sphinx.cmd.build -b html source build
```

## Configuration

Edit `config.yaml` to control all physics and numerical parameters.

### Physics

| Parameter | Default | Description |
|-----------|---------|-------------|
| `D` | `0.2` | Diffusion coefficient (cm²/s) |
| `rho` | `-1e-4` | Reactivity (1/s) — scalar or spatial profile (see below) |
| `sigma` | `-0.05` | Quadratic feedback coefficient (1/(cm³·s)); 0 = linear |
| `T1` | `1e-5` | Noise amplitude (cm³/s); 0 = deterministic |
| `beta` | `1e-3` | Delayed neutron fraction; 0 = no delayed neutrons |
| `lambda_D` | `0.01` | Precursor decay constant (1/s) |
| `T2` | `1e-6` | Precursor noise amplitude (cm³/s) |
| `L` | `[20,20,20]` | Domain size in x, y, z (cm) |
| `nodes` | `[32,32,32]` | Grid points per dimension |

### Spatial reactivity profiles

`rho` can be a scalar (uniform) or a dictionary:

```yaml
rho: -1.0e-4                                    # Uniform
rho: {type: step,       inside: 0.05, outside: -0.01, width: 5.0}
rho: {type: gaussian,   peak: 0.01, background: -0.01, width: 2.0}
rho: {type: sinusoidal, mean: 0.0, amplitude: 0.01, freq: [1, 1, 1]}
rho: {type: radial,     inner: 0.05, outer: -0.01, radius: 3, sharpness: 10}
```

### Initial condition

```yaml
initial_condition:
  type: "gaussian"          # gaussian, uniform, or point_source
  amplitude: 1.0
  width: 2.0                # Gaussian sigma (cm)
  center_offset: 0.0        # 0.0 = centered; scalar or [dx, dy, dz]
```

### Moment closure

```yaml
moment_closure:
  order: 2                  # 2 = Gaussian closure, 3 = third-order
```

### Solver

```yaml
simulation:
  t_end: 50.0               # Integration time (s)
  n_frames: 50              # Output frames (for plots)
  method: "RK45"            # Switch to "Radau" for stiff problems
  rtol: 1.0e-6
  atol: 1.0e-8
```

## Solution modes

| Mode | Conditions | What is solved |
|---|---|---|
| **Deterministic NDE** | `T1=0, sigma=0` | Linear diffusion + growth, adaptive RK45 |
| **Nonlinear deterministic** | `T1=0, sigma!=0` | Quadratic feedback, adaptive RK45 |
| **Direct stochastic** | `T1>0` | Single noise realisations, Euler--Maruyama with CFL sub-stepping |
| **Gaussian moment closure** | `T1>0, order: 2` | Evolves `⟨N⟩`, `⟨n²⟩` (closure: `⟨n³⟩=0`) |
| **Third-order closure** | `T1>0, order: 3` | Evolves `⟨N⟩`, `⟨n²⟩`, `⟨n³⟩` (closure: `⟨n⁴⟩=3⟨n²⟩²`) |
| **Delayed neutrons** | `beta>0` | Coupled N + precursor M system, all modes above |

Both moment closures are deterministic (adaptive RK45) and include sub-grid
diffusive dissipation. Critical exponents ($\nu=1/2$, $z=2$, $\eta=0$) are
extracted from spectral data.

## Output

| Setting | Artifacts |
|---------|-----------|
| `save_plots: true` | PNG figures in `results/` |
| `save_data: true` | `neutron_density_field.npy`, `time_points.npy`, `metadata.json`, `sim_session.pkl` |

Reload a saved session with:

```bash
python variable_explorer.py results/
```

## Project structure

| Path | Contents |
|------|----------|
| `main.py` | Entry point, runs solvers and plots |
| `config.yaml` | All run-time parameters |
| `models/base_model.py` | Abstract base class |
| `models/diffusion.py` | `SpectralNDE3D`, `MomentClosureNDE3D`, `ThirdOrderMomentClosureNDE3D`, `DelayedNeutronSpectralNDE3D` |
| `solvers/integrator.py` | `run_pde_solver` (RK45), `run_stochastic_pde_solver` (Euler--Maruyama) |
| `utils/plotting.py` | Spatial slices, Hovmöller, radial, population history, mean shift |
| `utils/diagnostics.py` | Save/load simulation state (NPY, JSON, pickle) |
| `utils/critical_exponents.py` | Mean-field exponent extraction ($\nu$, $z$) |
| `utils/timer.py` | Profiling with `tic`/`toc` |
| `variable_explorer.py` | Load `sim_session.pkl` into Spyder Variable Explorer |
| `tests/` | 39 pytests |
| `docs/` | Sphinx documentation |

## Boundary conditions

The FFT imposes **periodic boundary conditions**. The grid is constructed with
`endpoint=False` so that $x=0$ and $x=L$ are the same point. For problems
requiring reflecting or zero-flux boundaries, the domain should be large enough
that the solution does not reach the boundary within the simulation time.
