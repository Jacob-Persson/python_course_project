# -*- coding: utf-8 -*-
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
import numpy as np
from utils.timer import tic, toc

def run_pde_solver(model, config):
    r"""
    Run the discretized neutron diffusion PDE by integrating the induced ODE system.

    Model/PDE to ODE:
    After spatial discretization, the field ``N(x,y,z,t)`` becomes a flattened state
    vector ``N_flat(t)``. The model provides a RHS that matches

    .. math::
        \frac{d}{dt} N_{\mathrm{flat}}(t) = f(t, N_{\mathrm{flat}}(t)).

    Here, ``f`` is ``model.compute_rhs``.

    This function delegates the actual time stepping to SciPy's ``solve_ivp``,
    which implements an adaptive method such as RK45.

    Args:
        model (BaseModel): The physics model containing the RHS logic.
        config (dict): Configuration parameters including time span and method.

    Returns:
        scipy.integrate._ivp.ivp.OdeResult: The solution object from solve_ivp.
    """
    # Time interval for integration and times at which the solution is sampled.
    t_span = (0, config['simulation']['t_end'])
    t_eval = np.linspace(0, t_span[1], config['simulation']['n_frames'])

    # Build the initial field N(x,0) and flatten it to a solver state vector.
    N0 = model.get_initial_condition()

    sim_cfg = config['simulation']
    sol = solve_ivp(
        model.compute_rhs,
        t_span,
        N0,
        t_eval=t_eval,
        method=sim_cfg.get('method', 'RK45'),
        rtol=sim_cfg.get('rtol', 1e-3),
        atol=sim_cfg.get('atol', 1e-6),
    )
    return sol


def run_stochastic_pde_solver(model, config):
    r"""
    Euler–Maruyama integrator for the stochastic neutron diffusion equation.

    The SPDE (Ito interpretation)

    .. math::
        \partial_t N = D\nabla^2 N + \rho N + \sigma N^2
                      + \sqrt{\frac{2 T_1 N}{dv}}\,\dot{W}

    is discretised in time with a fixed sub-step :math:`\Delta t \le 0.05` so
    that the explicit Euler treatment of the diffusion term remains stable
    (CFL condition: :math:`D\Delta t / \Delta x^2 \lesssim 0.5`).

    The deterministic drift is obtained from ``model.compute_rhs`` (which uses
    the spectral Laplacian).  The multiplicative noise amplitude comes from
    ``model.compute_noise_amplitude``.

    Args:
        model: The physics model (must expose ``compute_rhs``,
               ``compute_noise_amplitude``, and ``nodes``).
        config: Configuration dictionary with a ``simulation`` section.

    Returns:
        OdeResult object with ``.t`` (time points) and ``.y`` (state array of
        shape ``(n_state, n_frames)``), following the same interface as
        ``run_pde_solver`` so that all plotting code works unchanged.
    """
    sim_cfg = config['simulation']
    t_span = (0, sim_cfg['t_end'])
    t_eval = np.linspace(0, t_span[1], sim_cfg['n_frames'])
    n_frames = len(t_eval)
    dt_frame = t_eval[1] - t_eval[0]

    # Sub-step chosen to satisfy the CFL condition for explicit Euler
    # on the diffusive term:  D * dt / dx^2 < 0.5.
    dx_min = float(np.min(model.dx_vec))
    dt_cfl = 0.4 * dx_min ** 2 / model.D
    dt_sub = min(dt_frame / 100, dt_cfl, 0.05)

    # Initial condition.
    state = model.get_initial_condition().copy()
    n_state = len(state)
    nc = int(np.prod(model.nodes))
    is_delayed = getattr(model, 'is_delayed', False)
    y = np.zeros((n_state, n_frames))
    y[:, 0] = state

    nodes = tuple(model.nodes.astype(int))
    rng = np.random.default_rng()

    for i in range(1, n_frames):
        t_start = t_eval[i - 1]
        t_target = t_eval[i]
        t = t_start
        while t < t_target - 1e-12:
            dt_step = min(dt_sub, t_target - t)

            tic('stoch.drift')
            drift = model.compute_rhs(t, state)
            toc('stoch.drift')

            tic('stoch.noise')
            if is_delayed:
                N_3d = state[:nc].reshape(nodes)
                M_3d = state[nc:].reshape(nodes)
                drift_N = drift[:nc].reshape(nodes)
                drift_M = drift[nc:].reshape(nodes)
                amp_N = model.compute_noise_amplitude(N_3d)
                amp_M = model.compute_precursor_noise_amplitude(M_3d)
                dW1 = rng.normal(0.0, 1.0, size=nodes).astype(np.float64)
                dW2 = rng.normal(0.0, 1.0, size=nodes).astype(np.float64)
                N_new = N_3d + drift_N * dt_step + amp_N * dW1 * np.sqrt(dt_step)
                M_new = M_3d + drift_M * dt_step + amp_M * dW2 * np.sqrt(dt_step)
                state = np.concatenate([N_new.flatten(), M_new.flatten()])
            else:
                state_3d = state.reshape(nodes)
                drift_3d = drift.reshape(nodes)
                amp = model.compute_noise_amplitude(state_3d)
                dW = rng.normal(0.0, 1.0, size=nodes).astype(np.float64)
                state = (state_3d + drift_3d * dt_step
                         + amp * dW * np.sqrt(dt_step)).flatten()
            toc('stoch.noise')

            t += dt_step

        y[:, i] = state.copy()

    return OdeResult(t=t_eval, y=y)