# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:24:01 2026

@author: jacpe396
"""

from scipy.integrate import solve_ivp
import numpy as np

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

    sol = solve_ivp(
        model.compute_rhs, 
        t_span, 
        N0, 
        t_eval=t_eval, 
        method=config['simulation']['method']
    )
    return sol