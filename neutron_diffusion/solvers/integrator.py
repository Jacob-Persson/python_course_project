# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:24:01 2026

@author: jacpe396
"""

from scipy.integrate import solve_ivp
import numpy as np

def run_pde_solver(model, config):
    """
    Executes the numerical integration for the given NDE model.

    Args:
        model (BaseModel): The physics model containing the RHS logic.
        config (dict): Configuration parameters including time span and method.

    Returns:
        scipy.integrate._ivp.ivp.OdeResult: The solution object from solve_ivp.
    """
    t_span = (0, config['simulation']['t_end'])
    t_eval = np.linspace(0, t_span[1], config['simulation']['n_frames'])
    N0 = model.get_initial_condition()

    sol = solve_ivp(
        model.compute_rhs, 
        t_span, 
        N0, 
        t_eval=t_eval, 
        method=config['simulation']['method']
    )
    return sol