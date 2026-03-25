# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:24:34 2026

@author: jacpe396
"""

import os
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from models.diffusion import SpectralNDE3D
from solvers.integrator import run_pde_solver
from utils.plotting import (
    plot_spatial_slice_1d, plot_spatial_slice_2d,
    plot_time_space_evolution, plot_radial_evolution,
    plot_population_history
)
from utils.diagnostics import save_simulation_state

def main():
    plt.close('all')
    
    # 1. Load Config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Run Simulation
    model = SpectralNDE3D(config)
    solution = run_pde_solver(model, config)

    # 3. Handle Output Directory
    out_cfg = config.get('output', {})
    save_enabled = out_cfg.get('save_plots', False)
    if save_enabled:
        save_dir = Path(out_cfg.get('save_dir', 'results'))
        save_dir.mkdir(exist_ok=True) # Create folder if missing
    
    # 4. Generate and optionally save plots
    plot_funcs = [
        (plot_spatial_slice_1d, "spatial_1d"),
        (plot_spatial_slice_2d, "spatial_2d"),
        (plot_time_space_evolution, "time_space"),
        (plot_radial_evolution, "radial_evolution"),
        (plot_population_history, "population_history")
    ]

    for func, name in plot_funcs:
        func(model, solution)
        if save_enabled:
            # Save the current active figure before showing it
            ext = out_cfg.get('format', 'png')
            dpi = out_cfg.get('dpi', 300)
            plt.savefig(save_dir / f"{name}.{ext}", dpi=dpi, bbox_inches='tight') #

    if config['output'].get('save_data', False):
        save_simulation_state(model, solution, config)
    
    plt.show()

if __name__ == "__main__":
    main()