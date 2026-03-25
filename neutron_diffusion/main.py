# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:24:34 2026

@author: jacpe396
"""

import yaml
import numpy as np
from pathlib import Path
from models.diffusion import SpectralNDE3D
from solvers.integrator import run_pde_solver
import matplotlib.pyplot as plt
from utils.plotting import (
    plot_spatial_slice_1d, 
    plot_time_space_evolution,
    plot_spatial_slice_2d,
    plot_population_history,
    plot_radial_evolution
)

def main():
    # 0. Close all existing plot windows from previous runs
    plt.close('all')
    
    # 1. Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Run Physics
    model = SpectralNDE3D(config)
    solution = run_pde_solver(model, config)

    # 3. Visualize
    print(f"Max N in solution: {np.max(solution.y[:, -1])}")
    # 1. 1D line plot (N vs X)
    plot_spatial_slice_1d(model, solution)
    
    # 2. 2D heatmap (N vs X, Y)
    plot_spatial_slice_2d(model, solution)
    
    # 3. 2D Hovmöller diagram (X vs Time)
    plot_time_space_evolution(model, solution)
    
    # 4. 1D Global growth (Total N vs Time)
    plot_population_history(model, solution)
    
    plot_radial_evolution(model, solution)
    
    # This single call opens ALL the windows above
    plt.show()

if __name__ == "__main__":
    main()