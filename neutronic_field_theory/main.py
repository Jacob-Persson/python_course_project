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
from utils.plotting import plot_dashboard

def main():
    # 1. Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Run Physics
    model = SpectralNDE3D(config)
    solution = run_pde_solver(model, config)

    # 3. Visualize
    print(f"Max N in solution: {np.max(solution.y[:, -1])}")
    plot_dashboard(model, solution)

if __name__ == "__main__":
    main()