# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:11:18 2026

@author: jacpe396
"""

import numpy as np
import pickle
import json
from pathlib import Path

def save_simulation_state(model, solution, config):
    """
    Automatically saves the model, solution, and config to disk.
    
    Args:
        model: The initialized SpectralNDE3D object.
        solution: The OdeResult object from the solver.
        config: The original configuration dictionary.
    """
    # Output directory is controlled by config.yaml so that experiments can be
    # reproduced/compared without changing code.
    save_dir = Path(config['output'].get('save_dir', 'results'))
    save_dir.mkdir(exist_ok=True)

    # 1) Save the "heavy" space-time data:
    #    solution.y has shape (n_state, n_times) where n_state = n_x*n_y*n_z.
    #    We reshape it into a 4D tensor N[x, y, z, t] suitable for later analysis.
    #
    #    Shape after reshape: (n_x, n_y, n_z, n_times)
    full_history = solution.y.reshape((*model.nodes, -1))
    np.save(save_dir / "neutron_density_field.npy", full_history)
    np.save(save_dir / "time_points.npy", solution.t)

    # 2) Save lightweight metadata for quick sanity checks and plotting.
    #
    # Total mass / population:
    #   M(t) = \int N(\mathbf{x}, t) dV  ~=  sum_i N_i(t) * dv
    metadata = {
        "physics": config['physics'],
        "final_mass": float(np.sum(solution.y[:, -1]) * model.dv),
        "peak_density": float(np.max(solution.y[:, -1])),
        "execution_status": "Success"
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # 3. Save the Python Objects (Pickle)
    # This allows you to 'rehydrate' the session in a new script
    with open(save_dir / "sim_session.pkl", "wb") as f:
        pickle.dump({'model': model, 'solution': solution}, f)

    print(f"Done. State saved to {save_dir}/")