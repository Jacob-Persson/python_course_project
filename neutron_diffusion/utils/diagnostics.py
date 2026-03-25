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
    save_dir = Path(config['output'].get('save_dir', 'results'))
    save_dir.mkdir(exist_ok=True)

    # 1. Save the "Heavy" Data (The 4D Space-Time Matrix)
    # Shape: (nodes_x, nodes_y, nodes_z, time_steps)
    full_history = solution.y.reshape((*model.nodes, -1))
    np.save(save_dir / "neutron_density_field.npy", full_history)
    np.save(save_dir / "time_points.npy", solution.t)

    # 2. Save the "Metadata" (Parameters) as JSON
    # This replaces the Spyder Variable Explorer for quick checking
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