# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:58:39 2026

@author: jacpe396
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_dashboard(model, solution):
    """
    Creates a 2-panel figure: 
    Left: 2D Spatial Slice (XY)
    Right: 1D Population History (Total Mass)
    """
    # Create the figure with two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: 2D Slice ---
    N_final_flat = solution.y[:, -1]
    N_3d = N_final_flat.reshape(model.nodes)
    z_mid = model.nodes[2] // 2
    n_slice = N_3d[:, :, z_mid]
    
    im = ax1.imshow(
        n_slice.T, 
        extent=[0, model.L[0], 0, model.L[1]],
        origin='lower', 
        cmap='viridis'
    )
    fig.colorbar(im, ax=ax1, label='Density N')
    ax1.set_title(f"3D Slice at Z = {model.L[2]/2:.1f}")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")

    # --- Panel 2: Time Series ---
    dv = np.prod(model.dv)
    total_pop = [np.sum(solution.y[:, i]) * dv for i in range(len(solution.t))]
    
    ax2.plot(solution.t, total_pop, 'o-', color='firebrick', markersize=4)
    ax2.set_yscale('log')
    ax2.set_title("Total Neutron Population (Log Scale)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Integral of N")
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout() # Prevents overlapping labels
    plt.show() # Call this once at the end