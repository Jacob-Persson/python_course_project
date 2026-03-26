# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:58:39 2026

@author: jacpe396
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plot_spatial_slice_2d(model, solution):
    """Plots a 2D XY-slice at the Z-center."""
    # Ensure we use integers for reshaping and indexing
    nodes = tuple(model.nodes.astype(int))
    N_final_3d = solution.y[:, -1].reshape(nodes)
    
    # Use a single integer for the Z-index to get a 2D result
    z_mid = nodes[2] // 2
    n_slice = N_final_3d[:, :, z_mid]  # This results in shape (nx, ny)
    
    plt.figure("2D Spatial Slice (XY)")
    plt.clf()
    # Extent needs the scalar values for the box edges
    im = plt.imshow(
        n_slice.T, 
        extent=[0, model.L[0], 0, model.L[1]],
        origin='lower', 
        cmap='viridis'
    )
    plt.colorbar(im, label='Density N')
    plt.title(f"3D Slice at Z = {model.L[2]/2:.1f} cm")
    plt.xlabel("X Position (cm)")
    plt.ylabel("Y Position (cm)")

def plot_population_history(model, solution):
    """Plots the total volume integral over time in a separate window."""
    dv = model.dv
    # Total population / mass:
    #   M(t) = ∫ N(x,y,z,t) dV  ~= sum_i N_i(t) * dv
    total_pop = [np.sum(solution.y[:, i]) * dv for i in range(len(solution.t))]
    
    plt.figure("Total Population History")
    plt.clf()
    plt.plot(solution.t, total_pop, 'o-', color='firebrick', markersize=4)
    plt.yscale('log')
    plt.title("Total Neutron Population (Log Scale)")
    plt.xlabel("Time (s)")
    plt.ylabel("Integral of N")
    plt.grid(True, which="both", ls="--", alpha=0.5)

def plot_spatial_slice_1d(model, solution):
    """
    Plots Neutron Density vs Position (X) at the center of Y and Z.
    """
    # Extract final time step
    N_final = solution.y[:, -1].reshape(model.nodes)
    
    # Take a 1D slice through the center (Y_mid, Z_mid)
    mid_y, mid_z = model.nodes[1] // 2, model.nodes[2] // 2
    n_1d = N_final[:, mid_y, mid_z]
    
    plt.figure("1D Spatial Distribution") # Unique window name
    plt.clf()
    plt.plot(np.linspace(0, model.L[0], model.nodes[0]), n_1d, 'b-', lw=2)
    plt.title(f"Density along X-axis (Y, Z center)")
    plt.xlabel("Position X (cm)")
    plt.ylabel("Neutron Density N")
    plt.grid(True, alpha=0.3)
    # Note: No plt.show() here yet

def plot_time_space_evolution(model, solution):
    """
    Plots Time-Space Evolution (Hovmöller diagram) along the X-axis.

    This visualizes the 1D slice:
        N(x, y_mid, z_mid, t)

    X-axis: Time, Y-axis: Position X, Color: Density (log scale).
    """
    plt.figure("Time-Space Evolution")
    plt.clf()

    mid_y, mid_z = model.nodes[1] // 2, model.nodes[2] // 2
    
    time_space_data = []
    for i in range(len(solution.t)):
        N_t = solution.y[:, i].reshape(model.nodes)
        time_space_data.append(N_t[:, mid_y, mid_z])
    
    data_matrix = np.array(time_space_data).T
    
    # Set vmin to avoid log(0). 
    # Usually, 1e-6 of the maximum provides a good dynamic range.
    vmax = data_matrix.max()
    vmin = max(1e-8, vmax * 1e-6)

    im = plt.imshow(
        data_matrix, 
        extent=[solution.t[0], solution.t[-1], 0, model.L[0]],
        origin='lower', 
        aspect='auto', 
        cmap='viridis',
        norm=colors.LogNorm(vmin=vmin, vmax=vmax) # Apply LogNorm here
    )
    
    plt.colorbar(im, label='Density N (Log Scale)')
    plt.title("Evolution along X-axis over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Position X (cm)")
    
def plot_radial_evolution(model, solution):
    r"""
    Plot the radial distribution of neutron density over time.

    For each time frame, we bin grid points by radius

    .. math::
        r = \sqrt{(x-c_x)^2 + (y-c_y)^2 + (z-c_z)^2},

    and compute a shell-integrated quantity (approximately):

    .. math::
        H(R_{\rm bin}, t) \approx \sum_{i\in {\rm bin}} N_i(t)\, dv
        \approx \int_{\rm shell} N(\mathbf{x},t)\, dV.
    """
    nodes = model.nodes.astype(int)
    # 1. Calculate distance from center for every grid point
    center_offset = model.ic_config.get('center_offset', 0.0)
    if np.isscalar(center_offset):
        offset_vec = np.array([float(center_offset)] * 3, dtype=float)
    else:
        offset_vec = np.array(center_offset, dtype=float)
        if offset_vec.size != 3:
            raise ValueError("center_offset must be a scalar or length-3 array.")
    cx, cy, cz = (model.L / 2 + offset_vec)
    r = np.sqrt((model.X - cx)**2 + (model.Y - cy)**2 + (model.Z - cz)**2)
    
    # Define radial bins (from center 0 to max corner distance)
    r_max = np.max(r)
    r_bins = np.linspace(0, r_max, nodes[0])
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2

    radial_time_data = []

    for i in range(len(solution.t)):
        N_3d = solution.y[:, i].reshape(nodes)
        # Bin-sum over the grid points belonging to each radial shell.
        hist, _ = np.histogram(r, bins=r_bins, weights=N_3d)
        # Approximate volume integral by including the cell volume element.
        radial_time_data.append(hist * model.dv)

    data_matrix = np.array(radial_time_data).T

    plt.figure("Radial Population Evolution")
    plt.clf()
    
    # Use LogNorm so we can see the small initial pulse and the large final growth
    current_cmap = plt.get_cmap('viridis').copy()
    current_cmap.set_bad(current_cmap(0)) # Use the darkest color in the map

    vmax = data_matrix.max()
    vmin = max(1e-6, vmax * 1e-8) # Dynamic vmin to keep contrast

    pcm = plt.pcolormesh(
        solution.t, r_centers, data_matrix,
        shading='auto',
        norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        cmap=current_cmap
    )
    
    plt.colorbar(pcm, label='Integral of N over Shell')
    #plt.colorbar(pcm, label='Total Neutrons in Shell (N * 4πr²dr)')
    plt.title("Radial Diffusion: Growth and Outward Spreading")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance from Center R (cm)")