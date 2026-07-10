# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from itertools import cycle

def _get_nc(model):
    return int(np.prod(model.nodes.astype(int)))

def plot_spatial_slice_2d(model, solution, solution_linear=None):
    """Plots a 2D XY-slice at the Z-center."""
    nodes = tuple(model.nodes.astype(int))
    nc = _get_nc(model)
    N_final_3d = solution.y[:nc, -1].reshape(nodes)
    
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
    t_final = solution.t[-1]
    plt.title(f"3D Slice at Z = {model.L[2]/2:.1f} cm, t = {t_final:.2f} s")
    plt.xlabel("X Position (cm)")
    plt.ylabel("Y Position (cm)")

def plot_population_history(model, solution, solution_linear=None):
    """Plots the total volume integral over time in a separate window."""
    dv = model.dv
    nc = _get_nc(model)
    total_pop = [np.sum(solution.y[:nc, i]) * dv for i in range(len(solution.t))]
    
    plt.figure("Total Population History")
    plt.clf()
    plt.plot(solution.t, total_pop, 'o-', color='firebrick', markersize=4,
             label=f'σ = {model.sigma:.3f}')
    
    if solution_linear is not None:
        total_pop_lin = [np.sum(solution_linear.y[:nc, i]) * dv
                         for i in range(len(solution_linear.t))]
        plt.plot(solution_linear.t, total_pop_lin, '--', color='grey',
                 label='σ = 0 (linear)')
    
    if max(total_pop) > 0:
        plt.yscale('log')
        scale_label = " (Log Scale)"
    else:
        scale_label = ""
    plt.title(f"Total Neutron Population{scale_label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Integral of N")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()


def plot_mean_shift(model, solution_mc, solution_nonlinear_det):
    r"""
    Plot the noise-induced mean shift :math:`\Delta N(\mathbf{x}) =
    \langle N \rangle - N_{\rm det}` at the final simulation time along the
    centre line (Y, Z mid-plane).

    This visualises the effect of the multiplicative noise on the ensemble-
    averaged density relative to the deterministic nonlinear reference.
    """
    nc = int(np.prod(model.nodes.astype(int)))
    N_mc = solution_mc.y[:nc, -1].reshape(model.nodes)
    N_det = solution_nonlinear_det.y[:nc, -1].reshape(model.nodes)

    is_3rd = (solution_mc.y.shape[0] == 3 * nc)
    label = ('⟨N⟩ (3rd-order closure)' if is_3rd
             else '⟨N⟩ (Gaussian closure)')

    mid_y, mid_z = model.nodes[1] // 2, model.nodes[2] // 2
    delta = N_mc[:, mid_y, mid_z] - N_det[:, mid_y, mid_z]
    x_axis = np.linspace(0, model.L[0], model.nodes[0])

    plt.figure("Mean Shift")
    plt.clf()
    plt.plot(x_axis, delta, 'r-', lw=2, label=label)
    plt.axhline(0, color='grey', ls='--', lw=1)
    t_final = solution_mc.t[-1]
    plt.title(f"Noise-Induced Mean Shift, t = {t_final:.2f} s")
    plt.xlabel("Position X (cm)")
    plt.ylabel(r"$\Delta N = \langle N \rangle - N_{\rm det}$")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)


def plot_spatial_slice_1d(model, solution, solution_linear=None,
                          solution_nonlinear_det=None, solution_mc=None):
    """
    Plots Neutron Density vs Position (X) at the center of Y and Z.
    """
    nc = _get_nc(model)
    N_final = solution.y[:nc, -1].reshape(model.nodes)
    mid_y, mid_z = model.nodes[1] // 2, model.nodes[2] // 2
    n_1d = N_final[:, mid_y, mid_z]
    x_axis = np.linspace(0, model.L[0], model.nodes[0])

    plt.figure("1D Spatial Distribution")
    plt.clf()

    # --- Stochastic / main solution (center slice) ---
    plt.plot(x_axis, n_1d, 'b-', lw=2,
             label=f'σ = {model.sigma:.3f}, T₁ = {model.T1:.1e}')

    # --- Moment-closure ensemble average ⟨N⟩ ---
    if model.T1 > 0 and solution_mc is not None:
        n_half = int(np.prod(model.nodes.astype(int)))
        is_3rd = (solution_mc.y.shape[0] == 3 * n_half)
        N_mc = solution_mc.y[:n_half, -1].reshape(model.nodes)
        n_1d_mc = N_mc[:, mid_y, mid_z]
        mc_label = ('⟨N⟩ (3rd-order closure)' if is_3rd
                    else '⟨N⟩ (moment closure)')
        plt.plot(x_axis, n_1d_mc, '-.', color='blue', lw=2, label=mc_label)

    # --- Deterministic nonlinear (same σ, T₁ = 0) ---
    if model.T1 > 0 and solution_nonlinear_det is not None:
        N_final_det = solution_nonlinear_det.y[:nc, -1].reshape(model.nodes)
        n_1d_det = N_final_det[:, mid_y, mid_z]
        plt.plot(x_axis, n_1d_det, '--', color='black', lw=1.5,
                 label=f'σ = {model.sigma:.3f}, T₁ = 0')

    # --- Linear reference (σ = 0, T₁ = 0) ---
    if solution_linear is not None:
        N_final_lin = solution_linear.y[:nc, -1].reshape(model.nodes)
        n_1d_lin = N_final_lin[:, mid_y, mid_z]
        plt.plot(x_axis, n_1d_lin, '--', color='grey', lw=1.5,
                 label='σ = 0, T₁ = 0 (linear)')

    t_final = solution.t[-1]
    plt.title(f"Density along X-axis (Y, Z center), t = {t_final:.2f} s")
    plt.xlabel("Position X (cm)")
    plt.ylabel("Neutron Density N")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')

def plot_time_space_evolution(model, solution, solution_linear=None):
    """
    Plots Time-Space Evolution (Hovmöller diagram) along the X-axis.

    This visualizes the 1D slice:
        N(x, y_mid, z_mid, t)

    X-axis: Time, Y-axis: Position X, Color: Density (log scale).
    """
    plt.figure("Time-Space Evolution")
    plt.clf()

    mid_y, mid_z = model.nodes[1] // 2, model.nodes[2] // 2
    nc = _get_nc(model)
    
    time_space_data = []
    for i in range(len(solution.t)):
        N_t = solution.y[:nc, i].reshape(model.nodes)
        time_space_data.append(N_t[:, mid_y, mid_z])
    
    data_matrix = np.array(time_space_data).T
    
    vmax = data_matrix.max()
    if vmax > 0:
        # LogNorm with a sensible dynamic range floor ensures non-zero
        # regions are visible while leaving true zeros white.
        vmin = max(1e-8, vmax * 1e-6)
        data_plot = np.nan_to_num(data_matrix, nan=vmin)
        data_plot = np.clip(data_plot, 1e-6, None)

        im = plt.imshow(
            data_plot,
            extent=[solution.t[0], solution.t[-1], 0, model.L[0]],
            origin='lower',
            aspect='auto',
            cmap='viridis',
            norm=colors.LogNorm(vmin=vmin, vmax=vmax)
        )
        plt.colorbar(im, label='Density N (Log Scale)')
    else:
        plt.imshow(
            data_matrix,
            extent=[solution.t[0], solution.t[-1], 0, model.L[0]],
            origin='lower',
            aspect='auto',
            cmap='viridis',
        )
    plt.title("Evolution along X-axis over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Position X (cm)")
    
def plot_radial_evolution(model, solution, solution_linear=None):
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

    nc = _get_nc(model)
    radial_time_data = []

    for i in range(len(solution.t)):
        N_3d = solution.y[:nc, i].reshape(nodes)
        # Bin-sum over the grid points belonging to each radial shell.
        hist, _ = np.histogram(r, bins=r_bins, weights=N_3d)
        # Approximate volume integral by including the cell volume element.
        radial_time_data.append(hist * model.dv)

    data_matrix = np.array(radial_time_data).T

    plt.figure("Radial Population Evolution")
    plt.clf()
    
    vmax = data_matrix.max()
    if vmax > 0:
        # Use LogNorm so we can see the small initial pulse and the large growth
        current_cmap = plt.get_cmap('viridis').copy()
        current_cmap.set_bad(current_cmap(0))

        vmin = max(1e-6, vmax * 1e-8)
        data_plot = np.nan_to_num(data_matrix, nan=vmin)
        data_plot = np.clip(data_plot, 1e-6, None)

        pcm = plt.pcolormesh(
            solution.t, r_centers, data_plot,
            shading='auto',
            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            cmap=current_cmap
        )
        plt.colorbar(pcm, label='Integral of N over Shell')
    else:
        plt.pcolormesh(
            solution.t, r_centers, data_matrix,
            shading='auto',
            cmap='viridis',
        )
    plt.title("Radial Diffusion: Growth and Outward Spreading")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance from Center R (cm)")


def plot_rho_field(model):
    r"""
    Plot the reactivity field :math:`\rho(\mathbf{x})` along the centre line
    (Y, Z mid-plane) if it is not spatially uniform.

    When the field is uniform the plot is skipped (no window created).
    """
    rho = model.rho_field
    if np.ptp(rho) < 1e-16:
        return

    mid_y, mid_z = model.nodes[1] // 2, model.nodes[2] // 2
    rho_1d = rho[:, mid_y, mid_z]
    x_axis = np.linspace(0, model.L[0], model.nodes[0])

    plt.figure("Reactivity Profile")
    plt.clf()
    plt.plot(x_axis, rho_1d, 'g-', lw=2)

    plt.title(r"Reactivity $\rho(x)$ along X-axis (Y, Z centre)")
    plt.xlabel("Position X (cm)")
    plt.ylabel(r"Reactivity $\rho$ (s$^{-1}$)")
    plt.grid(True, alpha=0.3)
    # Show min/max values in the corner.
    plt.text(0.98, 0.05,
             f"min = {rho.min():.4e}\nmax = {rho.max():.4e}",
             transform=plt.gca().transAxes,
             ha='right', va='bottom',
             bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))


def arrange_figure_windows(stack_on_target=None):
    """
    Place all open matplotlib figure windows in a non-overlapping grid
    across the screen, so that every plot is visible at once.

    Parameters
    ----------
    stack_on_target : dict, optional
        Mapping ``{source_label: target_label}``.  Source figures are
        excluded from the grid and placed at the same screen position
        as the target figure instead (they overlap).
    """
    fig_nums = plt.get_fignums()
    if not fig_nums:
        return

    # --- screen dimensions (cross-platform fallback) ---
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.destroy()
    except Exception:
        sw, sh = 1920, 1080

    # Determine which figures go in the grid and which are stacked.
    stack_sources = set(stack_on_target or {})
    grid_figs = [(fn, plt.figure(fn))
                 for fn in fig_nums
                 if plt.figure(fn).get_label() not in stack_sources]

    n = len(grid_figs)
    if n == 0:
        return
    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols
    win_w = sw // n_cols
    win_h = sh // n_rows

    # Build a {label: (x, y)} lookup for all grid figures.
    pos_by_label = {}
    for i, (fn, fig) in enumerate(grid_figs):
        col = i % n_cols
        row = i // n_cols
        x = col * win_w
        y = sh - (row + 1) * win_h
        pos_by_label[fig.get_label()] = (x, y)

        try:
            fig.canvas.manager.window.wm_geometry(f"+{x}+{y}")
        except AttributeError:
            try:
                fig.canvas.manager.window.move(x, y)
            except AttributeError:
                pass

    # Stack sources on top of their targets.
    if not stack_on_target:
        return
    for fn in fig_nums:
        label = plt.figure(fn).get_label()
        if label not in stack_sources:
            continue
        target_label = stack_on_target[label]
        if target_label in pos_by_label:
            tx, ty = pos_by_label[target_label]
        else:
            # Target not found — place at the next free grid slot.
            n_grid = len(pos_by_label)
            col = n_grid % n_cols
            row = n_grid // n_cols
            tx = col * win_w
            ty = sh - (row + 1) * win_h
        fig = plt.figure(fn)
        try:
            fig.canvas.manager.window.wm_geometry(f"+{tx}+{ty}")
        except AttributeError:
            try:
                fig.canvas.manager.window.move(tx, ty)
            except AttributeError:
                pass