# -*- coding: utf-8 -*-
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from models.diffusion import SpectralNDE3D, MomentClosureNDE3D
from solvers.integrator import run_pde_solver, run_stochastic_pde_solver
from utils.plotting import (
    plot_spatial_slice_1d, plot_spatial_slice_2d,
    plot_time_space_evolution, plot_radial_evolution,
    plot_population_history, plot_mean_shift, arrange_figure_windows
)
from utils.diagnostics import save_simulation_state
from utils.timer import tic, toc, report as timer_report, enable as timer_enable

def main():
    plt.close('all')

    # 1) Load configuration.
    tic('config_load')
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    toc('config_load')

    # Enable profiling if requested in config.
    if config.get('debug', {}).get('enable_timer', False):
        timer_enable()
    tic('total')

    # 2) Build model and run the primary solver.
    T1 = config['physics'].get('T1', 0.0)
    model = SpectralNDE3D(config)

    tic('solver.primary')
    if T1 > 0.0:
        solution = run_stochastic_pde_solver(model, config)
    else:
        solution = run_pde_solver(model, config)
    toc('solver.primary')

    # 2b) Linear reference (sigma=0, T1=0).
    tic('solver.linear')
    config_linear = deepcopy(config)
    config_linear['physics']['sigma'] = 0.0
    config_linear['physics']['T1'] = 0.0
    model_linear = SpectralNDE3D(config_linear)
    solution_linear = run_pde_solver(model_linear, config_linear)
    toc('solver.linear')

    # 2c) Deterministic nonlinear (same sigma, T1=0).
    tic('solver.nonlinear_det')
    config_nonlinear_det = deepcopy(config)
    config_nonlinear_det['physics']['T1'] = 0.0
    model_nonlinear_det = SpectralNDE3D(config_nonlinear_det)
    solution_nonlinear_det = run_pde_solver(model_nonlinear_det, config_nonlinear_det)
    toc('solver.nonlinear_det')

    # 2d) Moment-closure ensemble average.
    solution_mc = None
    if T1 > 0.0:
        tic('solver.moment_closure')
        model_mc = MomentClosureNDE3D(config)
        solution_mc = run_pde_solver(model_mc, config)
        toc('solver.moment_closure')

    # 3) Output directory.
    out_cfg = config.get('output', {})
    save_enabled = out_cfg.get('save_plots', False)
    if save_enabled:
        save_dir = Path(out_cfg.get('save_dir', 'results'))
        save_dir.mkdir(exist_ok=True)

    # 4) Plots.
    plot_funcs = [
        (plot_spatial_slice_1d, "spatial_1d"),
        (plot_spatial_slice_2d, "spatial_2d"),
        (plot_time_space_evolution, "time_space"),
        (plot_radial_evolution, "radial_evolution"),
        (plot_population_history, "population_history"),
    ]

    for func, name in plot_funcs:
        tic(f'plot.{name}')
        if func is plot_spatial_slice_1d:
            func(model, solution, solution_linear,
                 solution_nonlinear_det, solution_mc)
        else:
            func(model, solution, solution_linear)
        toc(f'plot.{name}')

        if save_enabled:
            ext = out_cfg.get('format', 'png')
            dpi = out_cfg.get('dpi', 300)
            plt.savefig(save_dir / f"{name}.{ext}", dpi=dpi, bbox_inches='tight')

    # 4b) Noise-induced mean shift (only when both MC and det-nonlinear exist).
    if T1 > 0.0 and solution_mc is not None and solution_nonlinear_det is not None:
        tic('plot.mean_shift')
        plot_mean_shift(model, solution_mc, solution_nonlinear_det)
        toc('plot.mean_shift')
        if save_enabled:
            ext = out_cfg.get('format', 'png')
            dpi = out_cfg.get('dpi', 300)
            plt.savefig(save_dir / f"mean_shift.{ext}", dpi=dpi, bbox_inches='tight')

    # 5) Save data.
    if config['output'].get('save_data', False):
        tic('save_data')
        save_simulation_state(model, solution, config)
        toc('save_data')

    # 6) Arrange and show.
    arrange_figure_windows()
    toc('total')
    timer_report()
    plt.show()

if __name__ == "__main__":
    main()