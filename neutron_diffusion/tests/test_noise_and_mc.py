import numpy as np
import pytest
from models.diffusion import SpectralNDE3D, MomentClosureNDE3D
from solvers.integrator import run_pde_solver


@pytest.fixture
def config_mc():
    return {
        'physics': {
            'D': 0.5,
            'rho': 0.1,
            'sigma': -0.1,
            'T1': 0.0,
            'L': [20.0, 20.0, 20.0],
            'nodes': [31, 31, 31],
        },
        'initial_condition': {'type': 'point_source', 'amplitude': 10.0},
        'simulation': {'t_end': 2.0, 'n_frames': 2, 'method': 'RK45'},
    }


def test_mc_deterministic_limit(config_mc):
    r"""
    MomentClosureNDE3D with T1=0 must reproduce the deterministic
    SpectralNDE3D result.

    When T1=0 the variance equation has no source and ⟨n²⟩ remains zero
    (to numerical precision).  The mean equation then reduces to the
    deterministic NDE, so ⟨N⟩ should match the SpectralNDE3D reference.
    """
    cfg = dict(config_mc)
    cfg["physics"] = dict(cfg["physics"])
    cfg["physics"]["T1"] = 0.0

    # Moment-closure model.
    model_mc = MomentClosureNDE3D(cfg)
    sol_mc = run_pde_solver(model_mc, cfg)

    # Deterministic reference.
    model_det = SpectralNDE3D(cfg)
    sol_det = run_pde_solver(model_det, cfg)

    n_half = model_mc.n_state_half

    # ⟨N⟩ should match N_det at every output time.
    N_mc = sol_mc.y[:n_half, :]
    N_det = sol_det.y[:, :]
    rel_err = np.max(np.abs(N_mc - N_det)) / np.max(np.abs(N_det))
    assert rel_err < 1e-6, (
        f"Mean ⟨N⟩ deviates from deterministic: rel_err={rel_err:.2e}"
    )

    # ⟨n²⟩ should be numerically zero.
    var = sol_mc.y[n_half:, :]
    assert np.all(var >= -1e-14), (
        "Variance ⟨n²⟩ has negative values below numerical tolerance"
    )
    assert np.max(var) < 1e-12, (
        f"Variance ⟨n²⟩ should be zero when T1=0, "
        f"but max={np.max(var):.2e}"
    )


def test_mc_variance_nonnegative(config_mc):
    r"""
    The variance ⟨n²⟩ must remain non-negative throughout the integration
    when noise is active (T1 > 0).

    The evolution equation for the variance contains both a positive source
    (2T₁⟨N⟩/dv) and negative sinks (diffusive dissipation, negative quadratic
    feedback).  This test ensures that the integrator never drives the
    variance below zero.
    """
    cfg = dict(config_mc)
    cfg["physics"] = dict(cfg["physics"])
    cfg["physics"]["T1"] = 1.0e-3

    model_mc = MomentClosureNDE3D(cfg)
    sol_mc = run_pde_solver(model_mc, cfg)

    n_half = model_mc.n_state_half
    var = sol_mc.y[n_half:, :]

    # Allow a small negative tolerance consistent with the solver's
    # absolute tolerance (atol=1e-6).  The RK45 internal stages can
    # produce tiny negative excursions that are well within the expected
    # numerical error; the restoring term in compute_rhs keeps them
    # bounded below 1 µ_variance.
    min_var = float(np.min(var))
    assert min_var > -1e-6, (
        f"Variance ⟨n²⟩ dropped below zero: min={min_var:.4e}"
    )


def test_mc_rhs_uniform():
    r"""
    Compare the moment-closure RHS against the analytical expression
    for a spatially uniform state with D=0 and ρ=0.

    When D=0 and the fields are uniform, all Laplacian terms vanish and
    the RHS reduces to:

    .. math::

        d⟨N⟩/dt    = σ ⟨N⟩² + σ ⟨n²⟩
        d⟨n²⟩/dt   = 4 σ ⟨N⟩ ⟨n²⟩ + 2 T₁ ⟨N⟩ / dv

    Notes:
        The sub-grid dissipative term ``−2 D k₀² ⟨n²⟩`` also vanishes
        when D=0, keeping the test expression simple.
    """
    cfg = {
        'physics': {
            'D': 0.0,
            'rho': 0.0,
            'sigma': -0.15,
            'T1': 5.0e-4,
            'L': [10.0, 10.0, 10.0],
            'nodes': [21, 21, 21],
        },
        'initial_condition': {'type': 'point_source', 'amplitude': 1.0},
        'simulation': {'t_end': 1.0, 'n_frames': 2, 'method': 'RK45'},
    }
    model = MomentClosureNDE3D(cfg)

    tot_dof = int(np.prod(model.nodes))

    # Set uniform mean and variance (already scaled by the physics).
    N0 = 3.0
    var0 = 0.5
    y = np.empty(2 * tot_dof)
    y[:tot_dof] = N0
    y[tot_dof:] = var0

    rhs = model.compute_rhs(0.0, y)

    dN_expected = cfg['physics']['sigma'] * (N0**2 + var0)
    dvar_expected = (4.0 * cfg['physics']['sigma'] * N0 * var0
                     + 2.0 * cfg['physics']['T1'] * N0 / model.dv)

    dN_actual = rhs[:tot_dof]
    dvar_actual = rhs[tot_dof:]

    assert np.allclose(dN_actual, dN_expected), (
        f"d⟨N⟩/dt mismatch: "
        f"expected {dN_expected:.6e}, got {dN_actual[0]:.6e}"
    )
    assert np.allclose(dvar_actual, dvar_expected), (
        f"d⟨n²⟩/dt mismatch: "
        f"expected {dvar_expected:.6e}, got {dvar_actual[0]:.6e}"
    )
