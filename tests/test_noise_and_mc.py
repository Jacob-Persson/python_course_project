import copy

import numpy as np
import pytest
from models.diffusion import (
    SpectralNDE3D,
    MomentClosureNDE3D,
    ThirdOrderMomentClosureNDE3D,
)
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
    cfg = copy.deepcopy(config_mc)
    cfg["physics"]["T1"] = 0.0

    # Moment-closure model.
    model_mc = MomentClosureNDE3D(cfg)
    sol_mc = run_pde_solver(model_mc, cfg)

    # Deterministic reference.
    model_det = SpectralNDE3D(cfg)
    sol_det = run_pde_solver(model_det, cfg)

    n_block = model_mc.n_state_block

    # ⟨N⟩ should match N_det at every output time.
    N_mc = sol_mc.y[:n_block, :]
    N_det = sol_det.y[:, :]
    rel_err = np.max(np.abs(N_mc - N_det)) / np.max(np.abs(N_det))
    assert rel_err < 1e-6, (
        f"Mean ⟨N⟩ deviates from deterministic: rel_err={rel_err:.2e}"
    )

    # ⟨n²⟩ should be numerically zero.
    var = sol_mc.y[n_block:, :]
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
    cfg = copy.deepcopy(config_mc)
    cfg["physics"]["T1"] = 1.0e-3

    model_mc = MomentClosureNDE3D(cfg)
    sol_mc = run_pde_solver(model_mc, cfg)

    n_block = model_mc.n_state_block
    var = sol_mc.y[n_block:, :]

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


# ---------------------------------------------------------------------------
# Third-order moment closure (ThirdOrderMomentClosureNDE3D)
# ---------------------------------------------------------------------------

@pytest.fixture
def config_mc3():
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


def test_mc3_deterministic_limit(config_mc3):
    r"""
    ThirdOrderMomentClosureNDE3D with T1=0 must reproduce the
    deterministic SpectralNDE3D result.

    When T1=0 the variance and third-moment equations have no source
    and both remain zero (to numerical precision).  The mean equation
    then reduces to the deterministic NDE.
    """
    cfg = copy.deepcopy(config_mc3)
    cfg["physics"]["T1"] = 0.0

    model_mc3 = ThirdOrderMomentClosureNDE3D(cfg)
    sol_mc3 = run_pde_solver(model_mc3, cfg)

    model_det = SpectralNDE3D(cfg)
    sol_det = run_pde_solver(model_det, cfg)

    nb = model_mc3.n_state_block

    N_mc3 = sol_mc3.y[:nb, :]
    N_det = sol_det.y[:, :]
    rel_err = np.max(np.abs(N_mc3 - N_det)) / np.max(np.abs(N_det))
    assert rel_err < 1e-6, (
        f"Mean ⟨N⟩ deviates from deterministic: rel_err={rel_err:.2e}"
    )

    var = sol_mc3.y[nb:2*nb, :]
    assert np.max(var) < 1e-12, (
        f"Variance should be zero when T1=0, max={np.max(var):.2e}"
    )

    third = sol_mc3.y[2*nb:, :]
    assert np.max(np.abs(third)) < 1e-12, (
        f"Third moment should be zero when T1=0, "
        f"max_abs={np.max(np.abs(third)):.2e}"
    )


def test_mc3_variance_nonnegative(config_mc3):
    r"""
    The variance ⟨n²⟩ must remain non-negative throughout the
    integration when noise is active (T1 > 0).

    This is the same check as test_mc_variance_nonnegative but for
    the third-order closure.
    """
    cfg = copy.deepcopy(config_mc3)
    cfg["physics"]["T1"] = 1.0e-3

    model_mc3 = ThirdOrderMomentClosureNDE3D(cfg)
    sol_mc3 = run_pde_solver(model_mc3, cfg)

    nb = model_mc3.n_state_block
    var = sol_mc3.y[nb:2*nb, :]
    min_var = float(np.min(var))
    assert min_var > -1e-6, (
        f"Variance ⟨n²⟩ dropped below zero: min={min_var:.4e}"
    )


def test_mc3_rhs_uniform():
    r"""
    Compare the third-order closure RHS against the analytical
    expression for a spatially uniform state with D=0 and ρ=0.

    When D=0 and the fields are uniform all Laplacian terms vanish
    and the RHS reduces to:

    .. math::

        d⟨N⟩/dt    = σ ⟨N⟩² + σ ⟨n²⟩
        d⟨n²⟩/dt   = 4 σ ⟨N⟩ ⟨n²⟩ + 2 σ ⟨n³⟩ + 2 T₁ ⟨N⟩ / dv
        d⟨n³⟩/dt   = 6 σ ⟨N⟩ ⟨n³⟩ + 6 σ ⟨n²⟩² + 6 T₁ ⟨n²⟩ / dv
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
    model = ThirdOrderMomentClosureNDE3D(cfg)

    tot_dof = int(np.prod(model.nodes))
    N0, var0, third0 = 3.0, 0.5, -0.05
    y = np.empty(3 * tot_dof)
    y[:tot_dof] = N0
    y[tot_dof:2*tot_dof] = var0
    y[2*tot_dof:] = third0

    rhs = model.compute_rhs(0.0, y)

    sig = cfg['physics']['sigma']
    T1 = cfg['physics']['T1']
    dv = model.dv

    dN_expected = sig * (N0**2 + var0)
    dvar_expected = (4.0 * sig * N0 * var0
                     + 2.0 * sig * third0
                     + 2.0 * T1 * N0 / dv)
    dthird_expected = (6.0 * sig * N0 * third0
                       + 6.0 * sig * var0**2
                       + 6.0 * T1 * var0 / dv)

    dN_actual = rhs[:tot_dof]
    dvar_actual = rhs[tot_dof:2*tot_dof]
    dthird_actual = rhs[2*tot_dof:]

    assert np.allclose(dN_actual, dN_expected), (
        f"d⟨N⟩/dt: expected {dN_expected:.6e}, got {dN_actual[0]:.6e}"
    )
    assert np.allclose(dvar_actual, dvar_expected), (
        f"d⟨n²⟩/dt: expected {dvar_expected:.6e}, "
        f"got {dvar_actual[0]:.6e}"
    )
    assert np.allclose(dthird_actual, dthird_expected), (
        f"d⟨n³⟩/dt: expected {dthird_expected:.6e}, "
        f"got {dthird_actual[0]:.6e}"
    )


def test_mc3_third_moment_source():
    r"""
    The third moment must receive a positive source from the variance
    when T₁ > 0, even with σ = 0.

    With σ = 0, T₁ > 0, and non-zero initial variance, the initial
    RHS for the third moment is d⟨n³⟩/dt = 6 T₁ ⟨n²⟩ / dv > 0.
    """
    cfg = {
        'physics': {
            'D': 0.0,
            'rho': 0.0,
            'sigma': 0.0,
            'T1': 1.0e-3,
            'L': [10.0, 10.0, 10.0],
            'nodes': [21, 21, 21],
        },
        'initial_condition': {'type': 'uniform', 'amplitude': 5.0},
        'simulation': {'t_end': 0.1, 'n_frames': 2, 'method': 'RK45'},
    }
    model = ThirdOrderMomentClosureNDE3D(cfg)

    tot_dof = int(np.prod(model.nodes))
    y = np.empty(3 * tot_dof)
    y[:tot_dof] = 5.0
    y[tot_dof:2*tot_dof] = 0.2
    y[2*tot_dof:] = 0.0

    rhs = model.compute_rhs(0.0, y)
    dthird_actual = rhs[2*tot_dof:]

    T1 = cfg['physics']['T1']
    dv = model.dv
    dthird_expected = 6.0 * T1 * 0.2 / dv

    assert np.allclose(dthird_actual, dthird_expected), (
        f"d⟨n³⟩/dt mismatch: expected {dthird_expected:.6e}, "
        f"got {dthird_actual[0]:.6e}"
    )
    assert float(dthird_actual[0]) > 0, (
        "Third-moment source must be positive when T₁ > 0 and ⟨n²⟩ > 0"
    )


# ---------------------------------------------------------------------------
# Noise amplitude helper
# ---------------------------------------------------------------------------

def test_compute_noise_amplitude_zero_when_T1_zero(config_mc):
    """Noise amplitude must be zero everywhere when T1=0."""
    cfg = copy.deepcopy(config_mc)
    cfg["physics"]["T1"] = 0.0

    model = SpectralNDE3D(cfg)
    N = np.ones(model.nodes)
    amp = model.compute_noise_amplitude(N)

    assert np.all(amp == 0.0), "Noise amplitude must be zero when T1=0"


def test_compute_noise_amplitude_form():
    r"""
    Noise amplitude must satisfy g = sqrt(2 * T1 * N / dv).
    """
    cfg = {
        'physics': {
            'D': 0.5, 'rho': 0.1, 'sigma': 0.0, 'T1': 2.0e-4,
            'L': [10.0, 10.0, 10.0], 'nodes': [8, 8, 8],
        },
        'initial_condition': {'type': 'uniform', 'amplitude': 3.0},
    }
    model = SpectralNDE3D(cfg)
    N = np.full(model.nodes, 4.0)
    amp = model.compute_noise_amplitude(N)
    expected = np.sqrt(2.0 * model.T1 * 4.0 / model.dv)
    assert np.allclose(amp, expected), (
        f"Noise amplitude mismatch: max diff = {np.max(np.abs(amp - expected)):.4e}"
    )


def test_compute_noise_amplitude_negative_N():
    """Noise amplitude must handle negative N by clipping to zero."""
    cfg = {
        'physics': {
            'D': 0.5, 'rho': 0.1, 'sigma': 0.0, 'T1': 2.0e-4,
            'L': [10.0, 10.0, 10.0], 'nodes': [8, 8, 8],
        },
        'initial_condition': {'type': 'uniform', 'amplitude': 3.0},
    }
    model = SpectralNDE3D(cfg)
    N = np.full(model.nodes, -2.0)
    amp = model.compute_noise_amplitude(N)
    assert np.all(amp == 0.0), "Noise amplitude must be zero for negative N"


# ---------------------------------------------------------------------------
# Stochastic solver (Euler–Maruyama)
# ---------------------------------------------------------------------------

def test_stochastic_solver_deterministic_limit():
    r"""
    With T1=0 the stochastic solver should be broadly consistent with
    the deterministic RK45 solver (Euler–Maruyama uses a fixed step
    while the deterministic solver uses adaptive RK45 — a small
    discrepancy is expected).
    """
    cfg = {
        'physics': {
            'D': 0.2, 'rho': 0.0, 'sigma': 0.0, 'T1': 0.0,
            'L': [10.0, 10.0, 10.0], 'nodes': [8, 8, 8],
        },
        'initial_condition': {'type': 'point_source', 'amplitude': 5.0},
        'simulation': {'t_end': 0.5, 'n_frames': 3},
    }
    from solvers.integrator import run_stochastic_pde_solver

    model_stoch = SpectralNDE3D(cfg)
    sol_stoch = run_stochastic_pde_solver(model_stoch, cfg)

    model_det = SpectralNDE3D(cfg)
    sol_det = run_pde_solver(model_det, cfg)

    abs_err = np.max(np.abs(sol_stoch.y - sol_det.y))
    assert abs_err < 0.01, (
        f"Stochastic solver (T1=0) deviates too much from deterministic: "
        f"{abs_err:.4e} (>1%)"
    )


def test_mc3_gaussian_closure_consistency():
    r"""
    The first two moments of the third-order closure must match
    the Gaussian closure when the third cumulant is zero.

    With ⟨n³⟩ = 0 and identical state for ⟨N⟩ and ⟨n²⟩, the
    right-hand sides for the first two fields should be identical
    between the two model classes.
    """
    cfg = {
        'physics': {
            'D': 0.5,
            'rho': 0.1,
            'sigma': -0.1,
            'T1': 5.0e-4,
            'L': [20.0, 20.0, 20.0],
            'nodes': [31, 31, 31],
        },
        'initial_condition': {'type': 'gaussian', 'amplitude': 10.0,
                              'width': 3.0},
        'simulation': {'t_end': 1.0, 'n_frames': 2, 'method': 'RK45'},
    }

    model_gauss = MomentClosureNDE3D(cfg)
    model_third = ThirdOrderMomentClosureNDE3D(cfg)

    nb = model_gauss.n_state_block
    n0 = model_gauss.get_initial_condition()[:nb]

    # Build a state with non-uniform mean, uniform variance, and zero
    # third moment so the test is non-trivial.
    y_gauss = np.empty(2 * nb)
    y_gauss[:nb] = n0
    y_gauss[nb:] = 0.3

    y_third = np.empty(3 * nb)
    y_third[:nb] = n0
    y_third[nb:2*nb] = 0.3
    y_third[2*nb:] = 0.0

    rhs_gauss = model_gauss.compute_rhs(0.0, y_gauss)
    rhs_third = model_third.compute_rhs(0.0, y_third)

    assert np.allclose(rhs_third[:nb], rhs_gauss[:nb]), (
        "d⟨N⟩/dt mismatch between Gaussian and third-order closure"
    )
    assert np.allclose(rhs_third[nb:2*nb], rhs_gauss[nb:]), (
        "d⟨n²⟩/dt mismatch between Gaussian and third-order closure"
    )
