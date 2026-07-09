# -*- coding: utf-8 -*-
"""
Diagnostics-related tests.

These tests validate that ``utils.diagnostics.save_simulation_state`` writes the
expected artifacts with consistent shapes.
"""

from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np

from models.diffusion import SpectralNDE3D
from utils.diagnostics import save_simulation_state


def test_save_simulation_state_writes_expected_files(tmp_path):
    """save_simulation_state should create .npy/.json/.pkl outputs."""
    config = {
        "physics": {"D": 0.2, "rho": 0.1, "L": [10.0, 10.0, 10.0], "nodes": [5, 5, 5]},
        "initial_condition": {"type": "point_source", "amplitude": 3.0},
        "output": {"save_dir": str(tmp_path)},
    }

    model = SpectralNDE3D(config)
    n0_flat = model.get_initial_condition()

    # Fake a 2-frame "solution" without running solve_ivp.
    y = np.stack([n0_flat, 0.9 * n0_flat], axis=1)
    t = np.array([0.0, 1.0], dtype=float)
    solution = SimpleNamespace(y=y, t=t)

    save_simulation_state(model, solution, config)

    # Check files exist.
    field_path = tmp_path / "neutron_density_field.npy"
    time_path = tmp_path / "time_points.npy"
    meta_path = tmp_path / "metadata.json"
    session_path = tmp_path / "sim_session.pkl"

    assert field_path.exists()
    assert time_path.exists()
    assert meta_path.exists()
    assert session_path.exists()

    # Validate array shapes.
    field = np.load(field_path)
    assert field.shape == (*model.nodes, len(t))

    times = np.load(time_path)
    assert np.allclose(times, t)

    # Validate metadata content.
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["execution_status"] == "Success"
    assert meta["peak_density"] == float(np.max(y[:, -1]))

    final_mass_expected = float(np.sum(y[:, -1]) * model.dv)
    assert meta["final_mass"] == final_mass_expected

    # Validate pickle snapshot is loadable.
    with open(session_path, "rb") as f:
        loaded = pickle.load(f)
    assert "model" in loaded and "solution" in loaded

