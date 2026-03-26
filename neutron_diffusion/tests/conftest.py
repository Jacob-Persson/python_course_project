"""
Pytest configuration.

This project currently imports subpackages as top-level modules, e.g.:

    - ``from models.diffusion import SpectralNDE3D``
    - ``from solvers.integrator import run_pde_solver``

When running ``pytest`` from the repository root, those imports require the
``neutron_diffusion`` directory to be on ``sys.path``. This conftest ensures
that automatically for all tests.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Add ``.../python_course_project/neutron_diffusion`` to sys.path so that
# imports like ``import models`` resolve correctly.
_NEUTRON_DIFFUSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NEUTRON_DIFFUSION_ROOT))

