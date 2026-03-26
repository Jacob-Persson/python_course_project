# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:51:47 2026

@author: jacpe396
"""

from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Interface for Neutron Diffusion Equation models.

    All physics models (e.g., Linear, Spectral, MSR) must inherit from 
    this class to ensure compatibility with the system solvers.
    """
    @abstractmethod
    def compute_rhs(self, t: float, N: "np.ndarray") -> "np.ndarray":
        r"""
        Compute the right-hand side (RHS) for the ODE system induced by the PDE.

        In the general setting, the PDE evolves a field ``N(x, t)``. After
        discretizing space and flattening the field into a 1D state vector
        ``y(t)``, the solver integrates:

        .. math::
            \frac{d}{dt} y(t) = f(t, y(t)).

        Implementations must return a flattened array with the same shape as
        the input ``N`` (i.e., matching the ordering used by the model).

        Args:
            t: Current time (s).
            N: Flattened spatial field at time ``t``.

        Returns:
            Flattened time derivative ``dN/dt`` with the same shape as ``N``.
        """
        pass

    @abstractmethod
    def get_initial_condition(self):
        r"""
        Construct the initial condition ``N(x, t=0)`` on the model's spatial grid.

        Implementations must return a flattened 1D array corresponding to the
        model's ``nodes``.
        """
        pass
