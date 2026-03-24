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
    def compute_rhs(self, t, N):
        """Calculate the time derivative dN/dt."""
        pass

    @abstractmethod
    def get_initial_condition(self):
        """Must return the starting N distribution."""
        pass