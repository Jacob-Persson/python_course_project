# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:51:47 2026

@author: jacpe396
"""

from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def compute_rhs(self, t, N):
        """Must return the derivative dN/dt."""
        pass

    @abstractmethod
    def get_initial_condition(self):
        """Must return the starting N distribution."""
        pass