# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:58:39 2026

@author: jacpe396
"""

import matplotlib.pyplot as plt

def plot_results(model, solution):
    plt.figure(figsize=(8, 5))
    for i, t in enumerate(solution.t):
        plt.plot(model.x, solution.y[:, i], label=f"t={t:.1f}s")
    
    plt.title("Neutron Density Evolution")
    plt.xlabel("Position")
    plt.ylabel("Density N")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()