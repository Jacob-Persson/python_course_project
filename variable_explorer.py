# -*- coding: utf-8 -*-
"""
Load a saved simulation session into Spyder's Variable Explorer.

Usage:  python variable_explorer.py [save_dir]
Default save_dir: results/
"""
import pickle
import sys
from pathlib import Path

save_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
path = Path(save_dir) / "sim_session.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

model = data['model']
sol = data['solution']
print(f"Loaded model from {path}")
print(f"  D={model.D}, rho={model.rho}, sigma={model.sigma}, T1={model.T1}")
print(f"  nodes={model.nodes}, L={model.L}") 