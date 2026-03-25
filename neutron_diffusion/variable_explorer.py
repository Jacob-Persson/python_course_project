# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:18:37 2026

@author: jacpe396
"""

import pickle
with open("results/sim_session.pkl", "rb") as f:
    data = pickle.load(f)

# Now you have full access to your variables in the Spyder Explorer!
model = data['model']
sol = data['solution']
print(model.D) 