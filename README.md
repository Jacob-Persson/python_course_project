# python\_course\_project

Project description: Solve the neutron diffusion equation and plot the results.



neutronic\_field\_theory/

├── \_\_init\_\_.py

├── config.yaml            # External parameters (D, rho, grid size, etc.)

├── main.py                # Entry point: loads config, runs solver, and plots

├── models/

│   ├── \_\_init\_\_.py

│   ├── base\_model.py      # Abstract base class defining the interface

│   └── diffusion.py       # Linear NDE implementation (Standard Laplacian)

├── solvers/

│   ├── \_\_init\_\_.py

│   └── integrator.py      # Numerical methods (solve\_ivp, Euler-Maruyama)

├── tests/

│   ├── \_\_init\_\_.py

│   ├── test\_models.py     # Tests for ICs and derivative

│   └── test\_physics.py    # Tests for conservation laws

└── utils/

&#x20;   ├── \_\_init\_\_.py

&#x20;   └── plotting.py        # Dedicated functions for Matplotlib/Visualization





