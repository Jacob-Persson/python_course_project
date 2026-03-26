# python\_course\_project

Project description: Solve the neutron diffusion equation and plot the results.

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run the simulation
From the repository root, run:

```bash
cd neutron_diffusion
python main.py
```

This reads `config.yaml`, solves the PDE, displays plots, and (optionally) saves figures and the full simulation state depending on `output.*` in the config.

## Run tests
From the repository root:

```bash
python -m pytest -q
```

## Build documentation (optional)
From `neutron_diffusion/docs/`:

```bash
python -m sphinx.cmd.build -b html source build
```



neutron\_diffusion/

├── docs/                  # Sphinx documentation source and build files

│   ├── source/            # .rst files and conf.py

│   └── build/             # Generated HTML files

├── models/                # Physics engines (Linear, Spectral, etc.)

├── solvers/               # Numerical integrators (solve\_ivp)

├── tests/                 # Pytest suite for physics verification

├── utils/                 # Plotting and diagnostics

├── config.yaml            # Simulation parameters

└── main.py                # Project entry point

