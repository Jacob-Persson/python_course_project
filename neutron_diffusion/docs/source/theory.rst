Theory of Neutron Diffusion
===========================

The core of this project is the linear Neutron Diffusion Equation (NDE) in 3D:

.. math::

   \frac{\partial N}{\partial t} = D \nabla^2 N + \rho N

Where:
* :math:`D` is the diffusion coefficient.
* :math:`\rho` is the reactivity (material growth/decay).

Numerical Implementation
------------------------
To achieve high precision, we use a **Spectral Method**. By taking the 
Fourier Transform of both sides, the Laplacian becomes a simple multiplication 
by :math:`-k^2`:

.. math::

   \mathcal{F}\{\nabla^2 N\} = -|k|^2 \hat{N}
