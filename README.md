# PyTorch Implementation for Structure-preserving GPs. This is part of the publication "Structure-preserving Gaussian Process dynamics" (ECML 2022).

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication "Structure-preserving Gaussian Process dynamics" (ECML
2022). It will neither be maintained nor monitored in any way.


## Installation

Packages are listed in requirements.txt 


## Experiments

To perform experiments, the following scripts are available:

- pendulum: pend_SGPD.py produces results for symplectic Euler-based SGPD
- nonseparable Hamiltonian nonseparable_SGPD.py: produces results for midpoint-based SGPD method

## Components 

### Integrators 

- explicit_integrator.py: includes standard implicit integrators as Euler and Heun-method. 
- nonsep_midpoint.py: includes the Hamiltonian implicit midpoint method as used for the nonseparable Hamiltonian system.
- RB_midpoint.py: standard implicit midpoint method. 
- symplectic_integrator.py: includes explicit symplectic integrators as the explicit Euler method. 
- The custom backpropagation for the implicit integrators is contained in integrator/Implicit. 

### Loss

- The log-probability is contained in Noise/noise_nd.py.

### Models

- sparse_GP.matherons_rule contains a 
  - sparse GP implementation: MatheronGP
  - an implementation for the derivatives of a GP: MatheronDerivative
  - a GP with quadratic contrains on the dynamics: ODE
  - Function to stack independent GPs: StackedGP  

## LICENSE

structure-preserving Gaussian processes is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE) file for details.
