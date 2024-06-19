---
hide:
  - navigation
  - toc
---
# Welcome to mixscatter

**mixscatter** is a versatile tool for calculating scattering 
functions of particle mixtures, particularly for [small-angle scattering (SAS)](
https://en.wikipedia.org/wiki/Small-angle_scattering)
and [dynamic light scattering (DLS)](
https://en.wikipedia.org/wiki/Dynamic_light_scattering) applications.

## Key Features
* Scattering Functions: Calculate scattering amplitudes, measurable intensities, form factors,  
  structure factors and diffusion coefficients for multi-component mixtures.
* Flexible Mixture Composition: Define particle mixtures with arbitrary compositions and complex 
  scattering length density profiles.
* Advanced Models: Use predefined scattering models or build entirely custom particles with the 
  powerful `ParticleBuilder`.

## Background
Understanding the principles behind scattering techniques is crucial for interpreting experimental 
data accurately. **mixscatter** provides tools to analyze systems of interacting spherical 
scatterers in the Born approximation
([Rayleigh-Gans-Debye scattering](https://en.wikipedia.org/wiki/Rayleigh-Gans_approximation)).
Very basic information and a couple of equations for the interested reader are given in the 
[Background](background.md) page of this documentation.

**Here are some useful references:**

  - P. Salgi and R. Rajagopalan, "Polydispersity in colloids: implications to static structure and
   scattering", [Adv. Colloid Interface Sci. 43, 169-288 (1993)](
   https://doi.org/10.1016/0001-8686(93)80017-6)
  - A. Vrij, "Mixtures of hard spheres in the Percus–Yevick approximation. Light scattering at
    finite angles",  [J. Chem. Phys. 71, 3267-3270 (1979)](https://doi.org/10.1063/1.438756)
  - R. Botet, S. Kwok and B. Cabane, "Percus-Yevick structure factors made simple",
    [J. Appl. Cryst. 53, 1570-1582 (2020)](https://doi.org/10.1107/S1600576720014041)
  - J. Diaz Maier, K. Gaus and J. Wagner, "Measurable structure factors of dense dispersions
    containing polydisperse, optically inhomogeneous particles",
    [arXiv:2404.03470 [cond-mat.soft]](https://doi.org/10.48550/arXiv.2404.03470)

## Installation

**mixscatter** is available at the [Python Package Index (PyPI)](
https://pypi.org/project/mixscatter). You can install the package via `pip`:
```shell
pip install mixscatter
```
The source code is currently hosted on GitHub at: <https://github.com/joelmaier/mixscatter>

## Getting Started

For detailed instructions on how to use **mixscatter**, refer to the 
[Getting Started Guide](getting_started/getting-started.md).

## API Reference
Explore detailed information on all functions, classes, and modules in the 
[API Reference](api/core_api.md).
