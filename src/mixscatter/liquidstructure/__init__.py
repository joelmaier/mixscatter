"""
This module provides classes for representing the liquid structure properties of
interacting multicomponent systems of spherical particles.

Classes:
    LiquidStructure: Abstract base class representing liquid structure properties.
    PercusYevick: Class for hard-sphere potential in the Percus-Yevick approximation.
    VerletWeis: Class for hard-sphere potential in the Percus-Yevick approximation
        with Verlet-Weiss correction.

References:
    - P. Salgi, R. Rajagopalan, "Polydispersity in colloids: implications to static structure and scattering",
      Adv. Colloid Interface Sci. 43, 169-288 (1993), https://doi.org/10.1016/0001-8686(93)80017-6
    - A. Vrij, "Mixtures of hard spheres in the Percus–Yevick approximation. Light scattering at finite angles",
      J. Chem. Phys. 71, 3267-3270 (1979), https://doi.org/10.1063/1.438756
    - J. Diaz Maier, K. Gaus, J. Wagner, "Measurable structure factors of dense dispersions containing polydisperse,
      optically inhomogeneous particles", arXiv:2404.03470 [cond-mat.soft], https://doi.org/10.48550/arXiv.2404.03470
"""

from .liquidstructure import LiquidStructure, PercusYevick, VerletWeis

__all__ = ["LiquidStructure", "PercusYevick", "VerletWeis"]
