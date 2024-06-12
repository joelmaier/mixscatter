"""
This module provides functionalities to represent and create mixtures of spherical particles
with various size distributions. It includes classes and functions to handle different
distributions like single-component, Flory-Schulz, Gaussian, and uniform distributions.

Classes:
    Mixture: Base class representing an N-component mixture of spherical particles.
    FlorySchulzMixture: Represents a Schulz-Flory distributed mixture.
    GaussianMixture: Represents a Gaussian distributed mixture.
    UniformMixture: Represents a uniformly distributed mixture.
    SingleComponent: Represents a pseudo one-component system.
"""

from .mixture import Mixture, FlorySchulzMixture, GaussianMixture, UniformMixture, SingleComponent

__all__ = ["Mixture", "FlorySchulzMixture", "GaussianMixture", "UniformMixture", "SingleComponent"]
