"""Calculate form factors of multicomponent systems.

This module provides classes and functions for calculating form factors of multicomponent systems.
It includes definitions for various layer profiles, particles composed of these layers, and models
to calculate scattering properties for these particles.

Classes:
    LayerProfile: Defines the interface for layer profiles.
    EmptyProfile: Represents an empty layer profile.
    ConstantProfile: Represents a layer profile with constant contrast.
    LinearProfile: Represents a layer profile with linearly varying contrast.
    Particle: Represents a particle composed of multiple layers.
    ParticleBuilder: A builder class for constructing Particle instances.
    ScatteringModel: Calculates scattering properties for a list of particles.
    SimpleSphere: A convenience class for creating a scattering model of homogeneously scattering spheres.
    SimpleCoreShell: A convenience class for creating a scattering model of core-shell particles.
    SimpleGradient: A convenience class for creating a scattering model with gradient profiles.

Examples:
    Create a simple scattering model for spheres:

    >>> from mixscatter.mixture import Mixture
    >>> import numpy as np
    >>> wavevector = np.linspace(0.01, 1.0, 100)
    >>> mixture = Mixture(number_fraction=[0.5, 0.5], radius=[1.0, 2.0])
    >>> model = SimpleSphere(wavevector, mixture, contrast=1.0)
    >>> form_factor = model.average_form_factor

    Create a core-shell particle model:

    >>> model = SimpleCoreShell(wavevector, mixture, core_to_total_ratio=0.5, core_contrast=1.0, shell_contrast=0.5)
    >>> form_factor = model.average_form_factor

    Create a gradient profile particle model:

    >>> model = SimpleGradient(wavevector, mixture, center_contrast=1.0, boundary_contrast=0.5)
    >>> form_factor = model.average_form_factor
"""

from .scatteringmodel import (
    LayerProfile,
    EmptyProfile,
    ConstantProfile,
    LinearProfile,
    Particle,
    ParticleBuilder,
    ScatteringModel,
    SimpleSphere,
    SimpleCoreShell,
    SimpleGradient,
)

__all__ = [
    "LayerProfile",
    "EmptyProfile",
    "ConstantProfile",
    "LinearProfile",
    "Particle",
    "ParticleBuilder",
    "ScatteringModel",
    "SimpleSphere",
    "SimpleCoreShell",
    "SimpleGradient",
]
