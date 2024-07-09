# noinspection PyShadowingNames
"""
Calculate scattering functions of multicomponent mixtures of interacting spherical scatterers \
within the Born approximation.

Examples:
    >>> import numpy as np
    >>> import mixscatter as ms

    >>> particle_mixture = ms.mixture.Mixture(radius=[100, 250], number_fraction=[0.5, 0.5])

    >>> wavevector = np.linspace(1e-3, 6e-2, 4096)
    >>> scattering_model = ms.scatteringmodel.SimpleSphere(wavevector, particle_mixture, contrast=1.0)
    >>> structure_factor = ms.liquidstructure.PercusYevick(
    ... wavevector, particle_mixture, volume_fraction_total=0.3
    ... )

    >>> measurable_intensity = ms.measurable_intensity(structure_factor, scattering_model)
    >>> measurable_structure_factor = ms.measurable_structure_factor(structure_factor, scattering_model)
    >>> average_form_factor = scattering_model.average_form_factor
"""

from importlib.metadata import version, PackageNotFoundError

from mixscatter.core import (
    measurable_intensity,
    measurable_structure_factor,
    measurable_diffusion_coefficient,
    LiquidStructureLike,
    ScatteringModelLike,
    MixtureLike,
)
from mixscatter import liquidstructure, scatteringmodel, mixture
from mixscatter.mixture import (
    Mixture,
    FlorySchulzMixture,
    GaussianMixture,
    UniformMixture,
    SingleComponent,
)
from mixscatter.liquidstructure import LiquidStructure, PercusYevick, VerletWeis
from mixscatter.scatteringmodel import (
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
    "measurable_intensity",
    "measurable_structure_factor",
    "measurable_diffusion_coefficient",
    "LiquidStructureLike",
    "ScatteringModelLike",
    "MixtureLike",
    "FlorySchulzMixture",
    "GaussianMixture",
    "UniformMixture",
    "SingleComponent",
    "Mixture",
    "LiquidStructure",
    "PercusYevick",
    "VerletWeis",
    "ScatteringModel",
    "SimpleSphere",
    "SimpleCoreShell",
    "SimpleGradient",
    "EmptyProfile",
    "ConstantProfile",
    "LinearProfile",
    "Particle",
    "ParticleBuilder",
    "liquidstructure",
    "scatteringmodel",
    "mixture",
]

try:
    __version__ = version("mixscatter")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass
