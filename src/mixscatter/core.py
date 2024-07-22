"""
This module provides convenience functions for calculating the scattered intensity, measurable structure factors
and apparent diffusion coefficients for a given scattering model and liquid structure.

Classes:
    LiquidStructureLike:
        A protocol which defines an interface for a class which behaves like `LiquidStructure`.
    ScatteringModelLike:
        A protocol which defines an interface for a class which behaves like `ScatteringModel`.
    MixtureLike:
        A protocol which defines an interface for a class which behaves like `Mixture`.

Functions:
    measurable_intensity:
        Calculate the measured intensity for a given scattering model and liquid structure.
    measurable_structure_factor:
        Calculate the measurable structure factor for a given scattering model and liquid
        structure.
    measurable_diffusion_coefficient:
        Calculate the measured diffusion coefficient for a given scattering model and liquid
"""

import numpy as np

from typing import Protocol

from numpy.typing import NDArray

__all__ = [
    "measurable_structure_factor",
    "measurable_intensity",
    "measurable_diffusion_coefficient",
    "LiquidStructureLike",
    "ScatteringModelLike",
    "MixtureLike",
]


class MixtureLike(Protocol):  # pragma: no cover
    """
    A protocol which defines an interface for a class which behaves like Mixture.

    Any object which implements this protocol can be used by the functions in this module instead of a standard
    `Mixture` object. This can be exploited to construct custom `MixtureLike` objects with
    only the functionality strictly necessary.
    """

    @property
    def radius(self) -> NDArray[np.float64]: ...

    @property
    def number_fraction(self) -> NDArray[np.float64]: ...


class LiquidStructureLike(Protocol):  # pragma: no cover
    """
    A protocol which defines an interface for a class which behaves like LiquidStructure.

    Any object which implements this protocol can be used by the functions in this module instead of a standard
    `LiquidStructure` object. This can be exploited to construct custom `LiquidStructureLike` objects with
    only the functionality strictly necessary.
    """

    @property
    def number_weighted_partial_structure_factor(self) -> NDArray[np.float64]: ...


class ScatteringModelLike(Protocol):  # pragma: no cover
    """
    A protocol which defines an interface for a class which behaves like ScatteringModel.

    Any object which implements this protocol can be used by the functions in this module instead of a standard
    `ScatteringModel` object. This can be exploited to construct custom `ScatteringModelLike` objects with
    only the functionality strictly necessary.
    """

    mixture: MixtureLike

    @property
    def amplitude(self) -> NDArray[np.float64]: ...

    @property
    def average_square_amplitude(self) -> NDArray[np.float64]: ...

    @property
    def average_square_forward_amplitude(self) -> float: ...

    @property
    def average_form_factor(self) -> NDArray[np.float64]: ...


def measurable_intensity(
    liquid_structure: LiquidStructureLike,
    scattering_model: ScatteringModelLike,
    scale: float = 1.0,
    background: float = 0.0,
) -> NDArray[np.float64]:
    # noinspection PyShadowingNames
    """
    Calculate the measured intensity for a given scattering model and liquid structure.

    Args:
        liquid_structure:
            `LiquidStructure` instance or an object with an interface similar to `LiquidStructure`.
        scattering_model:
            `ScatteringModel` instance or an object with an interface similar to `ScatteringModel`.
        scale:
            Scale the intensity by a factor.
        background:
            Add a constant background.

    Returns:
        Measurable scattered intensity.

    Examples:
        >>> import numpy as np
        >>> from mixscatter import Mixture, PercusYevick, SimpleSphere, measurable_intensity

        >>> wavevector = np.linspace(0.005, 0.05, 100)
        >>> mixture = Mixture([100, 200], [0.2, 0.8])

        >>> liquid_structure = PercusYevick(wavevector, mixture, volume_fraction_total=0.3)
        >>> scattering_model = SimpleSphere(wavevector, mixture, contrast=1.0)

        >>> intensity = measurable_intensity(
        ...     liquid_structure, scattering_model, scale=1e5, background=1e3
        ... )
    """
    average_intensity: NDArray[np.float64] = np.einsum(
        "iq, jq, ijq->q",
        scattering_model.amplitude,
        scattering_model.amplitude,
        liquid_structure.number_weighted_partial_structure_factor,
    )
    normalized_intensity = scale * average_intensity / scattering_model.average_square_forward_amplitude + background
    return normalized_intensity


def measurable_structure_factor(
    liquid_structure: LiquidStructureLike, scattering_model: ScatteringModelLike
) -> NDArray[np.float64]:
    # noinspection PyShadowingNames
    """
    Calculate the measurable structure factor for a given scattering model and liquid structure.

    This is the quotient of the measurable intensity of a system with interactions and the
    equivalent intensity of a system without interactions.

    Args:
        liquid_structure:
            `LiquidStructure` instance or an object with an interface similar to `LiquidStructure`.
        scattering_model:
            `ScatteringModel` instance or an object with an interface similar to `ScatteringModel`.

    Returns:
        Measurable structure factor.

    Examples:
        >>> import numpy as np
        >>> from mixscatter import Mixture, PercusYevick, SimpleSphere, measurable_structure_factor

        >>> wavevector = np.linspace(0.005, 0.05, 100)
        >>> mixture = Mixture([100, 200], [0.2, 0.8])

        >>> liquid_structure = PercusYevick(wavevector, mixture, volume_fraction_total=0.3)
        >>> scattering_model = SimpleSphere(wavevector, mixture, contrast=1.0)

        >>> structure_factor = measurable_structure_factor(liquid_structure, scattering_model)
    """
    return measurable_intensity(liquid_structure, scattering_model) / scattering_model.average_form_factor


def measurable_diffusion_coefficient(
    scattering_model: ScatteringModelLike, thermal_energy: float, viscosity: float
) -> NDArray[np.float64]:
    # noinspection PyShadowingNames
    """
    Calculate the measurable Stokes-Einstein diffusion coefficient of a dilute suspension.

    Args:
        scattering_model:
            `ScatteringModel` instance or an object with an interface similar to `ScatteringModel`.
        thermal_energy:
            Thermal energy.
        viscosity:
            Viscosity of the surrounding medium.

    Returns:
        Measurable diffusion coefficient.

    Examples:
        >>> import numpy as np
        >>> from mixscatter import Mixture, SimpleSphere, measurable_diffusion_coefficient

        >>> wavevector = np.linspace(0.005, 0.05, 100)
        >>> mixture = Mixture([100, 200], [0.2, 0.8])
        >>> scattering_model = SimpleSphere(wavevector, mixture, contrast=1.0)

        >>> D_0 = measurable_diffusion_coefficient(
        ...     scattering_model, thermal_energy=1.0, viscosity=1.0
        ... )
    """
    weighted_inverse_radius: NDArray[np.float64] = np.sum(
        scattering_model.mixture.number_fraction[:, np.newaxis]
        * scattering_model.amplitude**2
        / scattering_model.mixture.radius[:, np.newaxis],
        axis=0,
        dtype=np.float64,
    )
    weighted_inverse_radius /= scattering_model.average_square_amplitude
    prefactor = thermal_energy / (6.0 * np.pi * viscosity)
    return prefactor * weighted_inverse_radius
