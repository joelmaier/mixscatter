"""Core module of mixscatter."""

import numpy as np

from typing import Any, Protocol

from numpy.typing import NDArray

__all__ = ["measurable_structure_factor", "measurable_intensity", "measurable_diffusion_coefficient"]


# noinspection PyPropertyDefinition


class MixtureLike(Protocol):  # pragma: no cover
    @property
    def radius(self) -> NDArray[np.float64]: ...

    @property
    def number_fraction(self) -> NDArray[np.float64]: ...


# noinspection PyPropertyDefinition
class LiquidStructureLike(Protocol):  # pragma: no cover
    """A protocol which defines an interface for a class which behaves like LiquidStructure"""

    @property
    def number_weighted_partial_structure_factor(self) -> NDArray[np.float64]: ...


# noinspection PyPropertyDefinition
class ScatteringModelLike(Protocol):  # pragma: no cover
    """A protocol which defines an interface for a class which behaves like ScatteringModel"""

    mixture: MixtureLike

    @property
    def amplitude(self) -> NDArray[np.float64]: ...

    @property
    def average_square_amplitude(self) -> Any: ...

    @property
    def average_square_forward_amplitude(self) -> Any: ...

    @property
    def average_form_factor(self) -> Any: ...


def measurable_intensity(
    liquid_structure: LiquidStructureLike,
    scattering_model: ScatteringModelLike,
    scale: float = 1.0,
    background: float = 0.0,
) -> Any:
    # noinspection PyShadowingNames
    """
    The measurable scattered intensity.

    Args:
        liquid_structure:
            Object instance with an interface similar to a LiquidStructure object.
        scattering_model:
            Object instance with an interface similar to a ScatteringModel object.
        scale:
            Scale the intensity by a factor.
        background:
            Add a constant background.

    Returns:
        Measurable scattered intensity.

    Examples:
        >>> import mixscatter as ms
        >>> import numpy as np

        >>> wavevector = np.linspace(0.005, 0.05, 100)
        >>> mixture = ms.mixture.Mixture([100, 200], [0.2, 0.8])

        >>> liquid_structure = ms.liquidstructure.PercusYevick(wavevector, mixture, volume_fraction_total=0.3)
        >>> scattering_model = ms.scatteringmodel.SimpleSphere(wavevector, mixture, contrast=1.0)

        >>> intensity = measurable_intensity(liquid_structure, scattering_model, scale=1e5,
        ...                                  background=1e3)
    """
    average_intensity = np.einsum(
        "iq, jq, ijq->q",
        scattering_model.amplitude,
        scattering_model.amplitude,
        liquid_structure.number_weighted_partial_structure_factor,
    )
    normalized_intensity = average_intensity / scattering_model.average_square_forward_amplitude
    return scale * normalized_intensity + background


def measurable_structure_factor(liquid_structure: LiquidStructureLike, scattering_model: ScatteringModelLike) -> Any:
    # noinspection PyShadowingNames
    """The measurable structure factor.

    This is the quotient of the measurable intensity of a system with interactions and the
    equivalent intensity of a system without interactions.

    Args:
        liquid_structure:
            Object instance with an interface similar to a LiquidStructure object.
        scattering_model:
            Object instance with an interface similar to a ScatteringModel object.

    Returns:
        Measurable structure factor.

    Examples:
        >>> import mixscatter as ms
        >>> import numpy as np

        >>> wavevector = np.linspace(0.005, 0.05, 100)
        >>> mixture = ms.mixture.Mixture([100, 200], [0.2, 0.8])

        >>> liquid_structure = ms.liquidstructure.PercusYevick(wavevector, mixture, volume_fraction_total=0.3)
        >>> scattering_model = ms.scatteringmodel.SimpleSphere(wavevector, mixture, contrast=1.0)

        >>> structure_factor = measurable_structure_factor(liquid_structure, scattering_model)
    """
    return measurable_intensity(liquid_structure, scattering_model) / scattering_model.average_form_factor


def measurable_diffusion_coefficient(
    scattering_model: ScatteringModelLike, thermal_energy: float, viscosity: float
) -> Any:
    # noinspection PyShadowingNames
    """
    The measurable Stokes-Einstein diffusion coefficient of a dilute suspension.

    Args:
        scattering_model:
            Object instance with an interface similar to a ScatteringModel object.
        thermal_energy:
            Thermal energy.
        viscosity:
            Viscosity of the surrounding medium.

    Returns:
        Measurable diffusion coefficient.

    Examples:
        >>> import mixscatter as ms
        >>> import numpy as np

        >>> wavevector = np.linspace(0.005, 0.05, 100)
        >>> mixture = ms.mixture.Mixture([100, 200], [0.2, 0.8])

        >>> scattering_model = ms.scatteringmodel.SimpleSphere(wavevector, mixture, contrast=1.0)
        >>> D_0 = measurable_diffusion_coefficient(
        ...         scattering_model, thermal_energy=1.0, viscosity=1.0)
    """
    weighted_inverse_radius = np.sum(
        scattering_model.mixture.number_fraction[:, np.newaxis]
        * scattering_model.amplitude**2
        / scattering_model.mixture.radius[:, np.newaxis],
        axis=0,
    )
    weighted_inverse_radius /= scattering_model.average_square_amplitude
    prefactor = thermal_energy / (6.0 * np.pi * viscosity)
    return prefactor * weighted_inverse_radius
