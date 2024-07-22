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

    >>> from mixscatter import Mixture
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

from functools import cached_property
from typing import Protocol

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray


# noinspection PyPropertyDefinition
class MixtureLike(Protocol):  # pragma: no cover
    @property
    def number_fraction(self) -> NDArray[np.float64]: ...

    @property
    def radius(self) -> NDArray[np.float64]: ...


class LayerProfile(Protocol):  # pragma: no cover
    """Defines the interface for layer profiles."""

    radius_inner: float
    radius_outer: float

    def calculate_amplitude(self, wavevector: ArrayLike) -> NDArray[np.float64]:
        """Calculate the amplitude for the given wavevector."""
        ...

    def calculate_forward_amplitude(self) -> float:
        """Calculate the forward amplitude."""
        ...

    def get_profile(self, distance: ArrayLike) -> NDArray[np.float64]:
        """Get the profile for the given distance from the origin."""
        ...

    def calculate_second_moment(self) -> float:
        """Calculate the second moment."""
        ...


class EmptyProfile(LayerProfile):
    """
    Represents an empty layer profile.

    Attributes:
        radius_inner (float): Inner radius of the empty layer.
        radius_outer (float): Outer radius of the empty layer.

    Methods:
        __init__(radius_inner, radius_outer):
            Initialize an empty layer profile.

        calculate_amplitude(wavevector):
            Calculate the amplitude for the given wavevector.

        calculate_forward_amplitude():
            Calculate the forward amplitude.

        get_profile(distance):
            Get the profile for the given distance from the origin.

        calculate_second_moment():
            Calculate the second moment.
    """

    def __init__(self, radius_inner: float, radius_outer: float) -> None:
        """
        Initialize an empty layer profile.

        Args:
            radius_inner: Inner radius of the empty layer.
            radius_outer: Outer radius of the empty layer.

        Raises:
            RuntimeError: If `radius_inner` is greater than `radius_outer`.
        """
        if radius_inner > radius_outer:
            raise RuntimeError("'radius_inner' must be smaller than 'radius_outer'.")

        self.radius_inner = radius_inner
        self.radius_outer = radius_outer

    def calculate_amplitude(self, wavevector: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate the amplitude for the given wavevector.

        Args:
            wavevector: Scattering wavevector.

        Returns:
            Zero array of the same shape as wavevector.
        """
        wavevector = np.asarray(wavevector, dtype=np.float64)
        return np.zeros_like(wavevector)

    def calculate_forward_amplitude(self) -> float:
        """Calculate the forward amplitude.

        Returns:
            Zero.
        """
        return 0.0

    def get_profile(self, distance: ArrayLike) -> NDArray[np.float64]:
        """
        Get the profile for the given distance from the origin.

        Args:
            distance: Distance from the origin.

        Returns:
            Zero array of the same shape as distance.
        """
        distance = np.asarray(distance, dtype=np.float64)
        return np.zeros_like(distance)

    def calculate_second_moment(self) -> float:
        """Calculate the second moment.

        Returns:
            Zero.
        """
        return 0.0


class ConstantProfile(LayerProfile):
    """
    Represents a layer profile with constant contrast.

    Attributes:
        radius_inner (float): Inner radius of the layer.
        radius_outer (float): Outer radius of the layer.
        contrast (float): Scattering contrast of the layer.

    Methods:
        __init__(radius_inner, radius_outer, contrast):
            Initialize a constant layer profile.

        calculate_amplitude(wavevector):
            Calculate the amplitude for the given wavevector.

        calculate_forward_amplitude():
            Calculate the forward amplitude.

        get_profile(distance):
            Get the profile for the given distance from the origin.

        calculate_second_moment():
            Calculate the second moment.
    """

    def __init__(self, radius_inner: float, radius_outer: float, contrast: float) -> None:
        """
        Initialize a constant layer profile.

        Args:
            radius_inner: Inner radius of the layer.
            radius_outer: Outer radius of the layer.
            contrast: Scattering contrast of the layer.

        Raises:
            RuntimeError: If `radius_inner` is greater than `radius_outer`.
        """
        if radius_inner > radius_outer:
            raise RuntimeError("'radius_inner' must be smaller than 'radius_outer'.")

        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        self.contrast = contrast

    def calculate_amplitude(self, wavevector: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate the amplitude for the given wavevector.

        Args:
            wavevector: Scattering wavevector.

        Returns:
            Calculated amplitude array.
        """
        wavevector = np.asarray(wavevector, dtype=np.float64)
        QR_outer = wavevector * self.radius_outer
        QR_inner = wavevector * self.radius_inner
        amplitude: NDArray[np.float64] = np.sin(QR_outer) - QR_outer * np.cos(QR_outer)
        amplitude -= np.sin(QR_inner) - QR_inner * np.cos(QR_inner)
        amplitude *= 4.0 * np.pi / wavevector**3 * self.contrast
        return amplitude

    def calculate_forward_amplitude(self) -> float:
        """Calculate the forward amplitude.

        Returns:
            The calculated forward amplitude.
        """
        return 4.0 / 3.0 * np.pi * (self.radius_outer**3 - self.radius_inner**3) * self.contrast

    def get_profile(self, distance: ArrayLike) -> NDArray[np.float64]:
        """Get the profile for the given distance from the origin.

        Args:
            distance: Distance from the origin.

        Returns:
            The profile evaluated on the distance array.
        """
        distance = np.asarray(distance, dtype=np.float64)
        distance_mask = (distance >= self.radius_inner) & (distance < self.radius_outer)
        return np.where(distance_mask, self.contrast, 0.0)

    def calculate_second_moment(self) -> float:
        """Calculate the second moment.

        Returns:
            The calculated second moment.
        """
        return 4.0 / 5.0 * np.pi * (self.radius_outer**5 - self.radius_inner**5) * self.contrast


class LinearProfile(LayerProfile):
    """
    Represents a layer profile with linearly varying contrast.

    Attributes:
        radius_inner (float): Inner radius of the layer.
        radius_outer (float): Outer radius of the layer.
        contrast_inner (float): Contrast at the inner radius.
        contrast_outer (float): Contrast at the outer radius.

    Methods:
        __init__(radius_inner, radius_outer, contrast_inner, contrast_outer):
            Initialize a linearly varying layer profile.

        calculate_amplitude(wavevector):
            Calculate the amplitude for the given wavevector.

        calculate_forward_amplitude():
            Calculate the forward amplitude.

        get_profile(distance):
            Get the profile for the given distance from the origin.

        calculate_second_moment():
            Calculate the second moment.
    """

    def __init__(self, radius_inner: float, radius_outer: float, contrast_inner: float, contrast_outer: float) -> None:
        """
        Initialize a linearly varying layer profile.

        Args:
            radius_inner (float): Inner radius of the layer.
            radius_outer (float): Outer radius of the layer.
            contrast_inner (float): Contrast at the inner radius.
            contrast_outer (float): Contrast at the outer radius.

        Raises:
            RuntimeError: If radius_inner is greater than radius_outer.
        """
        if radius_inner > radius_outer:
            raise RuntimeError("'radius_inner' must be smaller than 'radius_outer'.")

        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        self.contrast_inner = contrast_inner
        self.contrast_outer = contrast_outer

    def calculate_amplitude(self, wavevector: ArrayLike) -> NDArray[np.float64]:
        """Calculate the amplitude for the given wavevector.

        Args:
            wavevector: Scattering wavevector.

        Returns:
            The calculated amplitude array.
        """
        wavevector = np.asarray(wavevector, dtype=np.float64)
        QR_outer = wavevector * self.radius_outer
        QR_inner = wavevector * self.radius_inner
        intercept, slope = self._two_point_to_slope_intercept(
            self.radius_inner, self.contrast_inner, self.radius_outer, self.contrast_outer
        )
        amplitude_intercept = ConstantProfile(self.radius_inner, self.radius_outer, intercept).calculate_amplitude(
            wavevector
        )
        amplitude_gradient: NDArray[np.float64] = (2.0 - QR_outer**2) * np.cos(QR_outer) + 2.0 * QR_outer * np.sin(
            QR_outer
        )
        amplitude_gradient -= (2.0 - QR_inner**2) * np.cos(QR_inner) + 2.0 * QR_inner * np.sin(QR_inner)
        amplitude_gradient *= 4.0 * np.pi / wavevector**4 * slope
        return amplitude_intercept + amplitude_gradient

    def calculate_forward_amplitude(self) -> float:
        """Calculate the forward amplitude.

        Returns:
            The calculated forward amplitude.
        """
        intercept, slope = self._two_point_to_slope_intercept(
            self.radius_inner, self.contrast_inner, self.radius_outer, self.contrast_outer
        )
        forward_amplitude_intercept = ConstantProfile(
            self.radius_inner, self.radius_outer, intercept
        ).calculate_forward_amplitude()
        forward_amplitude_gradient = np.pi * (self.radius_outer**4 - self.radius_inner**4) * slope
        return forward_amplitude_intercept + forward_amplitude_gradient

    @staticmethod
    def _two_point_to_slope_intercept(x_1: float, y_1: float, x_2: float, y_2: float) -> tuple[float, float]:
        """Calculate the slope and intercept for a linear function passing through two points.

        Returns:
            A tuple containing the intercept and slope.
        """
        intercept = (x_2 * y_1 - x_1 * y_2) / (x_2 - x_1)
        slope = (y_2 - y_1) / (x_2 - x_1)
        return intercept, slope

    def get_profile(self, distance: ArrayLike) -> NDArray[np.float64]:
        """Get the profile for the given distance from the origin.

        Args:
            distance: Distance from the origin.

        Returns:
            The profile evaluated on the distance array.
        """
        distance = np.asarray(distance, dtype=np.float64)
        distance_mask = (distance >= self.radius_inner) & (distance < self.radius_outer)
        intercept, slope = self._two_point_to_slope_intercept(
            self.radius_inner, self.contrast_inner, self.radius_outer, self.contrast_outer
        )
        profile = np.zeros_like(distance)
        profile[distance_mask] = intercept + slope * distance[distance_mask]
        return profile

    def calculate_second_moment(self) -> float:
        """Calculate the second moment.

        Returns:
            The calculated second moment.
        """
        intercept, slope = self._two_point_to_slope_intercept(
            self.radius_inner, self.contrast_inner, self.radius_outer, self.contrast_outer
        )
        moment_intercept = ConstantProfile(self.radius_inner, self.radius_outer, intercept).calculate_second_moment()
        moment_slope = 2.0 / 3.0 * np.pi * (self.radius_outer**6 - self.radius_inner**6) * slope
        return moment_intercept + moment_slope


class Particle:
    """
    Represents a particle composed of multiple layers.

    Attributes:
        layers (list): List of `LayerProfile` instances representing particle layers.

    Methods:
        __init__(layers):
            Initialize a particle with given layers.

        calculate_amplitude(wavevector):
            Calculate the amplitude for the given wavevector.

        calculate_forward_amplitude():
            Calculate the forward amplitude.

        calculate_form_factor(wavevector):
            Calculate the form factor for the given wavevector.

        get_profile(distance):
            Get the profile for the given distance from the origin.

        calculate_square_radius_of_gyration():
            Calculate the squared radius of the gyration.
    """

    def __init__(self, layers: list[LayerProfile]) -> None:
        """
        Initialize a particle with given layers.

        Args:
            layers: List of `LayerProfile` instances representing particle layers.
        """
        self.layers = layers

    def calculate_amplitude(self, wavevector: ArrayLike) -> NDArray[np.float64]:
        """Calculate the amplitude for the given wavevector.

        Args:
            wavevector: Scattering wavevector.

        Returns:
            The calculated amplitude array.
        """
        wavevector = np.asarray(wavevector, dtype=np.float64)
        amplitude = np.zeros_like(wavevector)
        for layer in self.layers:
            amplitude += layer.calculate_amplitude(wavevector)
        return amplitude

    def calculate_forward_amplitude(self) -> float:
        """Calculate the forward amplitude.

        Returns:
            The calculated forward amplitude.
        """
        forward_amplitude = 0.0
        for layer in self.layers:
            forward_amplitude += layer.calculate_forward_amplitude()
        return forward_amplitude

    def calculate_form_factor(self, wavevector: ArrayLike) -> NDArray[np.float64]:
        """Calculate the form factor for the given wavevector.

        Args:
            wavevector: Scattering wavevector.

        Returns:
            The calculated form factor array.
        """
        wavevector = np.asarray(wavevector, dtype=np.float64)
        amplitude = self.calculate_amplitude(wavevector)
        forward_amplitude = self.calculate_forward_amplitude()
        return (amplitude / forward_amplitude) ** 2.0

    def get_profile(self, distance: ArrayLike) -> NDArray[np.float64]:
        """Get the profile for the given distance from the origin.

        Args:
            distance: Distance from the origin.

        Returns:
            The profile evaluated on the distance array.
        """
        distance = np.asarray(distance, dtype=np.float64)
        profile = np.zeros_like(distance)
        for layer in self.layers:
            profile += layer.get_profile(distance)
        return profile

    def calculate_square_radius_of_gyration(self) -> float:
        """Calculate the squared radius of the gyration.
        Returns:
            The calculated squared radius of the gyration.
        """
        total_second_moment = 0.0
        for layer in self.layers:
            total_second_moment += layer.calculate_second_moment()
        square_radius_of_gyration = total_second_moment / self.calculate_forward_amplitude()
        return square_radius_of_gyration


class ParticleBuilder:
    """
    A builder class for constructing `Particle` instances.

    This class facilitates the step-by-step construction of `Particle` objects composed of multiple
    layers, where each layer is represented by a `LayerProfile` instance. The builder ensures that layers
    are added sequentially, with checks for connectivity between layers and overlap prevention.

    Attributes:
        layers (list): List to store `LayerProfile` instances representing the layers of the particle.

    Methods:
        __init__():
            Initializes an empty `ParticleBuilder` instance.

        add_layer(layer):
            Adds a layer to the particle being built. Ensures the new layer connects correctly to the
            previous layer and does not overlap with existing layers.

        reset():
            Resets the builder instance, clearing all layers stored in the layers list.

        get_particle():
            Constructs and returns a `Particle` instance using the layers added so far.

        pop_particle():
            Constructs and returns a `Particle` instance using the layers added so far, then resets the
            builder instance to start building a new particle.

    Usage:
        The typical usage involves creating a `ParticleBuilder` instance, adding layers using add_layer(),
        and finally retrieving the constructed particle using get_particle() or pop_particle().

    Example:
        Construct a particle using `ParticleBuilder`:

            >>> from mixscatter.scatteringmodel import ParticleBuilder, ConstantProfile
            >>> layer1 = ConstantProfile(radius_inner=0.0, radius_outer=2.0, contrast=1.0)
            >>> layer2 = ConstantProfile(radius_inner=2.0, radius_outer=5.0, contrast=0.5)
            >>> builder = ParticleBuilder()
            >>> particle = builder.add_layer(layer1).add_layer(layer2).get_particle()

    Notes:
        - The add_layer() method ensures that each added layer starts where the previous layer ends,
          preventing gaps or overlaps between layers.
        - The reset() method allows reusing the same builder instance to construct multiple particles.
        - The pop_particle() method is useful when constructing and retrieving particles in a single step,
          resetting the builder for the next particle construction.

    Raises:
        RuntimeError:
            Raised by add_layer() method if the added layer does not connect correctly to the
            previous layer or if there is an overlap with existing layers.
    """

    def __init__(self) -> None:
        """Initializes an empty `ParticleBuilder` instance."""
        self.layers: list[LayerProfile] = []

    def add_layer(self, layer: LayerProfile) -> Self:
        """Add a layer to the particle being built.

        Args:
            layer: `LayerProfile` instance to be added.

        Returns:
            The builder instance.

        Raises:
            RuntimeError: If the added layer does not connect to the previous layer or if layers overlap.
        """
        if self.layers:
            do_layers_connect = np.allclose(self.layers[-1].radius_outer, layer.radius_inner)
            if not do_layers_connect:
                raise RuntimeError(
                    f"Added layer with inner radius {layer.radius_inner} does not connect to previous layer's outer "
                    f"radius {self.layers[-1].radius_outer}."
                )
        self.layers.append(layer)
        return self

    def reset(self) -> None:
        """Resets the builder instance, clearing all layers stored in the layers list."""
        self.layers = []

    def get_particle(self) -> Particle:
        """
        Constructs and returns a `Particle` instance using the layers added so far.

        Returns:
            The constructed `Particle` instance.
        """
        return Particle(self.layers)

    def pop_particle(self) -> Particle:
        """
        Constructs and returns a `Particle` instance using the layers added so far,
        then resets the builder instance to start building a new particle.

        Returns:
            Particle: The constructed `Particle` instance.
        """
        particle = Particle(self.layers)
        self.reset()
        return particle


class ScatteringModel:
    # noinspection PyShadowingNames
    """
    Calculates scattering properties for a list of particles.

    This class computes various scattering properties, including amplitudes, form factors,
    and averages over multiple particles. It handles both single-particle and multi-particle
    scattering scenarios.

    Attributes:
        wavevector (NDArray[np.float64]): Array of wavevector values at which scattering properties are computed.
        mixture (MixtureLike): `Mixture` object containing number fractions and radii of particle components.
        particles (list): List of `Particle` instances representing the particles in the scattering model.

    Methods:
        __init__(wavevector, mixture, particles):
            Initializes a `ScatteringModel` instance with given wavevector, mixture, and particles.

    Usage:
        The typical usage involves creating an instance of `ScatteringModel` with wavevector, mixture,
        and a list of particles. Methods such as amplitude(), average_form_factor(), etc., can then
        be called to compute specific scattering properties.

    Example:
        Create a `ScatteringModel` instance from a list of particles:

            >>> import numpy as np
            >>> from mixscatter import Mixture, Particle, ScatteringModel
            >>> wavevector = np.linspace(0.01, 1.0, 100)
            >>> mixture = Mixture(number_fraction=[0.5, 0.5], radius=[1.0, 2.0])
            >>> particle1 = Particle([ConstantProfile(0, 1.0, 1.0)])
            >>> particle2 = Particle([ConstantProfile(0, 2.0, 0.5)])
            >>> model = ScatteringModel(wavevector, mixture, [particle1, particle2])
            >>> form_factor = model.average_form_factor

    Notes:
        - Scattering calculations are cached for efficiency.
    """

    def __init__(self, wavevector: ArrayLike, mixture: MixtureLike, particles: list[Particle]):
        """
        Initializes a `ScatteringModel` instance.

        Args:
            wavevector: Array of wavevector values at which scattering properties are computed.
            mixture: Mixture object containing number fractions and radii of particle components.
            particles: List of `Particle` instances representing the particles in the scattering model.
        """
        self.wavevector = np.asarray(wavevector, dtype=np.float64)
        self.mixture = mixture
        self.particles = particles

    @cached_property
    def amplitude(self) -> NDArray[np.float64]:
        """
        Calculates the scattering amplitude for each particle.

        Returns:
            Array of amplitudes for each particle at each wavevector point.
        """
        amplitude = np.empty((len(self.particles), len(self.wavevector)))
        for i, particle in enumerate(self.particles):
            amplitude[i] = particle.calculate_amplitude(self.wavevector)
        return amplitude

    @cached_property
    def forward_amplitude(self) -> NDArray[np.float64]:
        """
        Calculate the forward amplitude for each particle.

        Returns:
            An array of forward amplitudes for each particle.
        """
        forward_amplitude = np.empty(len(self.particles))
        for i, particle in enumerate(self.particles):
            forward_amplitude[i] = particle.calculate_forward_amplitude()
        return forward_amplitude

    @cached_property
    def single_form_factor(self) -> NDArray[np.float64]:
        """
        Computes the normalized form factors of the single species particles.

        Returns:
            Form factors of the single species particles.
        """
        return self.amplitude**2 / self.forward_amplitude[:, np.newaxis] ** 2

    @cached_property
    def average_square_amplitude(self) -> NDArray[np.float64]:
        """The sum of the squared scattering amplitudes, weighted by the number fraction.

        Returns:
            The average squared scattering amplitude.
        """
        average_square_amplitude: NDArray[np.float64] = np.sum(
            self.mixture.number_fraction[:, np.newaxis] * self.amplitude**2, axis=0, dtype=np.float64
        )
        return average_square_amplitude

    @cached_property
    def average_square_forward_amplitude(self) -> float:
        """The sum of the squared forward scattering amplitudes, weighted by the number fraction.

        Returns:
            The average squared forward scattering amplitude.
        """
        average_square_forward_amplitude: float = np.sum(
            self.mixture.number_fraction * self.forward_amplitude**2, axis=0, dtype=np.float64
        )
        return average_square_forward_amplitude

    @cached_property
    def average_form_factor(self) -> NDArray[np.float64]:
        """The average squared scattering amplitude, normalized by the average forward scattering amplitude.

        Returns:
            The average form factor.
        """
        return self.average_square_amplitude / self.average_square_forward_amplitude

    @cached_property
    def square_radius_of_gyration(self) -> NDArray[np.float64]:
        """
        Calculate the radius of gyration for each particle.

        Returns:
            An array containing the square radius of gyration for each particle.
        """
        square_radius_of_gyration = np.empty((len(self.particles)))
        for i, particle in enumerate(self.particles):
            square_radius_of_gyration[i] = particle.calculate_square_radius_of_gyration()
        return square_radius_of_gyration

    @cached_property
    def average_square_radius_of_gyration(self) -> float:
        """
        Computes the average, apparent radius of gyration of the system. The apparent radius of gyration
        determines the initial slope of the average form factor.

        Returns:
             The average radius of gyration of the system.
        """
        average_square_radius_of_gyration: float = (
            np.sum(
                self.mixture.number_fraction * self.forward_amplitude**2 * self.square_radius_of_gyration,
                axis=0,
                dtype=np.float64,
            )
            / self.average_square_forward_amplitude
        )
        return average_square_radius_of_gyration


class SimpleSphere(ScatteringModel):
    # noinspection PyShadowingNames
    """
    A convenience class for creating a scattering model of homogeneously scattering spheres.

    This class simplifies the creation of a scattering model where particles are represented
    by homogeneously scattering spheres with a common, constant contrast.

    Attributes:
        wavevector (NDArray[np.float64]): Array of wavevector values at which scattering properties are computed.
        mixture (MixtureLike): `Mixture` object containing number fractions and radii of particle components.
        particles (list): List of `Particle` instances representing the particles in the scattering model.

    Methods:
        __init__(wavevector, mixture, contrast):
            Initializes a `SimpleSphere` instance with given wavevector, mixture, and contrast.

    Usage:
        The typical usage involves creating an instance of `SimpleSphere` with specific wavevector,
        mixture, and contrast parameters, then using its inherited methods to compute scattering properties.

    Example:
        Initialize a `SimpleSphere` instance with wavevector, mixture, and contrast parameters:

            >>> import numpy as np
            >>> from mixscatter import Mixture, SimpleSphere
            >>> wavevector = np.linspace(0.01, 1.0, 100)
            >>> mixture = Mixture(number_fraction=[0.5, 0.5], radius=[1.0, 2.0])
            >>> model = SimpleSphere(wavevector, mixture, contrast=1.0)
            >>> form_factor = model.average_form_factor

    Notes:
        - The particle radii are inferred from the provided `Mixture` instance.
    """

    def __init__(self, wavevector: ArrayLike, mixture: MixtureLike, contrast: float) -> None:
        """
        Initializes a `SimpleSphere` instance.

        Args:
            wavevector: Array of wavevector values at which scattering properties are computed.
            mixture: `Mixture` object containing number fractions and radii of particle components.
            contrast: Scattering contrast of the spheres.
        """
        particles = []
        particle_builder = ParticleBuilder()
        for radius in mixture.radius:
            particle = particle_builder.add_layer(ConstantProfile(0, radius, contrast)).pop_particle()
            particles.append(particle)
        super().__init__(wavevector, mixture, particles)


class SimpleCoreShell(ScatteringModel):
    # noinspection PyShadowingNames
    """
    A convenience class for creating a scattering model of core-shell particles with a constant core-to-shell ratio.

    This class simplifies the creation of a scattering model where particles are represented
    by core-shell structures with varying contrasts for the core and shell layers.

    Attributes:
        wavevector (NDArray[np.float64]): Array of wavevector values at which scattering properties are computed.
        mixture (MixtureLike): `Mixture` object containing number fractions and radii of particle components.
        particles (list): List of `Particle` instances representing the particles in the scattering model.

    Methods:
        __init__(wavevector, mixture, core_to_total_ratio, core_contrast, shell_contrast):
            Initializes a `SimpleCoreShell` instance with given wavevector, mixture, and contrast parameters.

    Usage:
        The typical usage involves creating an instance of `SimpleCoreShell` with specific wavevector, mixture,
        core-shell ratio, and contrast parameters, then using its inherited methods to compute scattering
        properties.

    Example:
        Initialize a `SimpleCoreShell` instance with wavevector, mixture, and contrast parameters

            >>> import numpy as np
            >>> from mixscatter import Mixture, SimpleCoreShell
            >>> wavevector = np.linspace(0.01, 1.0, 100)
            >>> mixture = Mixture(number_fraction=[0.5, 0.5], radius=[1.0, 2.0])
            >>> model = SimpleCoreShell(
            ...     wavevector, mixture, core_to_total_ratio=0.5, core_contrast=1.0, shell_contrast=0.5
            ... )
            >>> form_factor = model.average_form_factor

    Notes:
        - The particle radii are inferred from the provided `Mixture` instance.
    """

    def __init__(
        self,
        wavevector: ArrayLike,
        mixture: MixtureLike,
        core_to_total_ratio: float,
        core_contrast: float,
        shell_contrast: float,
    ) -> None:
        """
        Initializes a `SimpleCoreShell` instance.

        Args:
            wavevector: Array of wavevector values at which scattering properties are computed.
            mixture: `Mixture` object containing number fractions and radii of particle components.
            core_to_total_ratio: Ratio of core radius to total particle radius.
            core_contrast: Scattering contrast of the core.
            shell_contrast: Scattering contrast of the shell.
        """
        particles = []
        particle_builder = ParticleBuilder()
        for total_radius in mixture.radius:
            core_radius = total_radius * core_to_total_ratio
            particle = (
                particle_builder.add_layer(ConstantProfile(0, core_radius, core_contrast))
                .add_layer(ConstantProfile(core_radius, total_radius, shell_contrast))
                .pop_particle()
            )
            particles.append(particle)
        super().__init__(wavevector, mixture, particles)


class SimpleGradient(ScatteringModel):
    # noinspection PyShadowingNames
    """
    A convenience class for creating a scattering model of particles with a linear contrast gradient.

    This class simplifies the creation of a scattering model representing particles with a scattering contrast
    varying linearly from the contrast at the particle center to the contrast at the boundary.

    Methods:
        __init__(wavevector, mixture, center_contrast, boundary_contrast):
            Initializes a `SimpleGradient` instance with given wavevector, mixture, and center and boundary
            contrast.

    Usage:
        The typical usage involves creating an instance of `SimpleGradient` with specific wavevector, mixture,
        and contrast profile, then using its inherited methods to compute scattering properties.

    Example:
        Initialize a `SimpleGradient` instance with wavevector, mixture, and contrast parameters:

            >>> import numpy as np
            >>> from mixscatter import Mixture
            >>> wavevector = np.linspace(0.01, 1.0, 100)
            >>> mixture = Mixture(number_fraction=[0.5, 0.5], radius=[1.0, 2.0])
            >>> model = SimpleGradient(wavevector, mixture, center_contrast=1.0, boundary_contrast=0.5)
            >>> form_factor = model.average_form_factor

    Notes:
        - The particle radii are inferred from the provided `Mixture` instance.
    """

    def __init__(
        self,
        wavevector: ArrayLike,
        mixture: MixtureLike,
        center_contrast: float,
        boundary_contrast: float,
    ) -> None:
        particles = []
        particle_builder = ParticleBuilder()
        for radius in mixture.radius:
            particle = particle_builder.add_layer(
                LinearProfile(0, radius, center_contrast, boundary_contrast)
            ).pop_particle()
            particles.append(particle)
        super().__init__(wavevector, mixture, particles)
