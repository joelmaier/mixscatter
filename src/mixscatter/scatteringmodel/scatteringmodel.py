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

Functions:
    _two_point_to_slope_intercept: Utility function for calculating slope and intercept between two points.

Examples:
    Create a simple scattering model for spheres:

    >>> from mixscatter.mixture import Mixture
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
from typing import Any, Protocol, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray


# noinspection PyPropertyDefinition
class MixtureLike(Protocol):  # pragma: no cover
    @property
    def number_fraction(self) -> NDArray[np.float_]: ...

    @property
    def radius(self) -> NDArray[np.float_]: ...


class LayerProfile(Protocol):  # pragma: no cover
    """Defines the interface for layer profiles."""

    radius_inner: float
    radius_outer: float

    def calculate_amplitude(self, wavevector: ArrayLike) -> NDArray[np.float_]:
        """Calculate the amplitude for the given wavevector."""
        ...

    def calculate_forward_amplitude(self) -> float:
        """Calculate the forward amplitude."""
        ...

    def get_profile(self, distance: ArrayLike) -> NDArray[np.float_]:
        """Get the profile for the given distance from the origin."""
        ...


class EmptyProfile(LayerProfile):
    """Represents an empty layer profile."""

    def __init__(self, radius_inner: float, radius_outer: float) -> None:
        if radius_inner > radius_outer:
            raise RuntimeError("'radius_inner' must be smaller than 'radius_outer'.")

        self.radius_inner = radius_inner
        self.radius_outer = radius_outer

    def calculate_amplitude(self, wavevector: ArrayLike) -> NDArray[np.float_]:
        """Calculate the amplitude for the given wavevector.

        Returns:
            A zero array of the same shape as the wavevector.
        """
        wavevector = np.asarray(wavevector)
        return np.zeros_like(wavevector)

    def calculate_forward_amplitude(self) -> float:
        """Calculate the forward amplitude.

        Returns:
            Zero.
        """
        return 0.0

    def get_profile(self, distance: ArrayLike) -> NDArray[np.float_]:
        """Get the profile for the given distance from the origin.

        Returns:
            A zero array of the same shape as the distance.
        """
        distance = np.asarray(distance)
        return np.zeros_like(distance)


class ConstantProfile(LayerProfile):
    """Represents a layer profile with constant contrast."""

    def __init__(self, radius_inner: float, radius_outer: float, contrast: float) -> None:
        if radius_inner > radius_outer:
            raise RuntimeError("'radius_inner' must be smaller than 'radius_outer'.")

        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        self.contrast = contrast

    def calculate_amplitude(self, wavevector: ArrayLike) -> NDArray[np.float_]:
        """Calculate the amplitude for the given wavevector.

        Returns:
            The calculated amplitude array.
        """
        wavevector = np.asarray(wavevector)
        QR_outer = wavevector * self.radius_outer
        QR_inner = wavevector * self.radius_inner
        amplitude: NDArray[np.float_] = np.sin(QR_outer) - QR_outer * np.cos(QR_outer)
        amplitude -= np.sin(QR_inner) - QR_inner * np.cos(QR_inner)
        amplitude *= 4.0 * np.pi / wavevector**3 * self.contrast
        return amplitude

    def calculate_forward_amplitude(self) -> float:
        """Calculate the forward amplitude.

        Returns:
            The calculated forward amplitude.
        """
        return 4.0 / 3.0 * np.pi * (self.radius_outer**3 - self.radius_inner**3) * self.contrast

    def get_profile(self, distance: ArrayLike) -> NDArray[np.float_]:
        """Get the profile for the given distance from the origin.

        Returns:
            The profile evaluated on the distance array.
        """
        distance = np.asarray(distance)
        distance_mask = (distance >= self.radius_inner) & (distance <= self.radius_outer)
        return np.where(distance_mask, self.contrast, 0.0)


class LinearProfile(LayerProfile):
    """Represents a layer profile with linearly varying contrast."""

    def __init__(self, radius_inner: float, radius_outer: float, contrast_inner: float, contrast_outer: float) -> None:
        if radius_inner > radius_outer:
            raise RuntimeError("'radius_inner' must be smaller than 'radius_outer'.")

        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        self.contrast_inner = contrast_inner
        self.contrast_outer = contrast_outer

    def calculate_amplitude(self, wavevector: ArrayLike) -> NDArray[np.float_]:
        """Calculate the amplitude for the given wavevector.

        Returns:
            The calculated amplitude array.
        """
        wavevector = np.asarray(wavevector)
        QR_outer = wavevector * self.radius_outer
        QR_inner = wavevector * self.radius_inner
        intercept, slope = self._two_point_to_slope_intercept(
            self.radius_inner, self.contrast_inner, self.radius_outer, self.contrast_outer
        )
        amplitude_intercept = ConstantProfile(self.radius_inner, self.radius_outer, intercept).calculate_amplitude(
            wavevector
        )
        amplitude_gradient: NDArray[np.float_] = (2.0 - QR_outer**2) * np.cos(QR_outer) + 2.0 * QR_outer * np.sin(
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

    def get_profile(self, distance: ArrayLike) -> NDArray[np.float_]:
        """Get the profile for the given distance from the origin.

        Returns:
            The profile evaluated on the distance array.
        """
        distance = np.asarray(distance)
        distance_mask = (distance >= self.radius_inner) & (distance <= self.radius_outer)
        intercept, slope = self._two_point_to_slope_intercept(
            self.radius_inner, self.contrast_inner, self.radius_outer, self.contrast_outer
        )
        profile = np.zeros_like(distance)
        profile[distance_mask] = intercept + slope * distance[distance_mask]
        return profile


class Particle:
    """Represents a particle composed of multiple layers."""

    def __init__(self, layers: list[LayerProfile]) -> None:
        self.layers = layers

    def calculate_amplitude(self, wavevector: ArrayLike) -> NDArray[np.float_]:
        """Calculate the amplitude for the given wavevector.

        Returns:
            The calculated amplitude array.
        """
        wavevector = np.asarray(wavevector)
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

    def calculate_form_factor(self, wavevector: ArrayLike) -> NDArray[np.float_]:
        """Calculate the form factor for the given wavevector.

        Returns:
            The calculated form factor array.
        """
        wavevector = np.asarray(wavevector)
        amplitude = self.calculate_amplitude(wavevector)
        forward_amplitude = self.calculate_forward_amplitude()
        return (amplitude / forward_amplitude) ** 2

    def get_profile(self, distance: ArrayLike) -> NDArray[np.float_]:
        """Get the profile for the given distance from the origin.

        Returns:
            The profile evaluated on the distance array.
        """
        distance = np.asarray(distance)
        profile = np.zeros_like(distance)
        for layer in self.layers:
            profile += layer.get_profile(distance)
        return profile


class ParticleBuilder:
    """A builder class for constructing Particle instances."""

    def __init__(self) -> None:
        self.layers: list[LayerProfile] = []

    def add_layer(self, layer: LayerProfile) -> Self:
        """Add a layer to the particle being built.

        Args:
            layer: LayerProfile instance to be added.

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
        """Reset the builder, clearing all layers."""
        self.layers = []

    def get_particle(self) -> Particle:
        """Get the constructed particle.

        Returns:
            The constructed Particle instance.
        """
        return Particle(self.layers)

    def pop_particle(self) -> Particle:
        """Get the constructed particle and reset the builder.

        Returns:
            The constructed Particle instance.
        """
        particle = Particle(self.layers)
        self.reset()
        return particle


class ScatteringModel:
    """Calculates scattering properties for a list of particles."""

    def __init__(self, wavevector: ArrayLike, mixture: MixtureLike, particles: list[Particle]):
        self.wavevector = np.asarray(wavevector)
        self.mixture = mixture
        self.particles = particles

    @cached_property
    def amplitude(self) -> NDArray[np.float_]:
        """Calculate the amplitude for each particle.

        Returns:
            An array of amplitudes for each particle.
        """
        amplitude = np.empty((len(self.particles), len(self.wavevector)))
        for i, particle in enumerate(self.particles):
            amplitude[i] = particle.calculate_amplitude(self.wavevector)
        return amplitude

    @cached_property
    def forward_amplitude(self) -> NDArray[np.float_]:
        """Calculate the forward amplitude for each particle.

        Returns:
            An array of forward amplitudes for each particle.
        """
        forward_amplitude = np.empty(len(self.particles))
        for i, particle in enumerate(self.particles):
            forward_amplitude[i] = particle.calculate_forward_amplitude()
        return forward_amplitude

    @cached_property
    def single_form_factor(self) -> NDArray[np.float_]:
        """The normalized form factors of the single species.

        Returns:
            The form factors.
        """
        return self.amplitude**2 / self.forward_amplitude[:, np.newaxis] ** 2

    @cached_property
    def average_square_amplitude(self) -> Any:
        """The sum of the squared scattering amplitudes, weighted by the number fraction.

        Returns:
            The average squared scattering amplitude.
        """
        return np.sum(self.mixture.number_fraction[:, np.newaxis] * self.amplitude**2, axis=0)

    @cached_property
    def average_square_forward_amplitude(self) -> Any:
        """The sum of the squared forward scattering amplitudes, weighted by the number fraction.

        Returns:
            The average squared forward scattering amplitude.
        """
        return np.sum(self.mixture.number_fraction * self.forward_amplitude**2, axis=0)

    @cached_property
    def average_form_factor(self) -> Any:
        """The average squared scattering amplitude, normalized by the average forward scattering amplitude.

        Returns:
            The average form factor.
        """
        return self.average_square_amplitude / self.average_square_forward_amplitude


class SimpleSphere(ScatteringModel):
    """A convenience class for creating a scattering model of homogeneously scattering spheres."""

    def __init__(self, wavevector: ArrayLike, mixture: MixtureLike, contrast: float) -> None:
        particles = []
        particle_builder = ParticleBuilder()
        for radius in mixture.radius:
            particle = particle_builder.add_layer(ConstantProfile(0, radius, contrast)).pop_particle()
            particles.append(particle)
        super().__init__(wavevector, mixture, particles)


class SimpleCoreShell(ScatteringModel):
    """A convenience class for creating a scattering model of core-shell particles."""

    def __init__(
        self,
        wavevector: ArrayLike,
        mixture: MixtureLike,
        core_to_total_ratio: float,
        core_contrast: float,
        shell_contrast: float,
    ) -> None:
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
    """A convenience class for creating a scattering model with gradient profiles."""

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
