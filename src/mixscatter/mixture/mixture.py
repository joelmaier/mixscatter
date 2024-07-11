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

import logging

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import roots_genlaguerre, eval_genlaguerre, roots_hermite, roots_legendre

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class Mixture:
    """Representation of the size distribution of an N-component mixture of spherical particles.

    Such a mixture can be interpreted as a collection of spheres with a discrete probability
    distribution of the radii, where the number fraction of each species describes their
    respective weight.

    Args:
        radius: The physical, geometric radius of each component.
        number_fraction: Number fraction of each component.
        normalize_number_fraction: If True, `number_fraction` will be explicitly normalized
                                   such that the sum over all elements equals 1.
    """

    def __init__(self, radius: ArrayLike, number_fraction: ArrayLike, normalize_number_fraction: bool = True) -> None:
        self._radius: NDArray[np.float64] = np.asarray(radius, dtype=np.float64)
        self._number_fraction: NDArray[np.float64] = np.asarray(number_fraction, dtype=np.float64)

        if normalize_number_fraction:
            self._number_fraction /= np.sum(self._number_fraction)

        self._number_of_components: int = len(self._radius)

    @property
    def radius(self) -> NDArray[np.float64]:
        """
        Get an array of the radii of each component.

        Returns:
            An array of the radii.
        """
        return self._radius

    @radius.setter
    def radius(self, radius_array: ArrayLike) -> None:
        """
        Set new `radius`.

        Args:
            radius_array: New radius array of the same shape as the current radius array.
        """
        new_array: NDArray[np.float64] = np.asarray(radius_array, dtype=np.float64)
        if self._radius.shape != new_array.shape:
            raise ValueError("The new radius array must be of same shape as the current radius array.")
        self._radius = new_array

    @property
    def number_fraction(self) -> NDArray[np.float64]:
        """
        Get an array of the number fractions of each component

        Returns:
            An array of the number fractions.
        """
        return self._number_fraction

    @number_fraction.setter
    def number_fraction(self, number_fraction_array: ArrayLike) -> None:
        """
        Set new `number_fraction`.

        Args:
            number_fraction_array:
        """
        new_array: NDArray[np.float64] = np.asarray(number_fraction_array, dtype=np.float64)
        if self._number_fraction.shape != new_array.shape:
            raise ValueError(
                "The new number fraction array must be of same shape as the current number fraction array."
            )
        self._number_fraction = new_array

    @property
    def number_of_components(self) -> int:
        """
        Get the number of components of the mixture.

        Returns:
            The number of components.
        """
        return self._number_of_components

    def moment(self, order: int) -> float:
        """Calculate the moment of the specified order.

        Args:
            order: Order of the moment to calculate.

        Returns:
            The moment of the specified order.
        """
        return float(np.sum(self.number_fraction * self.radius**order))

    def central_moment(self, order: int) -> float:
        """Calculate the central moment of the specified order.

        Args:
            order: Order of the central moment to calculate.

        Returns:
            The central moment of the specified order.
        """
        return float(np.sum(self.number_fraction * (self.radius - self.mean) ** order))

    @property
    def mean(self) -> float:
        """
        Calculate the first moment (mean) of the distribution.

        Returns:
            The mean of the distribution.
        """
        return self.moment(1)

    @property
    def variance(self) -> float:
        """
        Calculate the second central moment (variance) of the distribution.

        Returns:
            The variance of the distribution.
        """
        return self.central_moment(2)

    @property
    def polydispersity(self) -> float:
        """
        Calculate the polydispersity, which is the ratio of the standard deviation
        of the distribution to its mean.

        Returns:
            The polydispersity of the distribution.
        """
        return float(np.sqrt(self.variance)) / self.mean


class SingleComponent(Mixture):
    """Representation of a pseudo one-component system.

    Args:
        radius: Radius.
    """

    def __init__(self, radius: float) -> None:
        super().__init__([radius], [1.0])


class FlorySchulzMixture(Mixture):
    """Representation of a Schulz-Flory distributed mixture.

    Args:
        number_of_components: Number of components in the mixture.
        mean_radius: Mean radius of the distribution.
        shape_parameter: Flory parameter controlling the distribution shape.
    """

    def __init__(self, number_of_components: int, mean_radius: float, shape_parameter: float) -> None:
        roots, weights = self._flory_schulz_weights(
            int(number_of_components), float(mean_radius), float(shape_parameter)
        )
        super().__init__(roots, weights)

    @classmethod
    def _flory_schulz_weights(
        cls, number_of_components: int, mean_radius: float, shape_parameter: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        roots, weights = cls._generalized_laguerre_weights(number_of_components, shape_parameter)
        scaled_roots = roots * mean_radius / (shape_parameter + 1.0)
        weights /= np.sum(weights)
        return scaled_roots, weights

    @staticmethod
    def _generalized_laguerre_weights(order: int, exponent: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        roots, *_ = roots_genlaguerre(order, exponent)
        values_at_roots = eval_genlaguerre(order + 1, exponent, roots)
        abs_values_at_roots = np.abs(values_at_roots)
        log_weights = np.log(roots) - 2.0 * np.log(abs_values_at_roots)
        weights = np.exp(log_weights)
        return roots, weights


class GaussianMixture(Mixture):
    """Representation of a Gaussian distributed mixture.

    Args:
        number_of_components: Number of components in the mixture.
        mean_radius: Mean radius of the distribution.
        standard_deviation: Standard deviation of the distribution.
        truncate:
            Method to handle unphysical radii. Options are 'no', 'negatives', and 'symmetric'.

            - 'no': No truncation

            - 'negatives': Truncate negative radii.

            - 'symmetric': Truncate negative radii and force the distribution to stay symmetric.
    """

    def __init__(
        self, number_of_components: int, mean_radius: float, standard_deviation: float, truncate: str = "negatives"
    ) -> None:
        if truncate not in ["no", "negatives", "symmetric"]:
            raise ValueError("Parameter 'truncate' must be 'no', 'negatives', or 'symmetric'.")

        roots, weights = self._gaussian_weights(
            int(number_of_components), float(mean_radius), float(standard_deviation)
        )

        if truncate == "negatives":
            roots_to_remove = roots < 0
        elif truncate == "symmetric":
            roots_to_remove = np.abs(roots - mean_radius) > mean_radius
        else:
            roots_to_remove = np.full_like(roots, False, dtype=bool)

        if np.any(roots_to_remove):
            roots_to_keep = ~roots_to_remove
            number_of_components_new = np.count_nonzero(roots_to_keep)
            roots = roots[roots_to_keep]
            weights = weights[roots_to_keep]

            logger.warning(
                f"Components with unphysical radii were removed. Number of components "
                f"was reduced from {number_of_components} to {number_of_components_new}."
            )

        super().__init__(roots, weights)

    @staticmethod
    def _gaussian_weights(
        number_of_components: int, mean_radius: float, standard_deviation: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        roots, weights = roots_hermite(number_of_components)
        scaled_roots = roots * np.sqrt(2) * standard_deviation + mean_radius
        weights /= np.sum(weights)
        return scaled_roots, weights


class UniformMixture(Mixture):
    """Representation of a uniformly distributed mixture.

    Args:
        number_of_components: Number of components in the mixture.
        lower_bound: Lower bound of the distribution domain.
        upper_bound: Upper bound of the distribution domain.
    """

    def __init__(self, number_of_components: int, lower_bound: float, upper_bound: float) -> None:
        roots, weights = self._uniform_weights(int(number_of_components), float(upper_bound), float(lower_bound))
        super().__init__(roots, weights)

    @staticmethod
    def _uniform_weights(
        number_of_components: int, lower_bound: float, upper_bound: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        roots, weights = roots_legendre(number_of_components)
        scaled_roots = upper_bound * 0.5 * (roots + 1) - lower_bound * 0.5 * (roots - 1)
        weights /= np.sum(weights)
        return scaled_roots, weights
