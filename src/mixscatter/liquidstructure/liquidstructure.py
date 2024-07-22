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

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@runtime_checkable
class HasMutableRadius(Protocol):
    radius: NDArray[np.float64]


# noinspection PyPropertyDefinition
@runtime_checkable
class MixtureLike(Protocol):  # pragma: no cover
    """
    A protocol defining the interface for mixture-like objects.

    Properties:
        number_fraction (NDArray[np.float64]): Number fractions of the components in the mixture.
        radius (NDArray[np.float64]): Radii of the components in the mixture.
        number_of_components (int): The number of different components in the mixture.

    Methods:
        moment(order: int) -> float: Calculate the specified moment of the size distribution.
    """

    @property
    def number_fraction(self) -> NDArray[np.float64]: ...

    @property
    def radius(self) -> NDArray[np.float64]: ...

    @property
    def number_of_components(self) -> int: ...

    def moment(self, order: int) -> float: ...


class LiquidStructure(ABC):
    """
    Representation of the liquid structure properties of an interacting multicomponent system of
    spherical particles.

    Specialized classes inherit from this base class and only need to implement the
    'partial_direct_correlation_function' attribute.

    Attributes:
        wavevector (NDArray[np.float64]): Scattering wavevector.
        mixture (MixtureLike): Mixture object.

    References:
        - P. Salgi, R. Rajagopalan, "Polydispersity in colloids: implications to static structure and scattering",
          Adv. Colloid Interface Sci. 43, 169-288 (1993), https://doi.org/10.1016/0001-8686(93)80017-6
        - A. Vrij, "Mixtures of hard spheres in the Percus–Yevick approximation. Light scattering at finite angles",
          J. Chem. Phys. 71, 3267-3270 (1979), https://doi.org/10.1063/1.438756
        - J. Diaz Maier, K. Gaus, J. Wagner, "Measurable structure factors of dense dispersions containing polydisperse,
          optically inhomogeneous particles", arXiv:2404.03470 [cond-mat.soft],
          https://doi.org/10.48550/arXiv.2404.03470
    """

    def __init__(self, wavevector: ArrayLike, mixture: MixtureLike):
        """
        Initialize an instance based on the specified wavevectors and for a given mixture.

        Args:
            wavevector (ArrayLike): Scattering wavevector.
            mixture (MixtureLike): Mixture object.
        """
        self.wavevector: NDArray[np.float64] = np.asarray(wavevector, dtype=np.float64)
        self.mixture: MixtureLike = mixture

    @abstractmethod
    @cached_property
    def partial_direct_correlation_function(self) -> NDArray[np.float64]:  # pragma: no cover
        """
        Matrix of the partial direct correlation functions weighted by the total number density.

        Returns:
            NDArray[np.float64]: Partial direct correlation function matrix.
        """
        ...

    @cached_property
    def number_weighted_partial_direct_correlation_function(self) -> NDArray[np.float64]:
        """
        The partial direct correlation function matrix `c_ij` weighted by the square root of the
        product of the number fractions `x_i` and `x_j`.

        Returns:
            NDArray[np.float64]: Matrix of the weighted direct partial correlation functions.
        """
        c_weighted_ijq: NDArray[np.float64] = np.einsum(
            "i, ijq, j->ijq",
            np.sqrt(self.mixture.number_fraction),
            self.partial_direct_correlation_function,
            np.sqrt(self.mixture.number_fraction),
            optimize="greedy",
        )
        return c_weighted_ijq

    @cached_property
    def partial_structure_factor(self) -> NDArray[np.float64]:
        """
        The partial structure factor matrix.

        Returns:
            NDArray[np.float64]: Matrix of the partial structure factors.
        """
        c_weighted_ijq = self.number_weighted_partial_direct_correlation_function
        c_weighted_qij = np.moveaxis(c_weighted_ijq, -1, 0)
        unity_tensor_qij = np.eye(self.mixture.number_of_components)[np.newaxis, :, :]
        S_inv_qij = unity_tensor_qij - c_weighted_qij
        S_qij = np.linalg.solve(S_inv_qij, unity_tensor_qij)
        S_ijq = np.moveaxis(S_qij, 0, -1)
        return S_ijq

    @cached_property
    def number_weighted_partial_structure_factor(self) -> NDArray[np.float64]:
        """
        The partial direct correlation function matrix `S_ij` weighted by the square root of the
        product of the number fractions `x_i` and `x_j`.

        Returns:
            NDArray[np.float64]: Matrix of the weighted direct partial correlation functions.
        """
        S_weighted_ijq: NDArray[np.float64] = np.einsum(
            "i, ijq, j->ijq",
            np.sqrt(self.mixture.number_fraction),
            self.partial_structure_factor,
            np.sqrt(self.mixture.number_fraction),
            optimize="greedy",
        )
        return S_weighted_ijq

    @cached_property
    def average_structure_factor(self) -> NDArray[np.float64]:
        """
        The sum of the number weighted partial structure factors over all species.

        The number-number structure factor for all species, regardless their individual size.

        Returns:
            NDArray[np.float64]: Average structure factor.
        """
        average_structure_factor: NDArray[np.float64] = np.einsum(
            "ijq->q", self.number_weighted_partial_structure_factor
        )
        return average_structure_factor

    @cached_property
    def compressibility_structure_factor(self) -> NDArray[np.float64]:
        """
        The "compressibility" or "Kirkwood-Buff" structure factor.

        This structure factor gives the correct isothermal compressibility from S(Q->0) according
        to the Kirkwood-Buff-theory for multicomponent systems.

        Returns:
            NDArray[np.float64]: "Compressibility" structure factor.
        """
        S_weighted_ijq = self.number_weighted_partial_structure_factor
        S_weighted_qij = np.moveaxis(S_weighted_ijq, -1, 0)
        S_weighted_inverse_qij = np.linalg.inv(S_weighted_qij)
        compressibility_structure_factor: NDArray[np.float64] = 1.0 / np.einsum(
            "i, j, qij->q",
            self.mixture.number_fraction,
            self.mixture.number_fraction,
            S_weighted_inverse_qij,
        )
        return compressibility_structure_factor


class PercusYevick(LiquidStructure):
    """
    Hard-sphere potential in the Percus-Yevick approximation.

    References:
        - A. Vrij, "Mixtures of hard spheres in the Percus–Yevick approximation. Light scattering at finite angles",
          J. Chem. Phys. 71, 3267-3270 (1979), https://doi.org/10.1063/1.438756
        - T. Voigtmann, "Mode Coupling Theory of the Glass Transition in Binary Mixtures",
          PhD Thesis, TU München, https://mediatum.ub.tum.de/603008
        - J. Diaz Maier, K. Gaus, J. Wagner, "Measurable structure factors of dense dispersions containing polydisperse,
          optically inhomogeneous particles", arXiv:2404.03470 [cond-mat.soft],
          https://doi.org/10.48550/arXiv.2404.03470
    """

    def __init__(self, wavevector: ArrayLike, mixture: MixtureLike, volume_fraction_total: float):
        """
        Initialize an instance based on the specified wavevectors and for a given mixture.

        Args:
            wavevector (ArrayLike): Scattering wavevector.
            mixture (MixtureLike): Mixture object.
            volume_fraction_total (float): Total volume fraction.
        """
        self.volume_fraction_total: float = volume_fraction_total
        super().__init__(wavevector, mixture)

    @cached_property
    def partial_direct_correlation_function(self) -> NDArray[np.float64]:
        """
        Calculate the partial direct correlation function matrix `c_ij(q)` for the system using the
        Percus-Yevick approximation.

        Returns:
            NDArray[np.float64]: Partial direct correlation function matrix `c_ij(q)` times the total number density.
        """
        average_volume = 4.0 / 3.0 * np.pi * self.mixture.moment(3)
        number_density_total = self.volume_fraction_total / average_volume
        number_density = number_density_total * self.mixture.number_fraction

        diameter = 2.0 * self.mixture.radius
        xi = [np.pi / 6.0 * np.sum(number_density * diameter**x) for x in range(1, 4)]
        diameter_mean = 0.5 * (diameter[:, np.newaxis] + diameter)
        diameter_product = diameter[:, np.newaxis] * diameter
        beta_hat_0 = (9.0 * xi[1] ** 2 + 3.0 * xi[0] * (1 - xi[2])) / (1 - xi[2]) ** 3
        a_i = (1.0 - xi[2] + 3.0 * diameter * xi[1]) / (1 - xi[2]) ** 2
        a_tilde_2 = np.sum(number_density * a_i**2)
        A_ij = (diameter_mean * (1 - xi[2]) + 1.5 * diameter_product * xi[1]) / (1 - xi[2]) ** 2
        B_ij = 1.0 / (1 - 0 - xi[2]) - beta_hat_0 * diameter_product
        D_ij = (6.0 * xi[1] + 12.0 * diameter_mean * (xi[0] + 3.0 * xi[1] ** 2 / (1 - xi[2]))) / (1 - xi[2]) ** 2
        wavevector_reduced = self.mixture.radius[:, np.newaxis] * self.wavevector
        wavevector_reciprocal = 1.0 / self.wavevector
        sin_iq = np.sin(wavevector_reduced)
        cos_iq = np.cos(wavevector_reduced)

        c_ijq = 4.0 * np.pi * a_tilde_2 * sin_iq[:, np.newaxis] * sin_iq
        c_ijq *= wavevector_reciprocal

        c_ijq -= (
            2.0
            * np.pi
            * a_tilde_2
            * (
                cos_iq[:, np.newaxis] * sin_iq * diameter[:, np.newaxis, np.newaxis]
                + sin_iq[:, np.newaxis] * cos_iq * diameter[np.newaxis, :, np.newaxis]
            )
        )
        c_ijq *= wavevector_reciprocal

        c_ijq += (
            D_ij[:, :, np.newaxis] * sin_iq[:, np.newaxis] * sin_iq
            + np.pi * a_tilde_2 * cos_iq[:, np.newaxis] * cos_iq * diameter_product[:, :, np.newaxis]
        )
        c_ijq *= wavevector_reciprocal

        c_ijq += B_ij[:, :, np.newaxis] * (cos_iq[:, np.newaxis] * sin_iq + sin_iq[:, np.newaxis] * cos_iq)
        c_ijq *= wavevector_reciprocal

        c_ijq += A_ij[:, :, np.newaxis] * (sin_iq[:, np.newaxis] * sin_iq - cos_iq[:, np.newaxis] * cos_iq)
        c_ijq *= -4.0 * np.pi * wavevector_reciprocal**2
        c_ijq *= number_density_total
        return np.asarray(c_ijq, dtype=np.float64)


class VerletWeis(PercusYevick):
    """
    Hard-sphere potential in the Percus-Yevick approximation employing the Verlet-Weiss correction.

    Uses the PercusYevick class with an effective volume fraction and an effective radius.

    References:
        - E. W. Grundke and D. Henderson, "Distribution functions of multi-component fluid mixtures of hard spheres",
          Mol. Phys. 24, 269-281 (1972), https://doi.org/10.1080/00268977200101431
    """

    def __init__(self, wavevector: ArrayLike, mixture: MixtureLike, volume_fraction_total: float) -> None:
        """
        Initialize an instance based on the specified wavevectors and for a given mixture.

        Args:
            wavevector (ArrayLike): Scattering wavevector.
            mixture (MixtureLike): Mixture object.
            volume_fraction_total (float): Total volume_fraction.
        """
        effective_volume_fraction = volume_fraction_total * (1.0 - volume_fraction_total / 16.0)
        effective_radius = mixture.radius * (effective_volume_fraction / volume_fraction_total) ** (1.0 / 3.0)

        # Some static type checkers at the moment cannot handle protocols implementing property setters
        # and since a mutable radius property is only needed here, a manual isinstance assertion is performed
        if isinstance(mixture, HasMutableRadius) and getattr(type(mixture), "radius").fset is not None:
            mixture.radius = effective_radius
            assert isinstance(mixture, MixtureLike)
        else:
            raise AttributeError("`radius` property of `mixture` must be mutable.")

        super().__init__(wavevector, mixture, volume_fraction_total=effective_volume_fraction)
