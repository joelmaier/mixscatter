# type: ignore
import pytest
import numpy as np

from mixscatter.mixture import (
    Mixture,
    FlorySchulzMixture,
    GaussianMixture,
    UniformMixture,
    SingleComponent,
)


def test_mixture_initialization():
    radius = [100, 250]
    number_fraction = [0.5, 0.5]
    mixture = Mixture(radius, number_fraction)

    np.testing.assert_allclose(mixture.radius, np.array(radius))
    np.testing.assert_allclose(mixture.number_fraction, np.array(number_fraction))
    assert mixture.number_of_components == len(radius)


def test_mixture_normalization():
    radius = [100, 250]
    number_fraction = [1.0, 2.0]
    mixture = Mixture(radius, number_fraction, normalize_number_fraction=True)

    expected_number_fraction = np.array([1.0, 2.0]) / 3.0
    np.testing.assert_allclose(mixture.number_fraction, expected_number_fraction)


def test_mixture_no_normalization():
    radius = [100, 250]
    number_fraction = [1.0, 2.0]
    mixture = Mixture(radius, number_fraction, normalize_number_fraction=False)

    # Test that number_fraction remains unchanged
    np.testing.assert_array_equal(mixture.number_fraction, np.array(number_fraction))


def test_mixture_moment():
    radius = [100, 250]
    number_fraction = [0.5, 0.5]
    mixture = Mixture(radius, number_fraction)

    moment_order_1 = mixture.moment(1)
    expected_moment_order_1 = np.sum(np.array(number_fraction) * np.array(radius) ** 1)
    assert moment_order_1 == expected_moment_order_1


def test_mixture_central_moment():
    radius = [100, 250]
    number_fraction = [0.5, 0.5]
    mixture = Mixture(radius, number_fraction)

    central_moment_order_2 = mixture.central_moment(2)
    expected_central_moment_order_2 = np.sum(number_fraction * (np.array(radius) - mixture.mean) ** 2)
    assert central_moment_order_2 == expected_central_moment_order_2


def test_mixture_mean():
    radius = [100, 250]
    number_fraction = [0.5, 0.5]
    mixture = Mixture(radius, number_fraction)

    expected_mean = np.sum(np.array(number_fraction) * np.array(radius))
    assert mixture.mean == expected_mean


def test_mixture_variance():
    radius = [100, 250]
    number_fraction = [0.5, 0.5]
    mixture = Mixture(radius, number_fraction)

    expected_variance = np.sum(number_fraction * (np.array(radius) - mixture.mean) ** 2)
    assert mixture.variance == expected_variance


def test_mixture_polydispersity():
    radius = [100, 250]
    number_fraction = [0.5, 0.5]
    mixture = Mixture(radius, number_fraction)

    expected_polydispersity = np.sqrt(mixture.variance) / mixture.mean
    assert mixture.polydispersity == expected_polydispersity


def test_mixture_setters():
    radius = [100, 250]
    number_fraction = [0.5, 0.5]
    mixture = Mixture(radius, number_fraction)

    np.testing.assert_array_equal(mixture.radius, np.array(radius))
    np.testing.assert_array_equal(mixture.number_fraction, np.array(number_fraction))

    new_radius = [50, 500]
    new_number_fraction = [0.2, 0.8]
    mixture.radius = new_radius
    mixture.number_fraction = new_number_fraction
    np.testing.assert_array_equal(mixture.radius, np.array(new_radius))
    np.testing.assert_array_equal(mixture.number_fraction, np.array(new_number_fraction))


def test_mixture_setters_invalid_shape():
    radius = [100, 250]
    number_fraction = [0.5, 0.5]
    mixture = Mixture(radius, number_fraction)

    new_radius = [50, 500, 700]
    new_number_fraction = [0.2, 0.6, 0.2]

    with pytest.raises(ValueError, match="The new radius array must be of same shape as the current radius array."):
        mixture.radius = new_radius

    with pytest.raises(
        ValueError, match="The new number fraction array must be of same shape as the current number fraction array."
    ):
        mixture.number_fraction = new_number_fraction


def test_single_component_initialization():
    radius = 500
    mixture = SingleComponent(radius)
    assert np.allclose(mixture.radius, [radius])
    assert np.allclose(mixture.number_fraction, [1.0])


def test_flory_schulz_mixture_initialization():
    number_of_components = 16
    mean_radius = 100
    shape_parameter = 99
    mixture = FlorySchulzMixture(number_of_components, mean_radius, shape_parameter)
    assert mixture.number_of_components == number_of_components
    assert np.isclose(mixture.mean, mean_radius)
    assert np.isclose(mixture.polydispersity, 1 / np.sqrt(shape_parameter + 1))


def test_gaussian_mixture_initialization():
    number_of_components = 16
    mean_radius = 100
    standard_deviation = 10
    mixture = GaussianMixture(number_of_components, mean_radius, standard_deviation)
    assert mixture.number_of_components == number_of_components
    assert np.isclose(mixture.mean, mean_radius)
    assert np.isclose(mixture.polydispersity, standard_deviation / mean_radius)


def test_gaussian_mixture_truncate():
    number_of_components = 16
    mean_radius = 100
    standard_deviation = 40
    mixture_truncate_negatives = GaussianMixture(
        number_of_components, mean_radius, standard_deviation, truncate="negatives"
    )
    mixture_truncate_symmetric = GaussianMixture(
        number_of_components, mean_radius, standard_deviation, truncate="symmetric"
    )
    mixture_truncate_no = GaussianMixture(number_of_components, mean_radius, standard_deviation, truncate="no")
    assert mixture_truncate_negatives.number_of_components < number_of_components
    assert np.all(mixture_truncate_negatives.radius >= 0)
    assert np.all(
        mixture_truncate_symmetric.radius - mixture_truncate_symmetric.mean <= mixture_truncate_symmetric.mean
    )
    assert mixture_truncate_symmetric.number_of_components < number_of_components
    assert 2 * (number_of_components - mixture_truncate_negatives.number_of_components) == (
        number_of_components - mixture_truncate_symmetric.number_of_components
    )

    assert mixture_truncate_no.number_of_components == number_of_components


def test_gaussian_mixture_invalid_truncate():
    number_of_components = 16
    mean_radius = 100
    standard_deviation = 10

    with pytest.raises(ValueError, match="Parameter 'truncate' must be 'no', 'negatives', or 'symmetric'."):
        GaussianMixture(number_of_components, mean_radius, standard_deviation, truncate="invalid")


def test_uniform_mixture_initialization():
    number_of_components = 16
    lower_bound = 50
    upper_bound = 150
    mixture = UniformMixture(number_of_components, lower_bound, upper_bound)
    assert mixture.number_of_components == number_of_components
    assert np.isclose(mixture.mean, (upper_bound + lower_bound) / 2)
    assert np.isclose(mixture.variance, (upper_bound - lower_bound) ** 2 / 12)


if __name__ == "__main__":
    pytest.main()
