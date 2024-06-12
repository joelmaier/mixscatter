# type: ignore
import pytest
import numpy as np

from mixscatter.core import measurable_intensity, measurable_structure_factor, measurable_diffusion_coefficient


@pytest.fixture
def mock_structure_factor(mocker):
    sf = mocker.Mock()
    sf.number_weighted_partial_structure_factor = np.random.rand(2, 2, 100)
    return sf


@pytest.fixture
def mock_form_factor(mocker):
    ff = mocker.Mock()
    ff.mixture = mocker.Mock()
    ff.mixture.number_fraction = np.random.rand(2)
    ff.mixture.radius = np.random.rand(2)
    ff.amplitude = np.random.rand(2, 100)
    ff.average_square_amplitude = np.random.rand(100)
    ff.average_square_forward_amplitude = np.random.rand(100)
    ff.average_form_factor = np.random.rand(100)
    return ff


def test_measurable_intensity_default(mock_structure_factor, mock_form_factor):
    scale = 1.0
    background = 0.0

    result = measurable_intensity(mock_structure_factor, mock_form_factor)

    expected_intensity = np.einsum(
        "iq, jq, ijq->q",
        mock_form_factor.amplitude,
        mock_form_factor.amplitude,
        mock_structure_factor.number_weighted_partial_structure_factor,
    )
    expected_normalized_intensity = expected_intensity / mock_form_factor.average_square_forward_amplitude
    expected_result = scale * expected_normalized_intensity + background

    np.testing.assert_allclose(result, expected_result)


def test_measurable_intensity_scaled(mock_structure_factor, mock_form_factor):
    scale = 1e5
    background = 1e3

    result = measurable_intensity(mock_structure_factor, mock_form_factor, scale=scale, background=background)

    expected_intensity = np.einsum(
        "iq, jq, ijq->q",
        mock_form_factor.amplitude,
        mock_form_factor.amplitude,
        mock_structure_factor.number_weighted_partial_structure_factor,
    )
    expected_normalized_intensity = expected_intensity / mock_form_factor.average_square_forward_amplitude
    expected_result = scale * expected_normalized_intensity + background

    np.testing.assert_allclose(result, expected_result)


def test_measurable_intensity_with_background(mock_structure_factor, mock_form_factor):
    scale = 1.0
    background = 100.0

    result = measurable_intensity(mock_structure_factor, mock_form_factor, background=background)

    expected_intensity = np.einsum(
        "iq, jq, ijq->q",
        mock_form_factor.amplitude,
        mock_form_factor.amplitude,
        mock_structure_factor.number_weighted_partial_structure_factor,
    )
    expected_normalized_intensity = expected_intensity / mock_form_factor.average_square_forward_amplitude
    expected_result = scale * expected_normalized_intensity + background

    np.testing.assert_allclose(result, expected_result)


def test_measurable_structure_factor(mock_structure_factor, mock_form_factor):
    # Calculate expected measurable intensity
    expected_intensity = np.einsum(
        "iq, jq, ijq->q",
        mock_form_factor.amplitude,
        mock_form_factor.amplitude,
        mock_structure_factor.number_weighted_partial_structure_factor,
    )
    expected_normalized_intensity = expected_intensity / mock_form_factor.average_square_forward_amplitude
    expected_measurable_intensity = expected_normalized_intensity  # scale is 1.0 and background is 0.0 by default

    # Calculate expected measurable structure factor
    expected_measurable_structure_factor = expected_measurable_intensity / mock_form_factor.average_form_factor

    # Call the function under test
    result = measurable_structure_factor(mock_structure_factor, mock_form_factor)

    # Assert that the result is as expected
    np.testing.assert_allclose(result, expected_measurable_structure_factor)


def test_measurable_diffusion_coefficient(mock_form_factor):
    thermal_energy = 1.0
    viscosity = 1.0

    # Calculate expected weighted inverse radius
    weighted_inverse_radius = np.sum(
        mock_form_factor.mixture.number_fraction[:, np.newaxis]
        * mock_form_factor.amplitude**2
        / mock_form_factor.mixture.radius[:, np.newaxis],
        axis=0,
    )
    weighted_inverse_radius /= mock_form_factor.average_square_amplitude

    # Calculate expected prefactor
    prefactor = thermal_energy / (6.0 * np.pi * viscosity)

    # Calculate expected diffusion coefficient
    expected_diffusion_coefficient = prefactor * weighted_inverse_radius

    # Call the function under test
    result = measurable_diffusion_coefficient(mock_form_factor, thermal_energy, viscosity)

    # Assert that the result is as expected
    np.testing.assert_allclose(result, expected_diffusion_coefficient)


if __name__ == "__main__":
    pytest.main()
