# type: ignore
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from mixscatter.liquidstructure import LiquidStructure, PercusYevick, VerletWeis


@pytest.fixture
def mock_mixture(mocker):
    mock_mixture = mocker.Mock()
    mock_mixture.number_fraction = np.array([0.5, 0.5])
    mock_mixture.radius = np.array([1.0, 2.0])
    mock_mixture.number_of_components = 2
    mock_mixture.moment.side_effect = lambda order: np.sum(mock_mixture.number_fraction * mock_mixture.radius**order)

    return mock_mixture


def test_liquid_structure_initialization(mock_mixture):
    wavevector = [0.1, 0.2, 0.3]
    with pytest.raises(TypeError):
        _ = LiquidStructure(wavevector, mock_mixture)


def test_percus_yevick_initialization(mock_mixture):
    wavevector = [0.1, 0.2, 0.3]
    volume_fraction_total = 0.3
    py = PercusYevick(wavevector, mock_mixture, volume_fraction_total)
    assert np.allclose(py.wavevector, [0.1, 0.2, 0.3])
    assert py.mixture == mock_mixture
    assert py.volume_fraction_total == volume_fraction_total


def test_percus_yevick_partial_direct_correlation_function(mock_mixture):
    wavevector = [0.1, 0.2, 0.3]
    volume_fraction_total = 0.3
    py = PercusYevick(wavevector, mock_mixture, volume_fraction_total)
    c_ijq = py.partial_direct_correlation_function
    expected_c_ijq = np.array(
        [[[-1.316, -1.302, -1.278], [-5.417, -5.287, -5.074]], [[-5.417, -5.287, -5.074], [-19.565, -18.804, -17.587]]]
    )
    assert c_ijq.shape == (2, 2, 3)
    assert_array_almost_equal(c_ijq, expected_c_ijq, decimal=3)


def test_percus_yevick_number_weighted_partial_direct_correlation_function(mock_mixture):
    wavevector = [0.1, 0.2, 0.3]
    volume_fraction_total = 0.3
    py = PercusYevick(wavevector, mock_mixture, volume_fraction_total)
    c_weighted_ijq = py.number_weighted_partial_direct_correlation_function
    expected_c_weighted_ijq = np.array(
        [[[-0.658, -0.651, -0.639], [-2.709, -2.643, -2.537]], [[-2.709, -2.643, -2.537], [-9.783, -9.402, -8.793]]]
    )
    assert c_weighted_ijq.shape == (2, 2, 3)
    assert_array_almost_equal(c_weighted_ijq, expected_c_weighted_ijq, decimal=3)


def test_percus_yevick_partial_structure_factor(mock_mixture):
    wavevector = [0.1, 0.2, 0.3]
    volume_fraction_total = 0.3
    py = PercusYevick(wavevector, mock_mixture, volume_fraction_total)
    S_ijq = py.partial_structure_factor
    expected_S_ijq = np.array(
        [[[1.023, 1.021, 1.02], [-0.257, -0.260, -0.264]], [[-0.257, -0.260, -0.264], [0.157, 0.162, 0.170]]]
    )
    assert S_ijq.shape == (2, 2, 3)
    assert_array_almost_equal(S_ijq, expected_S_ijq, decimal=3)


def test_percus_yevick_number_weighted_partial_structure_factor(mock_mixture):
    wavevector = [0.1, 0.2, 0.3]
    volume_fraction_total = 0.3
    py = PercusYevick(wavevector, mock_mixture, volume_fraction_total)
    S_weighted_ijq = py.number_weighted_partial_structure_factor
    expected_S_weighted_ijq = np.array(
        [[[0.511, 0.511, 0.509], [-0.128, -0.130, -0.132]], [[-0.128, -0.130, -0.132], [0.079, 0.081, 0.085]]]
    )
    assert S_weighted_ijq.shape == (2, 2, 3)
    assert_array_almost_equal(S_weighted_ijq, expected_S_weighted_ijq, decimal=3)


def test_percus_yevick_average_structure_factor(mock_mixture):
    wavevector = [0.3, 1.9, 10.0]
    volume_fraction_total = 0.5
    py = PercusYevick(wavevector, mock_mixture, volume_fraction_total)
    S_avg_q = py.average_structure_factor
    assert S_avg_q.shape == (3,)
    assert_array_almost_equal(S_avg_q, np.array([0.344, 1.508, 0.994]), decimal=3)


def test_percus_yevick_compressibility_structure_factor(mock_mixture):
    wavevector = [0.3, 1.9, 10.0]
    volume_fraction_total = 0.5
    py = PercusYevick(wavevector, mock_mixture, volume_fraction_total)
    S_compressibility_q = py.compressibility_structure_factor
    assert S_compressibility_q.shape == (3,)
    assert_array_almost_equal(S_compressibility_q, np.array([0.021, 1.138, 0.994]), decimal=3)


def test_percus_yevick_verlet_weiss_initialization(mock_mixture):
    wavevector = [0.1, 0.2, 0.3]
    volume_fraction_total = 0.3
    pyvw = VerletWeis(wavevector, mock_mixture, volume_fraction_total)
    assert np.allclose(pyvw.wavevector, [0.1, 0.2, 0.3])
    assert pyvw.mixture == mock_mixture
    assert pyvw.volume_fraction_total == pytest.approx(volume_fraction_total * (1 - volume_fraction_total / 16))
    assert_array_almost_equal(pyvw.mixture.radius, np.array([0.994, 1.987]), decimal=3)


if __name__ == "__main__":
    pytest.main()
