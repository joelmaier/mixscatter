# type: ignore
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
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


@pytest.fixture
def wavevector():
    return np.array([0.1, 1.0, 10.0])


@pytest.fixture
def distance():
    return np.array([0.1, 0.5, 1.0, 10.0])


@pytest.fixture
def mock_mixture(mocker):
    mock = mocker.Mock()
    mock.number_fraction = np.array([0.5, 0.5])
    mock.radius = np.array([1.0, 2.0])
    return mock


def test_empty_profile(wavevector, distance):
    profile = EmptyProfile(0, 1)
    amplitude = profile.calculate_amplitude(wavevector)
    forward_amplitude = profile.calculate_forward_amplitude()
    layer_profile = profile.get_profile(distance)

    assert_array_almost_equal(amplitude, np.zeros_like(wavevector))
    assert forward_amplitude == 0.0
    assert_array_almost_equal(layer_profile, np.zeros_like(distance))


def test_empty_profile_invalid_radius():
    with pytest.raises(RuntimeError, match="'radius_inner' must be smaller than 'radius_outer'."):
        EmptyProfile(1, 0)


def test_constant_profile(wavevector, distance):
    profile = ConstantProfile(0, 1, 1.0)
    amplitude = profile.calculate_amplitude(wavevector)
    forward_amplitude = profile.calculate_forward_amplitude()
    layer_profile = profile.get_profile(distance)

    expected_forward_amplitude = 4.0 / 3.0 * np.pi * (1.0 ** 3 - 0.0 ** 3)

    assert_array_almost_equal(amplitude, np.array([4.185, 3.785, 0.099]), decimal=3)
    assert np.isclose(forward_amplitude, expected_forward_amplitude, atol=1e-5)
    assert_array_almost_equal(layer_profile, np.array([1.0, 1.0, 1.0, 0.0]), decimal=3)


def test_constant_profile_invalid_radius():
    with pytest.raises(RuntimeError, match="'radius_inner' must be smaller than 'radius_outer'."):
        ConstantProfile(1, 0, 1.0)


def test_linear_profile(wavevector, distance):
    profile = LinearProfile(0, 1, 1.0, 2.0)
    amplitude = profile.calculate_amplitude(wavevector)
    forward_amplitude = profile.calculate_forward_amplitude()
    layer_profile = profile.get_profile(distance)

    expected_forward_amplitude = 4.0 / 3.0 * np.pi * (1.0 ** 3 - 0.0 ** 3) + np.pi * (1.0 ** 4 - 0.0 ** 4)

    assert_array_almost_equal(amplitude, np.array([7.323, 6.590, 0.186]), decimal=3)
    assert np.isclose(forward_amplitude, expected_forward_amplitude, atol=1e-5)
    assert_array_almost_equal(layer_profile, np.array([1.1, 1.5, 2.0, 0.0]), decimal=3)


def test_linear_profile_invalid_radius():
    with pytest.raises(RuntimeError, match="'radius_inner' must be smaller than 'radius_outer'."):
        LinearProfile(1, 0, 1.0, 2.0)


def test_particle(wavevector, distance):
    profile1 = ConstantProfile(0, 1, 1.0)
    profile2 = LinearProfile(1, 2, 1.0, 2.0)
    particle = Particle([profile1, profile2])

    amplitude = particle.calculate_amplitude(wavevector)
    forward_amplitude = particle.calculate_forward_amplitude()
    form_factor = particle.calculate_form_factor(wavevector)
    layer_profile = particle.get_profile(distance)

    assert_array_almost_equal(amplitude, np.array([51.089, 32.012, -0.149]), decimal=3)
    assert np.isclose(forward_amplitude, 51.313, atol=1e-3)
    assert_array_almost_equal(form_factor, np.array([9.913e-01, 3.892e-01, 8.462e-06]), decimal=3)
    assert_array_almost_equal(layer_profile, np.array([1.0, 1.0, 2.0, 0.0]), decimal=3)


def test_particle_layer_connection():
    profile1 = EmptyProfile(0, 1)
    profile2 = ConstantProfile(1, 2, 1.0)
    profile3 = LinearProfile(2, 3, 5.0, 2.0)
    profile4 = EmptyProfile(3, 4)
    particle = Particle([profile1, profile2, profile3, profile4])

    assert_array_almost_equal(
        particle.get_profile([1.0, 2.0, 3.0]),
        np.array([1.0, 5.0, 0.0])
    )


def test_particle_builder():
    builder = ParticleBuilder()
    builder.add_layer(ConstantProfile(0, 1, 1.0))
    builder.add_layer(ConstantProfile(1, 2, 2.0))
    particle = builder.get_particle()

    assert len(particle.layers) == 2


def test_particle_builder_invalid_layer_connection():
    builder = ParticleBuilder()
    layer1 = ConstantProfile(0, 1, 1.0)
    layer2 = ConstantProfile(0.5, 2, 2.0)

    builder.add_layer(layer1)

    with pytest.raises(RuntimeError) as excinfo:
        builder.add_layer(layer2)

    expected_message = f"Added layer with inner radius {layer2.radius_inner} does not connect to previous layer's outer radius {layer1.radius_outer}."
    assert str(excinfo.value) == expected_message


def test_simple_sphere(wavevector, mock_mixture):
    model = SimpleSphere(wavevector, mock_mixture, 1.0)
    assert model.amplitude.shape == (2, len(wavevector))
    assert model.forward_amplitude.shape == (2,)


def test_simple_core_shell(wavevector, mock_mixture):
    model = SimpleCoreShell(wavevector, mock_mixture, 0.5, 1.0, 2.0)
    assert model.amplitude.shape == (2, len(wavevector))
    assert model.forward_amplitude.shape == (2,)


def test_simple_gradient(wavevector, mock_mixture):
    model = SimpleGradient(wavevector, mock_mixture, 1.0, 2.0)
    assert model.amplitude.shape == (2, len(wavevector))
    assert model.forward_amplitude.shape == (2,)


def test_scattering_model_properties(wavevector, mock_mixture):
    particles = [Particle([ConstantProfile(0, 1, 1.0)]), Particle([LinearProfile(0, 1, 1.0, 2.0)])]
    model = ScatteringModel(wavevector, mock_mixture, particles)

    assert model.amplitude.shape == (2, len(wavevector))
    assert model.forward_amplitude.shape == (2,)
    assert model.single_form_factor.shape == (2, len(wavevector))
    assert model.average_square_amplitude.shape == (len(wavevector),)
    assert np.isscalar(model.average_square_forward_amplitude)
    assert model.average_form_factor.shape == (len(wavevector),)
