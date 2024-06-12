The [`scatteringmodel`](../api/scatteringmodel_api.md#mixscatter.scatteringmodel) module 
provides tools to calculate scattering amplitudes and form factors of multicomponent systems 
consisting of spherical particles.

## Particles and Layers

Particles are represented by [`Particle`](
../api/scatteringmodel_api.md#mixscatter.scatteringmodel.Particle)
objects composed of layers, each represented by a specific scattering length density profile.
These layers can have different scattering properties, which are
defined by the [`LayerProfile`](
../api/scatteringmodel_api.md#mixscatter.scatteringmodel.LayerProfile)
interface. The module includes several predefined layer profiles:

- [**EmptyProfile**](
  ../api/scatteringmodel_api.md#mixscatter.scatteringmodel.EmptyProfile): Represents an
  empty layer.
- [**ConstantProfile**](
  ../api/scatteringmodel_api.md#mixscatter.scatteringmodel.ConstantProfile): Represents a
  layer with constant contrast.
- [**LinearProfile**](
  ../api/scatteringmodel_api.md#mixscatter.scatteringmodel.LinearProfile): Represents a
  layer with linearly varying contrast.

## ParticleBuilder

The [`ParticleBuilder`](
../api/scatteringmodel_api.md#mixscatter.scatteringmodel.ParticleBuilder) class is a
utility to help construct particles layer by layer. Once the desired layers are added, a
[`Particle`](
../api/scatteringmodel_api.md#mixscatter.scatteringmodel.Particle) instance can be created.

## ScatteringModel

The [`ScatteringModel`](
../api/scatteringmodel_api.md#mixscatter.scatteringmodel.ScatteringModel) class
calculates scattering properties for a list of particles. It can compute various properties such
as scattering amplitudes, forward scattering amplitudes, and form factors.

## Convenience Classes

There are several convenience classes provided to quickly create models for common scenarios
from a given [particle mixture](mixtures.md):

- [**SimpleSphere**](
  ../api/scatteringmodel_api.md#mixscatter.scatteringmodel.SimpleSphere): For a mixture of
  homogeneously scattering spheres.
- [**SimpleCoreShell**](
  ../api/scatteringmodel_api.md#mixscatter.scatteringmodel.SimpleCoreShell): For a
  mixture of core-shell particles with a common core-to-shell ratio.
- [**SimpleGradient**](
  ../api/scatteringmodel_api.md#mixscatter.scatteringmodel.SimpleGradient): For a mixture of
  particles displaying a linear gradient of the scattering length density.

## Example Usage

### Using `ParticleBuilder` to Construct Particles

The [`ParticleBuilder`](
../api/scatteringmodel_api.md#mixscatter.scatteringmodel.ParticleBuilder) class is
used to construct particles by adding layers. Here’s an example of how to use it:

```python
from mixscatter.scatteringmodel import (
    ParticleBuilder, ConstantProfile, LinearProfile
)

# Create a ParticleBuilder instance
builder = ParticleBuilder()

# Add a layer to the builder
builder.add_layer(ConstantProfile(0, 10, 1.0))

# Add another layer to the builder
builder.add_layer(LinearProfile(10, 20, 1.0, 0.0))

# Get the constructed particle
particle = builder.get_particle()
```

### Using `ScatteringModel`

The [`ScatteringModel`](
../api/scatteringmodel_api.md#mixscatter.scatteringmodel.ScatteringModel) class
calculates the scattering properties from a list of particles. Here’s an example of how to use it:

```python
import numpy as np
from mixscatter.scatteringmodel import (
    ScatteringModel, ParticleBuilder, ConstantProfile
)
from mixscatter.mixture import Mixture

mixture = Mixture(radius=[1.0, 2.0], number_fraction=[0.5, 0.5])

# Create particles using ParticleBuilder
builder = ParticleBuilder()
particles = [
    builder.add_layer(ConstantProfile(0, radius, 1.0)).pop_particle()
    for radius in mixture.radius
    ]

# Create a ScatteringModel instance
wavevector = np.linspace(0.01, 1.0, 100)
model = ScatteringModel(wavevector, mixture, particles)

# Calculate the average form factor
form_factor = model.average_form_factor
```

A `ScatteringModel` can contain almost any conceivable combination of particles with 
totally different optical properties. This flexibility makes the tool particularly powerful.

### Convenience Classes

The module includes several convenience classes to quickly create models for common particle types:

#### SimpleSphere

```python
from mixscatter.scatteringmodel import SimpleSphere

# Create a simple sphere model
model = SimpleSphere(wavevector, mixture, contrast=1.0)
form_factor = model.average_form_factor
```

#### SimpleCoreShell

```python
from mixscatter.scatteringmodel import SimpleCoreShell

# Create a core-shell model
model = SimpleCoreShell(
    wavevector,
    mixture,
    core_to_total_ratio=0.5,
    core_contrast=1.0,
    shell_contrast=0.5
    )
form_factor = model.average_form_factor
```

#### SimpleGradient

```python
from mixscatter.scatteringmodel import SimpleGradient

# Create a gradient profile model
model = SimpleGradient(
    wavevector, mixture, center_contrast=1.0, boundary_contrast=0.5
    )
form_factor = model.average_form_factor
```

### Implementing Your Own Model

To implement your own layer profile or scattering model from scratch, you can extend the
[`LayerProfile`](
../api/scatteringmodel_api.md#mixscatter.scatteringmodel.LayerProfile) and the
[`ScatteringModel`](
../api/scatteringmodel_api.md#mixscatter.scatteringmodel.ScatteringModel) class. Here’s a
basic example:

```python
from mixscatter.scatteringmodel import (
    ScatteringModel, ParticleBuilder, LayerProfile
    )
import numpy as np

class CustomProfile(LayerProfile):
    def __init__(self, radius_inner, radius_outer, custom_param):
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        self.custom_param = custom_param

    def calculate_amplitude(self, wavevector):
        wavevector = np.asarray(wavevector)
        # Custom amplitude calculation logic
        amplitude = ...  # Replace with actual calculation
        return amplitude

    def calculate_forward_amplitude(self):
        # Custom forward amplitude calculation logic
        forward_amplitude = ...  # Replace with actual calculation
        return forward_amplitude

    def get_profile(self, distance):
        distance = np.asarray(distance)
        # Custom profile calculation logic
        profile = ... # Replace with actual calculation
        return profile

class CustomModel(ScatteringModel):
    def __init__(self, wavevector, mixture, custom_param):
        particles = []
        particle_builder = ParticleBuilder()
        for radius in mixture.radius:
            particle = (
            particle_builder
            .add_layer(CustomProfile(0, radius, custom_param))
            .pop_particle()
            )
            particles.append(particle)
        super().__init__(wavevector, mixture, particles)

# Using the custom model
custom_param = 1.0
model = CustomModel(wavevector, mixture, custom_param)
form_factor = model.average_form_factor
```
