The [`liquidstructure`](
../api/liquidstructure_api.md#mixscatter.liquidstructure) module provides tools to
model the liquid structure properties of multicomponent systems of spherical particles. The
primary functionality revolves around calculating various structure factors using different
approximations. Currently, the following model classes are implemented:

- [**LiquidStructure**](
  ../api/liquidstructure_api.md#mixscatter.liquidstructure.LiquidStructure): An
  abstract base class representing the liquid structure properties of a multicomponent system.
- [**PercusYevick**](
  ../api/liquidstructure_api.md#mixscatter.liquidstructure.PercusYevick): A concrete class
  implementing the [hard-sphere potential in the Percus-Yevick approximation](
  https://en.wikipedia.org/wiki/Percus-Yevick_approximation).
- [**VerletWeis**](
  ../api/liquidstructure_api.md#mixscatter.liquidstructure.VerletWeis): A subclass of
  `PercusYevick` that employs the Verlet-Weiss correction.

## Example usage

### Initialize a model

Create an instance of [`PercusYevick`](
../api/liquidstructure_api.md#mixscatter.liquidstructure.PercusYevick)
with the wavevector, mixture, and total volume fraction.

```python
import numpy as np
from mixscatter.liquidstructure import PercusYevick
from mixscatter.mixture import Mixture

wavevector = np.linspace(0.01, 1, 100)
mixture = Mixture([5.0, 10.0], [0.5, 0.5])
volume_fraction_total = 0.3

# Initialize the PercusYevick model
py_model = PercusYevick(wavevector, mixture, volume_fraction_total)
```

### Calculate Structure Factors

Use the properties of the [`PercusYevick`](
../api/liquidstructure_api.md#mixscatter.liquidstructure.PercusYevick) class to calculate 
various structure factors and related quantities:

```python
# Partial direct correlation function
partial_direct_correlation = py_model.partial_direct_correlation_function

# Number-weighted partial direct correlation function
number_weighted_direct_correlation = py_model.number_weighted_partial_direct_correlation_function

# Partial structure factor
partial_structure_factor = py_model.partial_structure_factor

# Number-weighted partial structure factor
number_weighted_structure_factor = py_model.number_weighted_partial_structure_factor

# Average structure factor
average_structure_factor = py_model.average_structure_factor

# Compressibility structure factor
compressibility_structure_factor = py_model.compressibility_structure_factor
```

### Provide external structure factors

Say you have an external dataset of partial structure factors $S_{\alpha\beta}(Q)$ as a function of 
the wavevector $Q$, for example, from a computer simulation. You can easily use these structure 
factors in **mixscatter**.

To be used in the functions [`measurable_intensity`](
../api/core_api.md#mixscatter.measurable_intensity) and
[`measurable_structure_factor`](
../api/core_api.md#mixscatter.measurable_structure_factor),
you have to provide an object with an interface which follows the 
[`LiquidStructureLike`](
../api/core_api.md#mixscatter.LiquidStructureLike)
protocol.
```python
class LiquidStructureLike(Protocol):
    """A protocol which defines an interface for a class which behaves like LiquidStructure"""
    @property
    def number_weighted_partial_structure_factor(self) -> NDArray[np.float64]: ...
```

The only thing this object needs is a property named `number_weighted_partial_structure_factor` 
which must be a **numpy array** of type `float` with the shape (N, N, M), where N is the number 
of components and M the size of the wavevector array. 'Number weighted partial structure factor' 
means that the partial structure factors follow the convention $S_{\alpha\beta}(Q\to\infty) = 
x_\alpha \delta_{\alpha\beta}$, where $x_\alpha$ is the number fraction and $\delta_{\alpha\beta}$
is the [Kronecker symbol](https://en.wikipedia.org/wiki/Kronecker_symbol).

This can be as simple as 
```python
my_structure_factors = ... # Assume a numpy array of correct type, shape, and values
    
class MyLiquidStructure:
    def __init__(self, structure_factor_input):
        self.number_weighted_partial_structure_factor = structure_factor_input

my_liquid_structure = MyLiquidStructure(my_structure_factors)


# Calculate scattered intensity
intensity = measurable_intensity(my_liquid_structure, scattering_model)
```

If your partial structure factors follow the convention $\tilde{S}_{\alpha\beta}(Q\to\infty) = 
\delta_{\alpha\beta}$ and are not already weighted by the number fraction, an easy way to 
convert conventions is by using [`numpy.einsum`](
https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) and calculating 
$S_{\alpha\beta}(Q) = (x_\alpha x_\beta)^{1/2} \tilde{S}_{\alpha\beta}(Q)$:
```python
unweighted_structure_factors = ... # numpy array of shape (N, N, M)
number_fractions = ... # numpy array of shape (N, )

number_weighted_structure_factors = np.einsum(
  "i, j, ijq->ijq",
  np.sqrt(number_fractions),
  np.sqrt(number_fractions),
  unweighted_structure_factors
)
```