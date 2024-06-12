The [`mixture`](../api/mixture_api.md#mixscatter.mixture) module provides tools to 
represent and manipulate mixtures of spherical particles with various size distributions.

## Creating a Mixture

To create a mixture, you can use the [`Mixture`](
../api/mixture_api.md#mixscatter.mixture.Mixture)
class directly or use one of the convenience classes provided for specific distributions. Here
are some examples:

### Binary Mixture

Manually create a 50:50 binary mixture of two species with radii 100 and 250, respectively:

```python
from mixscatter.mixture import Mixture

binary_mixture = Mixture(radius=[100, 250], number_fraction=[0.5, 0.5])
```

### Monodisperse System

Create a pseudo-mixture representing a monodisperse system with a radius of 500:

```python
from mixscatter.mixture import SingleComponent

monodisperse_system = SingleComponent(radius=500)
```

### Flory-Schulz Distribution

Specifically, a [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution)
(often termed Schulz/Flory/Zimm-distribution depending on the
scientific community), with probability density function

$$
P(R) = \frac{1}{\Gamma(Z+1)} \left(  \frac{Z+1}{\langle R \rangle} \right)^{Z+1} R^Z \exp\left(
-\frac{Z+1}{\langle R \rangle} R \right),
$$

with mean radius $\langle R \rangle$ and variance $\langle R^2 \rangle - \langle R \rangle^2
=\langle R \rangle^2/(Z+1)$. $Z$ is also called shape parameter.

Create a mixture approximating a Schulz-Flory distribution with a mean radius of 100 and a shape
parameter of 99:

```python
from mixscatter.mixture import FlorySchulzMixture

flory_mixture = FlorySchulzMixture(
    number_of_components=16, mean_radius=100, shape_parameter=99)
```

The approximation is based on [generalized Gauss-Laguerre quadrature](
https://en.wikipedia.org/wiki/Gauss-Laguerre_quadrature).

#### Gaussian Distribution
Gaussian or [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)
with probability density function

$$
P(R) = \frac{1}{\sigma\sqrt{(2\pi)}} \exp  \left[ - \dfrac{1}{2} \left(\frac{R-\langle R \rangle}
{\sigma}\right)
^2 \right],
$$

with mean radius $\langle R \rangle$ and standard deviation $\sigma$.

Create a mixture approximating a Gaussian distribution with a mean radius of 100 and a standard
deviation of 10:

```python
from mixscatter.mixture import GaussianMixture

gaussian_mixture = GaussianMixture(
    number_of_components=16, mean_radius=100, standard_deviation=10)
```

The approximation is based on [Gauss-Hermite quadrature](
https://en.wikipedia.org/wiki/Gauss-Hermite_quadrature).

#### Uniform Distribution

[Continuous uniform distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)
with probability density function

$$
P(R) = \left\{
\begin{array}{ll}
\dfrac{1}{b-a} & a \leq R \leq b \\
0 & \, \textrm{otherwise} \\
\end{array}
\right.
$$

with lower bound $a$ and upper bound $b$.

Create a mixture approximating a uniform distribution between radii 50 and 150:

```python
from mixscatter.mixture import UniformMixture

uniform_mixture = UniformMixture(
    number_of_components=16, lower_bound=50, upper_bound=150)
```

The approximation is based on [Gauss-Legendre quadrature](
https://en.wikipedia.org/wiki/Gauss-Legendre_quadrature).

### Accessing Mixture Properties

You can access various properties of a mixture, such as the mean radius and polydispersity:

```python
print(f"Mean radius: {flory_mixture.mean}")
print(f"Polydispersity: {flory_mixture.polydispersity}")
```

See the [API Reference](../api/mixture_api.md#mixscatter.mixture.Mixture) for a full list of 
available attributes and methods.
