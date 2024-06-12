Welcome to the **mixscatter** Getting Started Guide! This guide will walk you through the basic
functionalities of the package, showcasing its power and versatility in analyzing scattering
experiments.

## Why mixscatter?

**mixscatter** is a comprehensive solution designed to tackle the complexities of analyzing 
multi-component particle mixtures in scattering experiments. What sets **mixscatter** apart are 
its three foundational concepts:

1. **Mixture Composition**: With **mixscatter**, you can define the composition of your 
   particle mixture with precision. Whether you're working with two components or ten, specifying 
   their sizes and relative abundances is straightforward. You can also create mixtures which 
   mimic the properties of continuous size distributions. 

2. **Scattering Models**: **mixscatter** enables you to define detailed scattering length 
   density profiles for each particle species independently, influencing single-particle scattering 
   amplitudes and overall form factors. Its flexible model-building capabilities allow for both 
   predefined simple models and more complex custom particles.

3. **Liquid Structure**: **mixscatter** offers solutions for the analytic calculation
   of partial structure factors. It also provides a seamless interface for incorporating externally 
   generated structure factors, ensuring versatility and adaptability.

## First Steps

Follow these steps to get to know the core functions of **mixscatter**:

### Step 1: Import Necessary Modules

```python
import mixscatter as ms
import numpy as np
```

### Step 2: Create a Mixture

Define the composition of your system using the `Mixture` object:

```python
radii = [100, 200]
number_fractions = [0.2, 0.8]
mixture = ms.mixture.Mixture(radii, number_fractions)
```

### Step 3: Define the Wavevector

Set up the scattering wavevector grid for calculations:

```python
wavevector = np.linspace(0.005, 0.05, 100)
```

### Step 4: Create a Scattering Model

Construct a scattering model to represent your system:

```python
scattering_model = ms.scatteringmodel.SimpleSphere(
    wavevector, mixture, contrast=1.0
)
```

### Step 5: Define the Liquid Structure

Specify the liquid structure of your system:

```python
liquid_structure = ms.liquidstructure.PercusYevick(
    wavevector, mixture, volume_fraction_total=0.3
)
```

### Step 6: Calculate Measurable Quantities

Now, let's compute some measurable quantities:

#### Measurable Scattered Intensity

```python
intensity = ms.measurable_intensity(
    liquid_structure, scattering_model, scale=1e5, background=1e3
)
```

#### Measurable Structure Factor

```python
structure_factor = ms.measurable_structure_factor(
    liquid_structure, scattering_model
)
```

#### Measurable Diffusion Coefficient

```python
thermal_energy = 1.0
viscosity = 1.0
diffusion_coefficient = ms.measurable_diffusion_coefficient(
    scattering_model, thermal_energy=1.0, viscosity=1.0
)
```

## A Note on Units of Measurement

**mixscatter** doesn't enforce any particular unit system. You have the flexibility to choose units of length, time, and energy that best suit your work. Just remember to maintain consistency throughout your analysis.
