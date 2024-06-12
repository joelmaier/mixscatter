import numpy as np
import matplotlib.pyplot as plt

from mixscatter.mixture import Mixture
from mixscatter.scatteringmodel import SimpleCoreShell
from mixscatter.liquidstructure import PercusYevick
from mixscatter import measurable_intensity, measurable_structure_factor, measurable_diffusion_coefficient

if __name__ == "__main__":
    plt.ion()
    plt.close("all")
    fig, ax = plt.subplots(3, 2, figsize=(6, 8), layout="constrained")

    # Initialize a particle mixture
    mixture = Mixture(radius=[100, 250], number_fraction=[0.4, 0.6])

    # Visualize mixture composition
    ax[0, 0].stem(mixture.radius, mixture.number_fraction)
    ax[0, 0].set_xlim(0, 300)
    ax[0, 0].set_ylim(-0.05, 1.05)
    ax[0, 0].set_xlabel("particle radius")
    ax[0, 0].set_ylabel("number fraction")

    # Provide a model for the optical properties of the system
    wavevector = np.linspace(1e-3, 7e-2, 1000)
    scattering_model = SimpleCoreShell(
        wavevector=wavevector, mixture=mixture, core_to_total_ratio=0.5, core_contrast=1.0, shell_contrast=0.5
    )

    # Visualize SLD profile
    distance = np.linspace(0, 350, 1000)
    for i, particle in enumerate(scattering_model.particles):
        profile = particle.get_profile(distance)
        ax[0, 1].plot(distance, profile, label=f"particle {i + 1}")
    ax[0, 1].set_xlim(0, 400)
    ax[0, 1].set_xlabel("distance from particle center")
    ax[0, 1].set_ylabel("scattering contrast")
    ax[0, 1].legend()

    # Visualize individual and average form factor(s)
    for i, form_factor in enumerate(scattering_model.single_form_factor):
        ax[1, 0].plot(wavevector, form_factor, label=f"particle {i + 1}")
    ax[1, 0].plot(wavevector, scattering_model.average_form_factor, label="average")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_ylim(1e-6, 3e0)
    ax[1, 0].legend()
    ax[1, 0].set_xlabel("wavevector")
    ax[1, 0].set_ylabel("form factor")

    # Provide a model for the liquid structure
    liquid_structure = PercusYevick(wavevector=wavevector, mixture=mixture, volume_fraction_total=0.45)

    # Calculate the scattered intensity of the system
    intensity = measurable_intensity(liquid_structure, scattering_model)
    ax[1, 1].plot(wavevector, intensity)
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel("wavevector")
    ax[1, 1].set_ylabel("intensity")

    # Calculate the experimentally obtainable, measurable structure factor
    structure_factor = measurable_structure_factor(liquid_structure, scattering_model)
    ax[2, 0].plot(wavevector, structure_factor)
    ax[2, 0].set_xlabel("wavevector")
    ax[2, 0].set_ylabel("structure factor")

    # Calculate the effective Stokes-Einstein diffusion coefficient which would be obtained from a
    # cumulant analysis in dynamic light scattering
    diffusion_coefficient = measurable_diffusion_coefficient(
        scattering_model, thermal_energy=1.0, viscosity=1.0 / (6.0 * np.pi)
    )
    # Visualize the apparent hydrodynamic radius, which is proportional to 1/diffusion_coefficient
    ax[2, 1].plot(wavevector, 1 / diffusion_coefficient)
    ax[2, 1].set_xlabel("wavevector")
    ax[2, 1].set_ylabel("apparent hydrodynamic radius")

    fig.savefig("simple_example_figure.png", dpi=300)
