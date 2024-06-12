---
hide:
  - navigation
  - toc
---

Most of this information can be found in more detail in the work of 
P. Salgi and R. Rajagopalan, "Polydispersity in colloids: implications to 
static structure and scattering", [Adv. Colloid Interface Sci. 43, 169-288 (1993)](
https://doi.org/10.1016/0001-8686(93)80017-6). A basic understanding of the principles of 
[small-angle scattering](
https://en.wikipedia.org/wiki/Small-angle_scattering) and 
[light scattering by particles](
https://en.wikipedia.org/wiki/Light_scattering_by_particles)
is recommended.

## Definitions and Naming Conventions

The scattered intensity $I(Q)$ in dependence of the scattering wavevector $Q$ of a
multicomponent system of spherical particles with $n$ distinct species can be written as

$$
I(Q) \propto \sum_{\alpha, \beta=1}^{n} (x_\alpha x_\beta)^{1/2} \,
    F_\alpha(Q) \, F_\beta(Q) \, S_{\alpha \beta}(Q),
$$

where $x_\alpha$ is the number fraction of the species $\alpha$.

The scattering amplitude $F_\alpha(Q)$ of the species $\alpha$ is the Fourier transform of the
scattering contrast $\Delta\rho_\alpha(r)$ of the particle. Because the particles are
spherically symmetric, the 3D Fourier transform can be formulated as a 1D Fourier-Bessel integral:

$$
F_\alpha(Q) = 4\pi \int\limits_{0}^{\infty} \Delta\rho_\alpha(r) \, r^2 \,
\dfrac{\sin(Q \, r)}{Q r} \,\mathrm{d} r.
$$

An important quantity is the size average of the squared amplitude

$$
\overline{F^2}(Q) = \sum_{\alpha=1}^{n} x_\alpha \, F^2_\alpha(Q).
$$

The size-averaged form factor can then be expressed as

$$
P(Q) = \dfrac{\overline{F^2}(Q)}{\overline{F^2}(0)}.
$$

The partial structure factors $S_{\alpha \beta}(Q)$ describe the interparticle correlations
between the particles of the species $\alpha$ and the species $\beta$. They form the matrix
$\mathbf{S}$, which is related to the matrix of weighted direct
correlation functions $\mathbf{\tilde{c}}$ with elements $\tilde{c}_{\alpha\beta} = \sqrt{x_i
x_j} \, c_{\alpha\beta}$ by the [Ornstein-Zernike equation](
https://en.wikipedia.org/wiki/Ornstein-Zernike_equation):

$$
\mathbf{S} = [\mathbf{1} - \rho \mathbf{\tilde{c}}]^{-1},
$$

where $\rho$ is the total number density. The partial structure factors defined here follow the 
property $S_{\alpha\beta}(Q\to\infty) = \delta_{\alpha\beta}$.

## Effective Structure Factors

There are multiple ways to define single effective, structure factors:

- The *measurable* structure factor

    $$
    S_{\mathrm{M}}(Q) = \left[ \overline{F^2}(Q) \right]^{-1} \sum_{\alpha, \beta=1}^{n}
        (x_\alpha x_\beta)^{1/2} \, F_\alpha(Q) \, F_\beta(Q) \, S_{\alpha \beta}(Q),
    $$

    which is simply the intensity of a suspension divided by the intensity of a non-interacting
    suspension with the same scattering properties.

- The *average number-number* structure factor

    $$
    S_{\mathrm{NN}}(Q) = \sum_{\alpha, \beta=1}^{n} (x_\alpha x_\beta)^{1/2} \, S_{\alpha \beta}(Q),
    $$

    which is just the sum of all partial structure factors weighted with the size distribution.
    This effective structure factor describes the overall correlation between all existing
    particles.

- The *"compressibility"* structure factor

    $$
    S_{\mathrm{\kappa}}(Q) = \dfrac{1}{\sum_{\alpha, \beta=1}^{n} x_\alpha x_\beta \,
      S_{\alpha\beta}^{-1}(Q)},
    $$

    where $S_{\alpha\beta}^{-1}(Q)$ denotes the $\alpha\beta$ element of the inverse of the partial
    structure factor matrix. According to the Kirkwood-Buff theory of solutions, the isothermal
    compressibility $\kappa_{\mathrm{T}}$ is related to the zero-$Q$ limit of the partial
    structure factors by

    $$
    \rho k_\mathrm B T \kappa_{\mathrm{T}} = \dfrac{1}{\sum_{\alpha, \beta=1}^{n} x_\alpha x_\beta \,
      S_{\alpha\beta}^{-1}(0)}.
    $$

    The compressibility structure factor extends this definition to finite $Q$ and the relation

    $$
    S_{\mathrm{\kappa}}(Q\to0) =  \rho k_\mathrm B T \kappa_{\mathrm{T}}
    $$

    is satisfied.

## Apparent Diffusion Coefficient

Of interest in the context of dynamic light scattering is the apparent Stokes-Einstein diffusion 
coefficient, which for mixtures of particles depends on the scattering amplitudes and also 
changes with the wavevector:

$$
\overline{D_0}(Q) = \left[ \overline{F^2}(Q) \right]^{-1}
\sum_{\alpha=1}^{n} x_\alpha \, F^2_\alpha(Q) D_{0, \alpha},
$$

with

$$
D_{0, \alpha} = \dfrac{k_{\mathrm{B}} T}{6 \pi \eta_0 R_{\mathrm{h}, \alpha}},
$$

where $k_{\mathrm{B}} T$ indicates the thermal energy, $\eta_0$ the
viscosity of the suspension medium and $R_{\mathrm{h}, \alpha}$ the hydrodynamic radius of the 
particles of species $\alpha$.

An apparent hydrodynamic radius can then be defined as 

$$
R_{\mathrm{h, app}}(Q) = \dfrac{k_{\mathrm{B}} T}{6 \pi \eta_0 \overline{D_0}(Q)}.
$$