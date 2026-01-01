---
title: "Active Brownian Particles in 2D: Multiple Scale Analysis"
date: 2025-01-01
tags: [active-matter, physics, homogenization, stochastic-processes]
description: "Deriving the effective equation for Active Brownian Particles using multiple scale expansion"
---

## Introduction

Active matter systems, from swimming bacteria to artificial microswimmers, exhibit fascinating non-equilibrium dynamics that emerge from energy consumption at the particle level. A fundamental challenge is bridging microscopic fluctuations with macroscopic collective behavior. How do we rigorously connect particle-level dynamics to the emergent transport properties we observe at larger scales?

The answer lies in **homogenization theory** and **multiple scale analysis**, powerful asymptotic techniques that systematically derive effective macroscopic equations from microscopic dynamics. In this post, I demonstrate this approach using the Active Brownian Particle (ABP) in 2D, deriving how self-propulsion and rotational diffusion produce enhanced effective diffusion at the macroscale.

While this is a pedagogical example, the methodology extends to far more complex systems. In my research on bioparticles in premelting ice ([Vachier & Wettlaufer, Phys. Rev. E, 2022](https://doi.org/10.1103/PhysRevE.105.024601); [Frontiers in Physics, 2022](https://doi.org/10.3389/fphy.2022.904836)), I applied these same techniques to systems where thermal regelation, chemotaxis, and bio-enhanced premelting all compete—phenomena critical for ice core dating and extremophile survival in glaciers.

---

## Method

We study the motion of one active Brownian particle with an active velocity $\tilde{v}_0$, in two dimension under an external force $\tilde{F}$ in the $\tilde{x}$-direction, considered here constant for simplicity.
The evolution of the particle's position $\tilde{\bm{r}}$ and its orientation are described by two overdamped Langevin equations

$$\frac{d\tilde{\bm{r}}}{d\tilde{t}} = \tilde{v}_0\bm{n} + \tilde{F}\bm{\hat{x}} + \sqrt{2\tilde{D}}\tilde{\bm{\xi}}, \tag{1}$$

$$\frac{d\phi}{d\tilde{t}} = \sqrt{2\tilde{D}_r}\tilde{\xi}_r, \tag{2}$$

where $\tilde{\bm{r}} = (\tilde{x},\tilde{y})^T$, $\bm{n} = (\cos(\phi),\sin(\phi))^{T}$ and $\bm{\hat{x}}=(1,0)^T$.
The random fluctuations are given by zero mean Gaussian white noise process $\langle \tilde{\xi}_i(\tilde{t}')\tilde{\xi}_j(\tilde{t}) \rangle = \delta_{ij} \delta(\tilde{t}'-\tilde{t})$ and $\langle \tilde{\xi}_{r_i}(\tilde{t}')\tilde{\xi}_{r_j}(\tilde{t}) \rangle = \delta_{ij} \delta(\tilde{t}'-\tilde{t})$. 
From Eqs. (1) - (2), we derive the associated Fokker-Planck equation

$$\frac{\partial}{\partial\tilde{t}}P(\tilde{\bm{r}},\phi,\tilde{t}) = -\tilde{v}_0\bm{n}\cdot \nabla_{\tilde{\bm{r}}}P(\tilde{\bm{r}},\phi,\tilde{t}) - \frac{\partial}{\partial \tilde{x}}\left[ \tilde{F} P(\tilde{\bm{r}},\phi,\tilde{t}) \right]+ \tilde{D}\nabla^2_{\tilde{\bm{r}}}P(\tilde{\bm{r}},\phi,\tilde{t}) + \tilde{D}_r \frac{\partial^2}{\partial \phi^2} P(\tilde{\bm{r}},\phi,\tilde{t}). \tag{3}$$

The two characteristic length scales $l$ and $L$ give the micro- and macroscale respectively. The microscale $l$ gives the cell size and is assumed to be spatially periodic. Taking the ratio of these two length scales, we define a small parameter $\epsilon$ such as

$$\epsilon = \frac{l}{L}. \tag{4}$$

We introduce the following dimensionless quantities 

$$\bm{R} = \frac{\tilde{\bm{r}}}{L}, \quad T = \frac{\tilde{t} D_c}{L^2}, \quad D = \frac{\tilde{D}}{D_c}, \quad D_r = \frac{\tilde{D}_r}{D_{rc}}, \quad F = \frac{\tilde{F}}{F_c}, \quad v_0 = \frac{\tilde{v}_0}{v_{oc}}, \tag{5}$$

where  $v_{oc}$ is the characteristic active velocity, $F_c$ and $D_c$ are the characteristic velocity associated to the external force and diffusity, respectively. Using the previous dimensionless quantities and $P = P(\bm{R},\phi,T)$ to lighten the notations, Eq. (3) becomes

$$\frac{\partial}{\partial T} P = P_e^L v_o \bm{n} \cdot \nabla_{\bm{R}}P - P_F^L\frac{\partial}{\partial X} \left[ F P \right] + D \nabla^2_{\bm{R}}P + P_r^L D_r \frac{\partial^2}{\partial \phi^2}P, \tag{6}$$

in which the following dimensionless numbers appear

$$P_e^L = \frac{v_{oc}L}{D_c}, \quad P_F^L = \frac{F_c L}{D_c}, \quad P_r^L = \frac{D_{rc} L^2}{D_c}. \tag{7}$$

The first two numbers $P_a^L$ and $P_e^L$ are the Péclet numbers associated to the active velocity and external force, respectively. In addition to the rotational time $\tau_r$ ($D_r \propto 1/\tau_r$), six characteristic time scales are identified: $t^{\text{diff}}_l=l^2/D_c$, $t^{\text{F}}_l=l/F_c$, $t^{\text{act}}_l=l/v_{oc}$, $t^{\text{diff}}_L=L^2/D_c$, $t^{\text{F}}_L=L/F_c$ and $t^{\text{act}}_L=L/v_{oc}$ associated with microscopic (macroscopic) diffusion and advections either given by the external force and the activity on the microscopic length scale $l$ (macroscopic length scale $L$). As a result, the ratio of characteristic time for diffusion and advections are the Péclet numbers

$$P_e^{L,l} = \frac{t^{\text{diff}}_{L,l}}{t^{\text{act}}_{L,l}}, \quad P_F^{L,l} = \frac{t^{\text{diff}}_{L,l}}{t^{\text{F}}_{L,l}}. \tag{8}$$

## Results

The potential ($t^{\text{F}}_L = L/F_c$) and the active velocity ($t^{\text{act}}_L=L/v_{oc}$) will dominate the diffusion ($t^{\text{diff}}_L=L^2/D_c$) at the macroscopic scale. As a result $P_e^L = \mathcal{O}(1/\epsilon)$ and $P_F^L = \mathcal{O}(1/\epsilon)$. The rotational time will overwhelmingly dominate diffusion at the macroscopic scale, giving $P_r^L = \mathcal{O}(1/\epsilon^2)$. Equation (6) becomes

$$\frac{\partial}{\partial T} P = \frac{1}{\epsilon} v_o \bm{n} \cdot \nabla_{\bm{R}}P - \frac{1}{\epsilon}\frac{\partial}{\partial X} \left[ F P \right] + D \nabla^2_{\bm{R}}P + \frac{1}{\epsilon^2} D_r \frac{\partial^2}{\partial \phi^2}P. \tag{9}$$

We introduce a dimensionless microscopic length $\bm{r}=\tilde{\bm{r}}/l$ and time $\theta=\tilde{t}D_c/l^2$. Moreover, by looking at the six characteristic time scales, we also need to introduce an intermediate time $t = \tilde{t} D_c/l L$. As a result, we have the following scales stretch

$$\bm{R} = \epsilon \bm{r}, \quad T = \epsilon t, \quad T = \epsilon^2\theta. \tag{10}$$

Taking the macroscopic description 

$$\nabla_{\bm{R}}= \nabla_{\bm{R}} + \frac{1}{\epsilon}\nabla_{\bm{r}},$$

$$\frac{\partial}{\partial T} = \frac{\partial}{\partial T} + \frac{1}{\epsilon}\frac{\partial}{\partial t} + \frac{1}{\epsilon^2}\frac{\partial}{\partial \theta}, \tag{11}$$

and using a power series ansatz 

$$P = P^0 + \epsilon P^1 + \epsilon^2 P^2, \tag{12}$$

we derive a system of equation at each order in $\epsilon$, which are

$$
\begin{align*}
\mathcal{O}(\frac{1}{\epsilon^2}): \quad & \mathcal{L}P^0 = 0, \tag{13}\\
\mathcal{O}(\frac{1}{\epsilon}): \quad & \mathcal{L}P^1 = -\frac{\partial}{\partial t}P^0-v_o \bm{n}\cdot \nabla_{\bm{R}}P^0 - \frac{\partial}{\partial X}\left[ F P^0\right] + 2 D \nabla_{\bm{r}}\cdot\nabla_{\bm{R}} P^0, \tag{14}\\
\mathcal{O}(1): \quad & \mathcal{L}P^2 = -\frac{\partial}{\partial T}P^0-\frac{\partial}{\partial t}P^1-v_o \bm{n}\cdot \nabla_{\bm{R}} P^1 - \frac{\partial}{\partial X}\left[ F P^1\right] + 2D\nabla_{\bm{r}}\cdot\nabla_{\bm{R}} P^1 + D \nabla^2_{\bm{R}}P^0, \tag{15}
\end{align*}
$$

where $\mathcal{L}=\mathcal{M}-\mathcal{Q}$, with $\mathcal{M}=v_0\bm{n}\cdot\nabla_{\bm{r}} + \frac{\partial}{\partial x}F - D\nabla^2_{\bm{r}}$ and $\mathcal{Q}=D_r \frac{\partial^2}{\partial \phi^2}$, where the microscopic time contribution is neglected. In order to solve Eqs. (13)-(15), we follow the steps described in [5, 7]. The solution of the leading order Eq. (13) is derived by making the following product ansatz

$$P^0(\bm{r},\bm{R},\phi,t,T)=w(\bm{r},\phi)\rho^0(\bm{R},t,T). \tag{16}$$

We integrate by parts over the microscopic variables $\bm{r}$ and $\phi$, and use the periodic boundary conditions, to obtain the so-called weak formulation of the leading order equation. To ensure the existence and uniqueness of $P^0$, we use the Lax-Milgram theorem, also known as the solvability condition or the Fredholm alternative. The external force $F$ is constant and as a result, $w$ is constant over the period: $w(\bm{r},\phi)=w(\phi)=\text{const}$.

The solvability condition for $\mathcal{O}(1/\epsilon)$  equation is 

$$\int\limits_0^{2\pi}\frac{d\phi}{2\pi}\left(-\frac{\partial}{\partial t} P^0 - v_o \bm{n}\cdot \nabla_{\bm{R}}P^0 - \frac{\partial}{\partial X} \left[ F P^0 \right] \right) = 0$$

which depends on the leading order result, $P^0$, from which we find

$$\frac{\partial}{\partial t}\rho^0 = -\frac{\partial}{\partial X}\left[ F\rho^0 \right]$$

and the $\mathcal{O}(1/\epsilon)$ equation becomes

$$\mathcal{L}P^1 = - w v_{o} \bm{n}\cdot\nabla_{\bm{R}}\rho^0$$

The solution of Eq. (19) is

$$P^1= - w v_{o}\bm{\alpha} \cdot \nabla_{\bm{R}}\rho^0$$

where $\bm{\alpha} = (\alpha_1(X) \cos(\phi),\alpha_2(Y) \sin(\phi))^T$, after which we find that

$$\bm{\alpha} = \frac{1}{D_r}\bm{n}$$

Substitution of $P^1$ into the $\mathcal{O}(1)$ equation and using the solvability condition 

$$\int\limits_0^{2\pi}\frac{d\phi}{2\pi}\left( -\frac{\partial}{\partial T}P^0-\frac{\partial}{\partial t}P^1-v_o \bm{n}\cdot \nabla_{\bm{R}} P^1 - \frac{\partial}{\partial X}\left[ F P^1\right] + D \nabla^2_{\bm{R}}P^0  \right) = 0$$

$$\int\limits_0^{2\pi}\frac{d\phi}{2\pi}\left( -w\frac{\partial}{\partial T}\rho^0 + w \frac{v_o^2}{D_r} \bm{n}^2\cdot \nabla^2_{\bm{R}} \rho^0 + w D \nabla^2_{\bm{R}}\rho^0  \right) = 0$$

we obtain

$$\frac{\partial}{\partial T}\rho^0 = \left( D + \frac{v_o^2}{2 D_r} \right) \nabla^2_{\bm{R}} \rho^0$$

Remembering

$$\frac{\partial}{\partial T} \rho^0 = \frac{\partial}{\partial T} \rho^0 + \frac{1}{\epsilon} \frac{\partial}{\partial t} \rho^0$$

Equation (23) becomes

$$\frac{\partial}{\partial T} \rho^0 =  \frac{1}{\epsilon} \frac{\partial}{\partial t} \rho^0 + \left( D + \frac{v_o^2}{2 D_r} \right) \nabla^2_{\bm{R}} \rho^0$$

$$\frac{\partial}{\partial T} \rho^0 = - \frac{1}{\epsilon} \frac{\partial}{\partial X} \left[ F \rho^0 \right] + \left( D + \frac{v_o^2}{2 D_r} \right) \nabla^2_{\bm{R}} \rho^0$$

In dimensional form, we have

$$\frac{\partial}{\partial \tilde{t}} \rho (\tilde{\bm{r}},\tilde{t}) = -\frac{1}{\epsilon} \frac{D_c}{L F_c} \frac{\partial}{\partial \tilde{x}}\left[ \tilde{F} \rho(\tilde{\bm{r}},\tilde{t}) \right] + \left( \tilde{D} + \frac{D_c D_{rc} }{v_{oc}^2} \frac{v_o^2}{2 D_r} \right) \nabla^2_{\tilde{\bm{r}}} \rho(\tilde{\bm{r}},\tilde{t})$$

with $D_c/L F_c = \mathcal{O}(\epsilon)$ and $D_c D_{rc} / v_{oc}^2=\mathcal{O}(1)$ leading to

$$\boxed{\frac{\partial}{\partial \tilde{t}} \rho (\tilde{\bm{r}},\tilde{t}) = -\frac{\partial}{\partial\tilde{x}}\left[\tilde{F} \rho(\tilde{\bm{r}},\tilde{t}) \right] + \left( \tilde{D} + \frac{\tilde{v}_0^2}{2 \tilde{D}_r} \right) \nabla^2_{\tilde{\bm{r}}}\rho(\tilde{\bm{r}},\tilde{t})}$$

The effective diffusion coefficient is:

$$\tilde{D}_{\text{eff}} = \tilde{D} + \frac{\tilde{v}_0^2}{2\tilde{D}_r}$$

This equation defines an effective equation derived using a homogenization procedure valid to the leading order $P^0$.

---

## Conclusion

Through multiple scale analysis, we have rigorously derived how microscopic self-propulsion and rotational diffusion produce enhanced macroscopic diffusion in Active Brownian Particles. The active contribution ṽ₀²/(2D̃ᵣ) emerges naturally from angular averaging—the factor of 1/2 reflecting the geometry of 2D rotational diffusion.

This derivation demonstrates the power of homogenization theory: systematic separation of scales, power series expansion, and solvability conditions at each order yield effective equations from first principles without phenomenological assumptions. The beauty lies in how a few lines of calculation connect microscopic stochastic dynamics to macroscopic deterministic transport.

The same mathematical framework extends to far richer physics. In my work on bioparticles in ice, thermal regelation competes with chemotaxis and bio-enhanced premelting, a multi-physics problem where homogenization reveals how biological activity modifies transport in extreme environments. These insights matter for ice core dating (where diffusion affects proxy redistribution), extremophile survival strategies, and biosignature detection in extraterrestrial ice.

Whether studying bacteria in aqueous environments, microswimmers in complex fluids, or microorganisms in glacial ice, multiple scale analysis provides a principled path from microscopic rules to emergent macroscopic behavior. The ABP derivation shown here is the gateway to this broader methodology.

**Code and resources:** Numerical implementations and extensions to more complex systems are available in my [GitHub repository](https://github.com/jvachier/Particle-in-harsh-environment-share).

## References

1. Bensoussan, A., Lions, J.-L., & Papanicolaou, G. (2011). *Asymptotic analysis for periodic structures*. Vol. 374. American Mathematical Society.

2. Sánchez-Palencia, E. (1980). Non-homogeneous media and vibration theory. *Lecture Notes in Physics*, 127.

3. Bender, C.M., & Orszag, S.A. (2013). *Advanced mathematical methods for scientists and engineers I: Asymptotic methods and perturbation theory*. Springer Science & Business Media.

4. Mei, C.C., & Vernescu, B. (2010). *Homogenization methods for multiscale mechanics*. World Scientific.

5. Aurell, E., & Bo, S. (2016). Diffusion of a Brownian ellipsoid in a force field. *Europhysics Letters*, 114(3), 30005.

6. Biferale, L., Crisanti, A., Vergassola, M., & Vulpiani, A. (1995). Eddy diffusivities in scalar transport. *Physics of Fluids*, 7(11), 2725-2734.

7. Pavliotis, G.A., & Stuart, A.M. (2008). *Multiscale methods: averaging and homogenization*. Springer Science & Business Media.

8. Auriault, J.-L., Boutin, C., & Geindreau, C. (2010). *Homogenization of coupled phenomena in heterogenous media*. Vol. 149. Wiley & Sons.

9. Chipot, M. (2009). Weak formulation of elliptic problems. In *Elliptic Equations: An Introductory Course*, pp. 35-42. Birkhäuser Basel.

10. Hsiao, G.C., & Wendland, W.L. (2008). *Boundary integral equations*. Springer.

11. Jikov, V.V., Kozlov, S.M., & Oleinik, O.A. (2012). *Homogenization of differential operators and integral functionals*. Vol. 36. Springer Science & Business Media.

12. Bakhvalov, N.S., & Panasenko, G. (2012). *Homogenisation: Averaging processes in periodic media: Mathematical problems in the mechanics of composite materials*. Vol. 36. Springer Science & Business Media.

13. Kreß, R. (1977). On the Fredholm alternative for compact bilinear forms. *Journal of Differential Equations*, 25, 216.

14. Reed, M. (2012). *Methods of modern mathematical physics: Functional analysis*. Elsevier.

15. Mauri, R. (1991). Dispersion, convection, and reaction in porous media. *Physics of Fluids A*, 3, 743.

16. Auriault, J.-L., & Adler, P. (1995). Taylor dispersion in porous media: analysis by multiple scale expansions. *Advances in Water Resources*, 18, 217.

17. Romanczuk, P., Bär, M., Ebeling, W., Lindner, B., & Schimansky-Geier, L. (2012). Active Brownian particles. *The European Physical Journal Special Topics*, 202(1), 1-162.

### My Work

* Vachier, J., & Wettlaufer, J.S. (2022). Premelting controlled active matter in ice. *Physical Review E*, 105, 024601. [https://doi.org/10.1103/PhysRevE.105.024601](https://doi.org/10.1103/PhysRevE.105.024601)

* Vachier, J., & Wettlaufer, J.S. (2022). Biolocomotion and premelting in ice. *Frontiers in Physics*, 10, 904836. [https://doi.org/10.3389/fphy.2022.904836](https://doi.org/10.3389/fphy.2022.904836)

---

*Jeremy Vachier is Associate Director of Data Science at AstraZeneca Sweden with a PhD in Theoretical Physics from the Max Planck Institute, specializing in stochastic processes, active matter, and physics-informed machine learning.*







