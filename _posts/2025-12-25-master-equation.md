---
layout: default
title: Deriving the Master Equation from Chapman-Kolmogorov
date: 2025-12-25
excerpt: A rigorous mathematical derivation of the master equation for jump Markov processes, with applications spanning bacterial biofilm dynamics, antibiotic resistance strategies, and machine learning optimization.
---

# Deriving the Master Equation from Chapman-Kolmogorov

This post complements my earlier work on [absorbing Markov chains](link-to-absorbing-markov-post) by providing a detailed derivation of the master equation using the Chapman-Kolmogorov equation. This fundamental result in stochastic process theory underpins countless applications across physics, chemistry, biology, and beyond.

## The Power of Stochastic Processes

Stochastic processes provide a unified mathematical framework for understanding systems evolving under uncertainty. During my [PhD](https://ediss.uni-goettingen.de/bitstream/handle/21.11130/00-1735-0000-0005-136F-A/Jeremy_Vachier_PhDthesis_correction_16_03_2020.pdf?sequence=1&isAllowed=y) and subsequent postdoctoral research in applied mathematics, I worked extensively with these mathematical structures, studying how randomness shapes complex systems. The master equation derived here serves as a fundamental tool that enables us to:

- Simulate chemical reactions via the **Gillespie algorithm** (Gillespie, 1976, 1977)
- Model gene expression dynamics and epigenetic switching
- Understand population dynamics and extinction events
- Analyze queuing systems and service networks
- Study neural spike trains and information processing

What makes stochastic processes so powerful is their ability to bridge microscopic randomness with macroscopic behavior, providing both analytical insight and computational tools.

## Jump Markov Processes

The master equation governs the evolution of **jump Markov processes** (also called continuous-time Markov chains), where the state space is discrete while time is continuous. These processes are characterized by piecewise-constant sample paths with instantaneous jumps between states at random times. For a comprehensive introduction to jump Markov processes and their mathematical foundations, see [Vachier (2020)](https://ediss.uni-goettingen.de/bitstream/handle/21.11130/00-1735-0000-0005-136F-A/Jeremy_Vachier_PhDthesis_correction_16_03_2020.pdf?sequence=1&isAllowed=y), Chapter 1.

## Setup: Conditional Probabilities

Let's denote the conditional probability as:

$$P_{nm}(t) = \mathbb{P}[X(t+u) = m \mid X(u) = n]. \tag{1}$$

If the time is homogeneous (time intervals are identical), this simplifies to:

$$P_{nm}(t) = \mathbb{P}[X(t) = m \mid X(0) = n]. \tag{2}$$

This homogeneity is crucial for deriving the master equation in a tractable form.

## The Chapman-Kolmogorov Equation

The Chapman-Kolmogorov equation expresses how probabilities compose over intermediate times. For the forward direction:

$$\frac{dP}{dt} = PQ \quad \text{(forward)}, \tag{3}$$

and for the backward direction:

$$\frac{dP}{dt} = QP \quad \text{(backward)}, \tag{4}$$

where $Q$ is the infinitesimal generator (or rate matrix) of the process.

## Forward Chapman-Kolmogorov Equation

Let's write the forward Chapman-Kolmogorov equation, $t \ge s$, is given by:

$$P_{nm}(t+s) = \sum_{k=0}^{+\infty} P_{nk}(t) P_{km}(s), \tag{5}$$

or using matrix notation

$$ P(t+s) =P(t)P(s). \tag{5 bis}$$

Similarly, the backward Chapman-Kolmogorov equation, $t \le s$, is given by:

$$P_{nm}(t+s) = \sum_{k=0}^{+\infty} P_{mk}(s) P_{km}(t), \tag{6}$$

or using matrix notation

$$ P(t+s) =P(s)P(t). \tag{6 bis}$$

## Deriving the Master Equation

Rewriting Eq. (5) gives:

$$
\begin{aligned}
P_{nm}(t+s) - P_{nm}(t) &= \sum_{\substack{k=0}}^{+\infty} P_{nk}(t) P_{km}(s) - P_{nm}(t) \\
&= \sum_{\substack{k\neq m}}^{+\infty} P_{nk}(t) P_{km}(s) + P_{nm}(t)P_{mm}(s) - P_{nm}(t) \\
&= \sum_{\substack{k\neq m}}^{+\infty} P_{nk}(t) P_{km}(s) + P_{nm}(t)\left(P_{mm}(s) - 1\right). 
\end{aligned}
$$

Taking the limit as $s \to 0$ of the previous equation gives:

$$
\begin{aligned}
\lim_{s \to 0} \frac{P_{nm}(t+s) - P_{nm}(t)}{s} = \lim_{s \to 0} \frac{1}{s} &\left[\sum_{\substack{k \neq m}}^{+\infty} P_{nk}(t) P_{km}(s) \right. \\
&\quad \left. + P_{nm}(t) \left(P_{mm}(s) - 1\right)\right]
\end{aligned}
$$

Noticing $1=\sum\limits_{k=0}^{+\infty}P_{mk}(s) \Leftrightarrow 1 = \sum\limits_{k\neq m}^{+\infty}P_{mk}(s) + P_{mm}(s)$ gives

$$
\begin{aligned}
\lim_{s \to 0} \frac{P_{nm}(t+s) - P_{nm}(t)}{s} = \lim_{s \to 0} \frac{1}{s} &\left[\sum_{\substack{k \neq m}}^{+\infty} P_{nk}(t) P_{km}(s) \right. \\
&\quad \left. - P_{nm}(t) \sum\limits_{k\neq m}^{+\infty}P_{mk}(s)\right],
\end{aligned}
$$
$$
\lim_{s \to 0} \frac{P_{nm}(t+s) - P_{nm}(t)}{s} = \lim_{s \to 0} \frac{1}{s} \sum_{\substack{k \neq m}}^{+\infty} \left[P_{nk}(t) P_{km}(s) - P_{nm}(t)P_{mk}(s)\right].
$$

Equivalently,

$$
\lim_{s \to 0} \frac{P_{nm}(t+s) - P_{nm}(t)}{s} = \sum_{\substack{k \neq m}}^{+\infty} \left[P_{nk}(t) Q_{km} - P_{nm}(t)Q_{mk}\right], \tag{7}
$$

where $Q_{nm} = \lim\limits_{s \to 0} \dfrac{P_{nm}(s)}{s}$ denotes the infinitesimal generator of the process.

## The Master Equation

Combining everything, we arrive at the forward **master equation**:

$$\boxed{\frac{dP_{nm}(t)}{dt} = \sum_{\substack{k \neq m}}^{+\infty} \left[ P_{nk}(t) Q_{km} - P_{nm}(t) Q_{mk}\right]}. \tag{8}$$

## Physical Interpretation

The master equation possesses a clear physical interpretation:

- **First term**: The rate at which conditional probability flows into state $m$ from all other states $k$
- **Second term**: The rate at which conditional probability flows out of state $m$ to all other states $k$

This balance between probability gain and loss governs the time evolution of the conditional probability distribution.

## Birth-Death Processes

An important special case of jump Markov processes is the **birth-death process**, where transitions are restricted to nearest neighbors in a one-dimensional state space. From state $n$, the system can only transition to $n+1$ (birth) or $n-1$ (death). For a birth-death process with birth rate $\lambda_n$ and death rate $\mu_n$ in state $n$, the master equation (8) simplifies to:

$$\frac{dP_n(t)}{dt} = \lambda_{n-1} P_{n-1}(t) + \mu_{n+1} P_{n+1}(t) - (\lambda_n + \mu_n) P_n(t) \tag{9}$$

Birth-death processes provide tractable yet powerful models across diverse applications, from population dynamics to bacterial colonization kinetics. A notable feature of birth-death processes is their potential for exact analytical solutions. In my research, I developed an exactly solvable birth-death model with time-dependent (temporal) attachment and detachment rates to quantify social cooperativity during bacterial reversible surface attachment in early biofilm formation (Lee and Vachier, 2020). This analytical solution enabled quantitative comparison of cooperative attachment strategies between different Pseudomonas aeruginosa strains.

This work has direct implications for combating antibiotic resistance, a critical challenge in pharmaceutical research. By understanding the temporal dynamics of bacterial surface attachment, the first stage of biofilm formation, the model identifies optimal time windows for drug administration. Targeting bacteria during vulnerable attachment phases, before mature biofilm formation confers drug resistance, offers a strategic approach to maximize treatment efficacy and mitigate the development of antibiotic-resistant strains.

## From Theory to Computation: The Gillespie Algorithm

While the master equation provides analytical insight, in practice we often need to simulate individual stochastic trajectories. This is where the **Gillespie algorithm** (also known as the Stochastic Simulation Algorithm or SSA) becomes invaluable.

Developed by Daniel Gillespie in 1976-1977, this algorithm provides an exact method for simulating systems governed by the master equation. Rather than solving the master equation directly (which becomes intractable for systems with many states), the Gillespie algorithm generates individual sample paths of the stochastic process.

### Applications of the Gillespie Algorithm

The algorithm has become particularly important in:

**Chemical Kinetics**: Simulating reaction networks where molecule counts are small and stochasticity matters (Gillespie, 1977)

**Epigenetic Switching**: Modeling bistable gene regulatory networks where cells can switch between different expression states. The stochastic nature of gene expression, captured by the master equation and simulated via Gillespie's algorithm, explains how genetically identical cells can adopt different phenotypes.

**Systems Biology**: Understanding noise in gene expression, protein production, and cellular decision-making

**Epidemiology**: Modeling disease spread in small populations where stochastic effects dominate

## Stochastic Processes: A Universal Language

What makes stochastic process theory so powerful is its universality. The same mathematical framework: continuous-time Markov chains, master equations, and simulation algorithms, applies across vastly different domains:

**Physics**: Brownian motion, diffusion processes, non-equilibrium statistical mechanics

**Chemistry**: Reaction kinetics, especially when molecule counts are low

**Biology**: Population dynamics, evolutionary processes, neural dynamics, gene regulatory networks

**Engineering**: Queuing theory, reliability analysis, communication networks

**Finance**: Option pricing, risk modeling, portfolio dynamics

**Machine Learning**: The master equation framework underpins fundamental ML algorithms. Stochastic gradient descent, the workhorse of deep learning, is governed by stochastic differential equations closely related to the master equation. Markov Decision Processes (MDPs) in reinforcement learning directly employ continuous-time Markov chains for policy optimization. Generative models, including diffusion models for image synthesis, leverage the mathematical machinery of stochastic processes to learn complex probability distributions. The same theoretical tools used to model bacterial attachment inform modern ML architectures

This universality underscores the fundamental importance of rigorous mathematical foundations such as the Chapman-Kolmogorov equation and master equation formalism. The same theoretical framework governs diverse phenomena across chemical kinetics, queuing theory, and stochastic optimization in machine learning, demonstrating the deep structural connections underlying seemingly disparate scientific domains.

## Connection to My Research

Throughout my academic career, from my PhD in theoretical physics to postdoctoral research and a research position in applied mathematics, I have extensively employed stochastic process theory to investigate systems where randomness plays a fundamental role. The remarkable aspect of this mathematical framework is its domain-independent nature: the same master equation formalism applies whether modeling bacterial biofilms, optimizing machine learning algorithms, or analyzing financial markets.

This universality enables seamless transitions between disparate fields. The analytical techniques for solving birth-death processes in biological systems translate directly to reinforcement learning problems. The Gillespie simulation methods developed for chemical kinetics power modern computational biology and stochastic optimization. My current work in industrial data science continues to leverage these fundamental mathematical tools, demonstrating that rigorous theoretical foundations transcend specific application domains. For more on my academic research, see my [PhD thesis](https://ediss.uni-goettingen.de/bitstream/handle/21.11130/00-1735-0000-0005-136F-A/Jeremy_Vachier_PhDthesis_correction_16_03_2020.pdf?sequence=1&isAllowed=y).

## Further Reading

### Foundational Texts
- Van Kampen, N. G. (2007). *Stochastic Processes in Physics and Chemistry*. North-Holland.
- Gardiner, C. (2009). *Stochastic Methods: A Handbook for the Natural and Social Sciences*. Springer.
- Gillespie, D. T. (1992). *Markov Processes: An Introduction for Physical Scientists*. Academic Press.
- Gillespie, D. T., & Seitaridou, E. (2013). *Simple Brownian Diffusion: An Introduction to the Standard Theoretical Models*. Oxford University Press.

### Gillespie's Seminal Papers
- Gillespie, D. T. (1976). "A general method for numerically simulating the stochastic time evolution of coupled chemical reactions." *Journal of Computational Physics*, 22(4), 403-434.
- Gillespie, D. T. (1977). "Exact stochastic simulation of coupled chemical reactions." *Journal of Physical Chemistry*, 81(25), 2340-2361.


### My Work
- Vachier, J. (2020). [*Nonequilibrium Statistical Mechanics: Collective Behavior of Active Particles*](https://ediss.uni-goettingen.de/bitstream/handle/21.11130/00-1735-0000-0005-136F-A/Jeremy_Vachier_PhDthesis_correction_16_03_2020.pdf?sequence=1&isAllowed=y). PhD Thesis, Georg-August-Universität Göttingen.
- Lee, C. K., & Vachier, J., et al. (2020). "Social cooperativity of bacteria during reversible surface attachment in young biofilms: a quantitative comparison of Pseudomonas aeruginosa PA14 and PAO1." *mBio*, 11(1), e02644-19. https://doi.org/10.1128/mbio.02644-19

---

*Written during Christmas 2025, revisiting the mathematical foundations that continue to inform my work across theoretical physics, applied mathematics, and industrial machine learning.*