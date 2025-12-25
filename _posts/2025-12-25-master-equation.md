---
title: "Deriving the Master Equation from Chapman-Kolmogorov"
date: 2025-12-25
tags: [stochastic-processes, master-equation, theoretical-physics, markov-chains]
description: "A rigorous derivation of the master equation for continuous-time Markov chains using the Chapman-Kolmogorov equation"
---

# Deriving the Master Equation from Chapman-Kolmogorov

This post complements my earlier work on [absorbing Markov chains](link-to-absorbing-markov-post) by providing a detailed derivation of the master equation using the Chapman-Kolmogorov equation. This fundamental result in stochastic process theory underpins countless applications across physics, chemistry, biology, and beyond.

## The Power of Stochastic Processes

Stochastic processes provide a unified mathematical framework for understanding systems evolving under uncertainty. During my [PhD](https://ediss.uni-goettingen.de/bitstream/handle/21.11130/00-1735-0000-0005-136F-A/Jeremy_Vachier_PhDthesis_correction_16_03_2020.pdf?sequence=1&isAllowed=y), I worked extensively with these mathematical structures, studying how randomness shapes complex systems. The master equation derived here is not just an abstract mathematical object, it serves as a fundamental tool that enables us to:

- Simulate chemical reactions via the **Gillespie algorithm** (Gillespie, 1976, 1977)
- Model gene expression dynamics and epigenetic switching
- Understand population dynamics and extinction events
- Analyze queuing systems and service networks
- Study neural spike trains and information processing

What makes stochastic processes so powerful is their ability to bridge microscopic randomness with macroscopic behavior, providing both analytical insight and computational tools.

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

where $Q$ is the rate matrix (or generator matrix) of the process.

## Forward Chapman-Kolmogorov Equation

Let's write the forward Chapman-Kolmogorov equation, $t \ge s$, is given by:

$$P_{nm}(t+s) = \sum_{k=0}^{+\infty} P_{nk}(t) P_{km}(s), \tag{5}$$

or using matrices notation

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
&\quad \left. + P_{nm}(t) \sum\limits_{k\neq m}^{+\infty}P_{mk}(s)\right],
\end{aligned}
$$
$$
\lim_{s \to 0} \frac{P_{nm}(t+s) - P_{nm}(t)}{s} = \lim_{s \to 0} \frac{1}{s} \sum_{\substack{k \neq m}}^{+\infty} \left[P_{nk}(t) P_{km}(s) + P_{nm}(t)P_{mk}(s)\right].
$$

Or

$$
\lim_{s \to 0} \frac{P_{nm}(t+s) - P_{nm}(t)}{s} = \sum_{\substack{k \neq m}}^{+\infty} \left[P_{nk}(t) Q_{km} + P_{nm}(t)Q_{mk}\right], \tag{7}
$$

with $Q_{nm} = \lim\limits_{s \to 0} \dfrac{P_{nm}(s)}{s}$.

## The Master Equation

Combining everything, we arrive at the forward **master equation**:

$$\boxed{\frac{dP_{nm}(t)}{dt} = \sum_{\substack{k \neq m}}^{+\infty} P_{nk}(t) Q_{km} - P_{nm}(t) Q_{mk}}. \tag{8}$$

## Physical Interpretation

The master equation has a beautiful physical interpretation:

- **First term**: The rate of **gain** of probability in state $m$ from all other states $k$
- **Second term**: The rate of **loss** of probability from state $m$ to all other states $k$

This balance between gain and loss describes the time evolution of the probability distribution over states.

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

**Machine Learning**: Stochastic optimization, reinforcement learning, generative models

This universality underscores the fundamental importance of rigorous mathematical foundations such as the Chapman-Kolmogorov equation and master equation formalism. The same theoretical framework governs diverse phenomena across chemical kinetics, queuing theory, and stochastic optimization in machine learning, demonstrating the deep structural connections underlying seemingly disparate scientific domains.

## Connection to My Research

During my PhD work on stochastic processes, I used these tools to understand complex dynamical systems where randomness plays a fundamental role. The transition from theoretical foundations to practical applications, from deriving master equations to implementing Gillespie simulations to analyzing real data, mirrors my current work in industrial data science, where rigorous mathematical foundations meet real, world complexity. For more on my PhD research, see my [thesis](https://ediss.uni-goettingen.de/bitstream/handle/21.11130/00-1735-0000-0005-136F-A/Jeremy_Vachier_PhDthesis_correction_16_03_2020.pdf?sequence=1&isAllowed=y).

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
- Vachier, J. (2020). [*Nonequilibrium Statistical Mechanics COLLECTIVE BEHAVIOR OF ACTIVE PARTICLES*](https://ediss.uni-goettingen.de/bitstream/handle/21.11130/00-1735-0000-0005-136F-A/Jeremy_Vachier_PhDthesis_correction_16_03_2020.pdf?sequence=1&isAllowed=y). PhD Thesis, Georg-August-Universität Göttingen.

---

*Written during my Christmas vacation 2025, revisiting the mathematical foundations that continue to shape my work, from theoretical physics to industrial machine learning, the power of stochastic thinking remains constant.*