# Caputo-DLPNN

Caputo-DLPNN is a physics-informed research framework designed for solving **coupled time-fractional reaction-diffusion systems**. By leveraging a **Deep Legendre Polynomial Neural Network (DLPNN)** architecture, this method incorporates **Caputo fractional derivatives** through an **analytical basis-derivative approach**, supporting numerical stability and spectral efficiency.

## Highlights

- **Analytical Fractional Differentiation**  
  Uses exact closed-form derivatives of shifted Legendre polynomials via Gamma functions, bypassing the noise and overhead of numerical fractional schemes.

- **Coupled System Dynamics**  
  Specifically engineered to handle interacting state variables \(u, v\) with distinct spectral demands.

- **Adaptive Gating & Weighting**  
  Features variable-specific basis gating and weighted residual scheduling to balance the learning convergence of coupled components.

- **Physics-Informed Architecture**  
  Training is driven by the minimization of equation residuals, initial conditions (IC), and boundary conditions (BC).

## Problem Formulation

The framework targets coupled time-fractional reaction-diffusion equations of the form:

\[
{}^C D_t^\alpha u(x,t) = d_1 \frac{\partial^2 u}{\partial x^2} + R_u(u, v, x, t) + f_1(x,t)
\]

\[
{}^C D_t^\alpha v(x,t) = d_2 \frac{\partial^2 v}{\partial x^2} + R_v(u, v, x, t) + f_2(x,t)
\]

Where:

- \({}^C D_t^\alpha\) is the Caputo fractional derivative of order \(0 < \alpha \leq 1\)
- \(R_u(u, v, x, t)\) and \(R_v(u, v, x, t)\) represent the nonlinear interaction/reaction terms
- The domain is typically \((x, t) \in [0,1] \times [0,1]\)

## Methodology: The DLPNN Advantage

Unlike standard Physics-Informed Neural Networks (PINNs) that rely on automatic differentiation for all operators, Caputo-DLPNN utilizes a hybrid spectral-neural approach.

### 1. Feature Encoding
Input features are expanded using a tensor product of shifted Legendre polynomials.

### 2. Analytical Residuals
The fractional derivative of the basis functions is pre-calculated analytically:

\[
\frac{\partial^\alpha}{\partial t^\alpha} \psi_j(t)
=
\sum_{k=\lceil \alpha \rceil}^{j}
w_{jk}
\frac{\Gamma(k+1)}{\Gamma(k+1-\alpha)}
t^{k-\alpha}
\]

### 3. Adaptive Gating
A gating mechanism selects the most relevant basis functions for each state variable, reducing representational competition between coupled outputs.

## Results and Insights

The current proof-of-concept experiments indicate:

- **Accuracy**  
  Approximately **5% relative error** for the primary state variable \(u\), and around **15%** for the more challenging coupled component \(v\).

- **Saturation Regime**  
  A spectral bottleneck is observed when fixed-order basis expansions reach their expressive limit, motivating higher-order or adaptive basis selection.

- **Stability**  
  The analytical formulation maintains stable convergence even at lower fractional orders where purely numerical methods often struggle.

## Repository Structure

```text
Caputo-DLPNN/
├── caputo_dlpnn_research_notebook_20k.ipynb  # Main research notebook (20,000 epochs)
├── results/                                  # Visualization and error plots
└── README.md                                 # Project documentation
