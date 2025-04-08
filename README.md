# Bayesian Deep Kernel Learning on Olivetti Faces

This repository contains the code and experiments developed as part of a mathematics specialization project focused on approximate Bayesian inference with Gaussian Processes (GPs) and Deep Kernel Learning (DKL). The models are applied to the Olivetti Faces dataset with the goal of predicting the rotation angle of facial images from pixel intensities, while capturing predictive uncertainty.

## Overview

We implement and compare different Gaussian Process approximations:

- **Full GP (exact)** [Rasmussen & Williams, 2006]
- **Sparse GP** with Titsias' variational formulation [Titsias, 2009]
- **Stochastic Variational GP (SVGP)** [Hensman et al., 2013]
- **Deep Kernel Learning (DKL)** [Wilson et al., 2016] using CNN encoders
- **Stochastic Variational Deep Kernel Learning (SVDKL)** [Wilson et al., 2016] combining CNNs and SVGP

All models are implemented in PyTorch.

## Mathematical Formulations

### Gaussian Process Regression

Given input data \( X \in \mathbb{R}^{n \times d} \) and targets \( y \in \mathbb{R}^n \), a GP prior is defined by:

\[
f(x) \sim \mathcal{GP}(0, k(x, x'))
\]

The posterior predictive distribution at a new point \( x_* \) is:

\[
\begin{aligned}
\mu_* &= k(x_*, X) [K(X, X) + \sigma^2 I]^{-1} y \\
\sigma_*^2 &= k(x_*, x_*) - k(x_*, X) [K + \sigma^2 I]^{-1} k(X, x_*)
\end{aligned}
\]

### Sparse GP (Titsias, 2009)

Let \( Z \in \mathbb{R}^{m \times d} \) be the inducing points. Define:

- \( K_{mm} = k(Z, Z) \)
- \( K_{nm} = k(X, Z) \)
- \( Q_{nn} = K_{nm} K_{mm}^{-1} K_{mn} \)

The ELBO is:

\[
\text{ELBO} = \log \mathcal{N}(y \mid 0, Q_{nn} + \sigma^2 I) - \frac{1}{2\sigma^2} \mathrm{Tr}(K_{nn} - Q_{nn})
\]

### Stochastic Variational GP (Hensman et al., 2013)

We optimize a variational distribution over the inducing outputs \( q(f_m) = \mathcal{N}(m, S) \), with \( S = L L^\top \). The mini-batch ELBO becomes:

\[
\text{ELBO} = \sum_{i \in \text{batch}} \mathbb{E}_{q(f_i)} [ \log p(y_i | f_i) ] - \mathrm{KL}[q(f_m) || p(f_m)]
\]

### Deep Kernel Learning (Wilson et al., 2016)

We define a deep kernel by learning a nonlinear feature map \( \phi(x) \) via a CNN:

\[
k_{\text{deep}}(x, x') = k(\phi(x), \phi(x'))
\]

This is combined with a sparse GP or SVGP for scalability.

## Files

```
.
├── create_data.py             # Augments and saves the Olivetti dataset
├── dataset.py                 # Loads and splits the dataset
├── visualisation.py           # Displays prediction results with uncertainty
├── models/
│   ├── GP.py                  # Full GP
│   ├── SGP.py                 # Sparse GP (Titsias)
│   ├── SVGP.py                # Stochastic Variational GP (Hensman)
│   ├── NN.py                  # CNN encoders
│   ├── DKL.py                 # DKL with SGP
│   ├── SVDKL.py               # SVDKL with SVGP
├── dkl_olivetti.py            # Train + test DKL model
├── svdkl_olivetti.py          # Train + test SVDKL model
├── sgp_olivetti.py            # Train + test SGP model
├── svgp_olivetti.py           # Train + test SVGP model
├── gp_olivetti.py             # Train + test full GP
```

## Dataset: Olivetti Faces

We use `fetch_olivetti_faces()` from `sklearn.datasets` and augment the images with random rotations. The task is to regress the applied rotation angle from the raw pixel data.

## Installation

```bash
pip install -r requirement.txt
```

## Citation and References

- Titsias, M. K. (2009). Variational Learning of Inducing Variables in Sparse Gaussian Processes.
- Hensman, J., Fusi, N., & Lawrence, N. D. (2013). Gaussian Processes for Big Data.
- Wilson, A. G., Hu, Z., Salakhutdinov, R., & Xing, E. P. (2016). Deep Kernel Learning.
- Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.
