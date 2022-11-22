# ProximalMethods

This is a package for non-smooth optimization algorithms based on proximal methods.

Provides proximal operator evaluation routines and proximal optimization algorithms, such as (accelerated) proximal gradient methods and alternating direction method of multipliers (ADMM), for non-smooth/non-differentiable objective functions.

# Proximal Operators

The following proximal operators are supported

- `soft_thresh(x, λ)`: proximal operator of $\ell_{1}$-norm
- `block_soft_thresh(x, λ)`: proximal operator of $\ell_{2}$-norm
- `shrinkage(x, λ)`: proximal operator of $\ell^{2}_{2}$-norm (ridge)
- `shrinkage(x, λ, A, b)`: proximal operator of quadratic function $f(x) = c + b^{\prime}x + x^{\prime}Ax$
- `smooth(x, λ, f, ∇f!, y_prev)`: proximal operator of a general smooth objective function $f(x)$

# Algorithms

Proximal gradient methods, w/ and w/o acceleration, can be used to optimize an objective function that can be split into two components, one of which is differentiable, i.e.

$$
\min f(x) + g(x)
$$

where $f$ is differentiable and $g$ potentially non-smooth.

Alternating direction method of multipliers (ADMM), also known as Douglas-Rachford splitting, can be used to optimize an objective function that can be split into two components, where both components can be non-smooth.

# Acceleration

For the proximal gradient method there exist so-called accelerated versions, which implies the following update step at iteration $k$

$$
\begin{aligned}
    y^{k+1} & = x^{k} + \omega^{k}(x^{k} - x^{k-1}) \\
    x^{k+1} & = \text{prox}_{\lambda^{k} g}(y^{k+1} - \lambda^{k} \nabla f(y^{k+1}))
\end{aligned}
$$

Two flavours of this acceleration are implemented

- Simple extrapolation, i.e. $\omega^{k} = \frac{k - 1}{k + 2}$ for every iteration $k$.
- Nesterov momentum extrapolation, i.e. $\omega^{k} = \frac{\lambda^{k}\theta^{k-1}(1 - \theta^{k-1})}{\lambda^{k-1}\theta^{k} + \lambda^{k}(\theta^{k-1})^{2}}$ where $\theta^{k}$ is the positive root of the quadratic equation

  $$
  \frac{(\theta^{k})^{2}}{\lambda^{k}} = (1 - \theta^{k})\frac{(\theta^{k-1})^{2}}{\lambda^{k-1}} + m\theta^{k}
  $$

# Documentation

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://qntwrsm.github.io/ProximalMethods.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://qntwrsm.github.io/ProximalMethods.jl/dev)

# Installation

[![Build Status](https://github.com/qntwrsm/ProximalMethods.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/qntwrsm/ProximalMethods.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/qntwrsm/ProximalMethods.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/qntwrsm/ProximalMethods.jl)
