#=
ProximalMethods.jl

    Provides proximal operator evaluation routines and proximal optimization 
	algorithms, such as (accelerated) proximal gradient methods and alternating 
	direction method of multipliers (ADMM), for non-smooth/non-differentiable 
	objective functions.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/01/12
=#

module ProximalMethods

using LinearAlgebra

export 
# Proximal Operators
	prox_l1, prox_l1!,
	prox_l2, prox_l2!,
	prox_ridge, prox_ridge!,
	prox_elnet, prox_elnet!,
# Acceleration Methods 
	extrapolation,
	nesterov,
# Proximal Gradient Methods
	prox_grad, prox_grad!, 
	prox_grad_expol, prox_grad_expol!,
# ADMM
	admm

# Include programs
include("proximal.jl")
include("acceleration.jl")
include("gradient.jl")
include("admm.jl")

end