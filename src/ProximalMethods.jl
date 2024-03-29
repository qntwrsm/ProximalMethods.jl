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

using   LinearAlgebra, 
        Optim

export 
# Proximal Operators
	soft_thresh,
	block_soft_thresh, block_soft_thresh!,
	shrinkage, shrinkage!,
    smooth, smooth!,
# Proximal Gradient Methods
	prox_grad, prox_grad!, 
# ADMM
	admm, admm!,
# linearized ADMM
    ladmm, ladmm!

# Types
abstract type AbstractAccelScheme end

# Structs
# Line Search
mutable struct BackTrack{Tf, Gf}
	λ::Tf		# stepsize
	λ_prev::Tf 	# previous stepsize
	β::Gf		# contraction factor
end

# Acceleration
# No acceleration
struct NoAccel{Tf} <: AbstractAccelScheme
	ω::Tf	# extrapolation paramater 
end
# Simple extrapolation
mutable struct Simple{Tf, Gi} <: AbstractAccelScheme
	ω::Tf	# extrapolation paramater 
	k::Gi	# iteration counter
end
# Nesterov momentum
Base.@kwdef mutable struct Nesterov{Tf, G} <: AbstractAccelScheme
	ω::Tf		# extrapolation parameter
	θ::Tf       # momentum parameter
	m::Tf=.0	# convexity parameter
	ls::G		# line search parameters
end

# Proximal Gradient Methods
struct ProxGradState{Tv}
	x::Tv		# current state
	x_prev::Tv	# previous state
	y::Tv		# extrapolated state
	Δ::Tv		# change in state
	∇f::Tv		# gradient of f
end

# ADMM
struct ADMMState{Tv}
	x::Tv		# current state primal variable
	z::Tv		# current state primal variable
	u::Tv		# current scaled dual variable
	z_prev::Tv	# previous state primal variable
	r::Tv		# primal residuals
	s::Tv		# dual residuals
end

# linearized ADMM
struct lADMMState{Tv}
	x::Tv		# current state primal variable
	z::Tv		# current state primal variable
	u::Tv		# current scaled dual variable
	z_prev::Tv	# previous state primal variable
	r::Tv		# primal residuals
	s::Tv		# dual residuals
    Ax::Tv		# primal variable restrictions
    Atz::Tv     # transformed primal variable
    AtAx::Tv    # linearized term
    Atu::Tv		# transformed scaled dual variable
end


# Include programs
include("proximal.jl")
include("acceleration.jl")
include("gradient.jl")
include("admm.jl")
include("linearized_admm.jl")

end