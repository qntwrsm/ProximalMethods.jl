#=
acceleration.jl

    Acceleration routines for proximal optimization algorithms, such as simple 
    extrapolation steps, Nesterov momentum steps, etc.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/01/18
=#

"""
    update_acc!(acc, state)
  
Update the extrapolation parameter `ω` using a specific acceleration scheme,
such as simple extrapolation and Nesterov momentum.

#### Arguments
  - `acc::AbstractAccelScheme`	: acceleration scheme variables
  - `state::ProxGradState`		: state variables
"""
function update_acc! end

function update_acc!(acc::NoAccel, state::ProxGradState)
	# Do nothing... 

	return nothing
end

function update_acc!(acc::Simple, state::ProxGradState)
	# Gradient based adaptive restart
    acc.k= dot(state.∇f, state.Δ) > zero(eltype(state.Δ)) ? 1 : acc.k

    # Simple extrapolation
    acc.ω= (acc.k - 1)*inv(acc.k + 2)

	return nothing
end

function update_acc!(acc::Nesterov, state::ProxGradState)
	# Gradient based adaptive restart
    acc.θ= dot(state.∇f, state.Δ) > zero(eltype(state.Δ)) ? one(acc.θ) : acc.θ

	# Clean up notation
	λ_prev= acc.ls.λ_prev
	λ= acc.ls.λ
	θ_prev= acc.θ

    # Quadratic formula
    b = θ_prev^2 * inv(λ_prev) - acc.m
    D= b^2 + 4 * inv(λ_prev * λ) * θ_prev^2

    # Update Nesterov momentum
    acc.θ= .5 * λ * (-b + sqrt(D))

    # Nesterov extrapolation
    acc.ω= λ * θ_prev * (one(θ_prev) - θ_prev) * inv(λ_prev * acc.θ + λ * θ_prev^2)

	return nothing
end