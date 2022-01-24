#=
acceleration.jl

    Acceleration routines for proximal optimization algorithms, such as simple 
    extrapolation steps, Nesterov momentum steps, etc.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/01/18
=#

"""
    extrapolation(k, ∇f_x, Δx)

Calculate simple extrapolation parameter with gradient based adaptive restart.

#### Arguments
  - `k::Integer`            : iteration counter
  - `∇f_x::AbstractVector`  : gradient of ``f(x)`` at ``x`` (n x 1)
  - `Δx::AbstractVector`    : difference (n x 1)

#### Returns
  - `ω::Real`   : extrapolation parameter
"""
function extrapolation(k::Integer, ∇f_x::AbstractVector, Δx::AbstractVector)
    # Gradient based adaptive restart
    k= dot(∇f_x, Δx) > zero(eltype(Δx)) ? 1 : k

    # Simple extrapolation
    ω= (k - 1)*inv(k + 2)

    return ω
end

"""
    nesterov(θ_k, λ, λ_k, ∇f_x, Δx, m=.0)

Calculate Nesterov extrapolation parameter with gradient based adaptive restart.

#### Arguments
  - `θ_k::Real`             : current momentum parameter
  - `λ::Real`               : current stepsize
  - `λ_k::Real`             : previous stepsize
  - `∇f_x::AbstractVector`  : gradient of ``f(x)`` at ``x`` (n x 1)
  - `Δx::AbstractVector`    : difference (n x 1)
  - `m::Real`               : convexity parameter

#### Returns
  - `β::Real`   : extrapolation parameter
  - `θ::Real`   : updated momentum parameter
"""
function nesterov(θ_k::Real, λ::Real, λ_k::Real, ∇f_x::AbstractVector, 
                    Δx::AbstractVector, m::Real=.0)
    # Gradient based adaptive restart
    θ_k= dot(∇f_x, Δx) > zero(eltype(Δx)) ? one(θ_k) : θ_k

    # Quadratic formula
    b = θ_k^2 * inv(λ_k) - m
    D= b^2 + 4 * inv(λ_k * λ) * θ_k^2

    # Update θ
    θ= .5 * λ * (-b + sqrt(D))

    # Update β
    β= λ * θ_k * (one(θ_k) - θ_k) * inv(λ_k * θ + λ * θ_k^2)

    return (β, θ)
end
