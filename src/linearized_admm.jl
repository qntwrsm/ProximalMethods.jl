#=
linearized_admm.jl

    Linearized alternating direction method of multipliers (linearized ADMM) to 
    optimize an objective function that can be split into two components, where 
    both components can be nonsmooth. The linearized ADMM method is a variant of 
    the ADMM method that allows for the second components to depend on a matrix 
    transformation of the input variable. Linearized ADMM works by linearizing 
    the augemnted Lagrangian.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/01/27
=#

"""
    ladmm(x0, prox_f!, prox_g!, A; λ=1., μ=λ*inv(norm(A)), α=1., ϵ_abs=1e-7, ϵ_rel=1e-4, max_iter=1000)

Minimize an objective function ``f(x) + g(Ax)``, where ``f(x)`` and ``g(Ax)``
can both be nonsmooth, using linearized alternating direction method of
multipliers.
The alternating direction method of multipliers method has three
hyperparameters, `μ`, `λ`, and `α`. `μ` and `λ` control the scaling of the
update steps, i.e. a pseudo step sizes. `λ` is equal to the inverse of the
augmented Lagrangian parameter, while `μ` is the inverse of the augmented
Lagrangian parameter corresponding to the approximation term and ``μ ∈
(0,λ/||A||²₂]``. ``α ∈ [0,2]`` is the relaxation parameter, where ``α < 1``
denotes under-relaxation and ``α > 1`` over-relaxation.

#### Arguments
  - `x0::AbstractVector`: initial parameter values (n x 1)
  - `prox_f!::Function` : proximal operator of ``f(x)``
  - `prox_g!::Function` : proximal operator of ``g(x)``
  - `A::AbstractMatrix` : matrix transformation of ``g(x)`` (p x n)
  - `λ::Real`           : proximal scaling parameter of ``g(x)``
  - `μ::Real`           : proximal scaling parameter of ``f(x)``
  - `α::Real`           : relaxation parameter
  - `ϵ_abs::Real`       : absolute tolerance
  - `ϵ_rel::Real`       : relative tolerance
  - `max_iter::Integer` : max number of iterations
  
#### Returns
  - `x::AbstractVector` : minimizer ``∈ dom f`` (n x 1)
  - `z::AbstractVector` : minimizer ``∈ dom g`` (p x 1)
"""
function ladmm(
    x0::AbstractVector, 
    prox_f!::Function, 
    prox_g!::Function,
    A::AbstractMatrix;
    λ::Real=1., 
    μ::Real=λ*inv(norm(A)^2),
    α::Real=1., 
    ϵ_abs::Real=1e-7, 
    ϵ_rel::Real=1e-4, 
    max_iter::Integer=1000
)
    # Dimensions
    (p, n) = size(A) 
    
    # Initialize state
    z0 = A * x0
    state = lADMMState(
        similar(x0), 
        z0, 
        zero(z0), 
        similar(z0), 
        similar(z0), 
        similar(z0),
        similar(x0),
        similar(x0)
    )

    # Initialize stopping parameters
    iter = 1
    ℓ₂_pri = one(ϵ_abs)
    ℓ₂_dual = one(ϵ_abs)
    ϵ_pri = zero(ϵ_abs)
    ϵ_dual = zero(ϵ_abs)
    # ADMM
    while (ℓ₂_pri > ϵ_pri || ℓ₂_dual > ϵ_dual) && iter < max_iter
        # Update states
        update_state!(state, A, μ, λ, α, prox_f!, prox_g!)

        # Primal residual
        mul!(state.Ax, A, state.x)
        state.r .= state.Ax .- state.z
        ℓ₂_pri = norm(state.r)

        # Dual residual
        mul!(state.s, transpose(A), state.z, -inv(λ), zero(λ))
        mul!(state.s, transpose(A), state.z_prev, inv(λ), one(λ))
        ℓ₂_dual = norm(state.s)

        # Tolerance
        ϵ_pri = √p * ϵ_abs + ϵ_rel * max(norm(state.Ax), norm(state.z))
        mul!(state.Atu, transpose(A), state.u)
        ϵ_dual = √n * ϵ_abs + ϵ_rel * norm(inv(λ) .* state.Atu)

        # Update iteration counter
        iter += 1
    end
   
    return (state.x, state.z)
end