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
    update_state!(state, A, μ, λ, α, prox_f!, prox_g!)

Update states using proximal updates with scale `μ` and `λ` and relaxation
determined by `α`, storing the result in `x`, `z`, and `u`.

#### Arguments
  - `state::lADMMState` : state variables
  - `A::AbstractMatrix` : matrix transformation of ``g(x)`` (p x n)
  - `μ::Real`           : proximal scaling parameter of ``f(x)``
  - `λ::Real`           : proximal scaling parameter of ``g(x)``
  - `α::Real`           : relaxation parameter
  - `prox_f!::Function` : proximal operator of ``f(x)``
  - `prox_g!::Function` : proximal operator of ``g(x)``
"""
function update_state!(
    state::lADMMState, 
    A::AbstractMatrix,
    μ::Real,
    λ::Real, 
    α::Real, 
    prox_f!::Function, 
    prox_g!::Function
)
    # Proximal update x
    state.x .-= μ .* inv(λ) .* (state.AtAx .- state.Atz .+ state.Atu)
    prox_f!(state.x, μ)

    # update Ax and A'Ax
    mul!(state.Ax, A, state.x)
    mul!(state.AtAx, transpose(A), state.Ax)

    # Store previous z
    copyto!(state.z_prev, state.z)

    # Proximal update z
    state.z .= α .* state.Ax .+ (one(α) - α) .* state.z_prev .+ state.u
    prox_g!(state.z, λ)

    # update A'z
    mul!(state.Atz, transpose(A), state.z)

    # Update u
    state.u .+= .+ α .* state.Ax .+ (one(α) - α) .* state.z_prev .- state.z

    # update A'u
    mul!(state.Atu, transpose(A), state.u)

    return nothing
end

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
    μ::Real=λ*inv(opnorm(A)^2),
    α::Real=1., 
    ϵ_abs::Real=1e-7, 
    ϵ_rel::Real=1e-4, 
    max_iter::Integer=1000
)
    # Dimensions
    (p, n) = size(A)
    
    # Initialize state
    z0 = A * x0
    Atz0 = transpose(A) * z0
    state = lADMMState(
        copy(x0), 
        z0, 
        zero(z0),
        similar(z0), 
        similar(z0), 
        similar(x0),
        copy(z0),
        Atz0,
        copy(Atz0),
        zero(x0)
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
        state.r .= state.Ax .- state.z
        ℓ₂_pri = norm(state.r)

        # Dual residual
        mul!(state.s, transpose(A), state.z_prev, inv(λ), zero(λ))
        state.s .-= inv(λ) .* state.Atz
        ℓ₂_dual = norm(state.s)

        # Tolerance
        ϵ_pri = √p * ϵ_abs + ϵ_rel * max(norm(state.Ax), norm(state.z))
        ϵ_dual = √n * ϵ_abs + ϵ_rel * norm(inv(λ) .* state.Atu)

        # Update iteration counter
        iter += 1
    end
   
    return (state.x, state.z)
end

"""
    ladmm!(x, prox_f!, prox_g!, A; λ=1., μ=λ*inv(norm(A)), α=1., ϵ_abs=1e-7, ϵ_rel=1e-4, max_iter=1000)

Minimize an objective function ``f(x) + g(Ax)``, where ``f(x)`` and ``g(Ax)``
can both be nonsmooth, using linearized alternating direction method of
multipliers, overwriting `x`. See also `ladmm`.
"""
function ladmm!(
    x::AbstractVector, 
    prox_f!::Function, 
    prox_g!::Function,
    A::AbstractMatrix;
    λ::Real=1., 
    μ::Real=λ*inv(opnorm(A)^2),
    α::Real=1., 
    ϵ_abs::Real=1e-7, 
    ϵ_rel::Real=1e-4, 
    max_iter::Integer=1000
)
    # Dimensions
    (p, n) = size(A)
    
    # Initialize state
    z0 = A * x
    Atz0 = transpose(A) * z0
    state = lADMMState(
        x, 
        z0, 
        zero(z0),
        similar(z0), 
        similar(z0), 
        similar(x),
        copy(z0),
        Atz0,
        copy(Atz0),
        zero(x)
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
        state.r .= state.Ax .- state.z
        ℓ₂_pri = norm(state.r)

        # Dual residual
        mul!(state.s, transpose(A), state.z_prev, inv(λ), zero(λ))
        state.s .-= inv(λ) .* state.Atz
        ℓ₂_dual = norm(state.s)

        # Tolerance
        ϵ_pri = √p * ϵ_abs + ϵ_rel * max(norm(state.Ax), norm(state.z))
        ϵ_dual = √n * ϵ_abs + ϵ_rel * norm(inv(λ) .* state.Atu)

        # Update iteration counter
        iter += 1
    end
   
    return state.z
end