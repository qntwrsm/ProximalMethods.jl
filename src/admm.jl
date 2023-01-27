#=
admm.jl

    Alternating direction method of multipliers (ADMM), also known as 
    Douglas-Rachford splitting, to optimize an objective function that can 
    be split into two components, where both components can be nonsmooth.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/01/20
=#

"""
    update_state!(state, λ, α, prox_f!, prox_g!)

Update states using proximal updates with scale `λ` and relaxation determined by
`α`, storing the result in `x`, `z`, and `u`.

#### Arguments
  - `state::ADMMState`  : state variables
  - `λ::Real`           : proximal scaling parameter
  - `α::Real`           : relaxation parameter
  - `prox_f!::Function` : proximal operator of ``f(x)``
  - `prox_g!::Function` : proximal operator of ``g(x)``
"""
function update_state!(
    state::ADMMState, 
    λ::Real, 
    α::Real, 
    prox_f!::Function, 
    prox_g!::Function
)
    # Proximal update x
    state.x.= state.z .- state.u
    prox_f!(state.x, λ)

    # Store previous z
    copyto!(state.z_prev, state.z)

    # Proximal update z
    state.z.= α .* state.x .+ (one(α) - α) .* state.z_prev .+ state.u
    prox_g!(state.z, λ)

    # Update u
    state.u.= state.u .+ α .* state.x .+ (one(α) - α) .* state.z_prev .- state.z

     return nothing
end

"""
    admm(x0, prox_f!, prox_g!; λ=1., α=1., ϵ_abs=1e-7, ϵ_rel=1e-4, max_iter=1000)

Minimize an objective function ``f(x) + g(x)``, where ``f(x)`` and ``g(x)`` can
both be nonsmooth, using alternating direction method of multipliers, also known
as Douglas-Rachford splitting.
The alternating direction method of multipliers method has two hyperparameters,
`λ` and `α`. `λ` controls the scaling of the update steps, i.e. a pseudo step
size, and is equal to the inverse of the augmented Lagrangian parameter. ``α ∈
[0,2]`` is the relaxation parameter, where ``α < 1`` denotes under-relaxation
and ``α > 1`` over-relaxation.

#### Arguments
  - `x0::AbstractVector`: initial parameter values (n x 1)
  - `prox_f!::Function` : proximal operator of ``f(x)``
  - `prox_g!::Function` : proximal operator of ``g(x)``
  - `λ::Real`           : proximal scaling parameter
  - `α::Real`           : relaxation parameter
  - `ϵ_abs::Real`       : absolute tolerance
  - `ϵ_rel::Real`       : relative tolerance
  - `max_iter::Integer` : max number of iterations
  
#### Returns
  - `x::AbstractVector` : minimizer ``∈ dom f`` (n x 1)
  - `z::AbstractVector` : minimizer ``∈ dom g`` (n x 1)
"""
function admm(
    x0::AbstractVector, 
    prox_f!::Function, 
    prox_g!::Function; 
    λ::Real=1., 
    α::Real=1., 
    ϵ_abs::Real=1e-7, 
    ϵ_rel::Real=1e-4, 
    max_iter::Integer=1000
)
    # Dimensions
    n= length(x0)
    
    # Initialize state
    state= ADMMState(
        similar(x0), 
        copy(x0), 
        zero(x0), 
        similar(x0), 
        similar(x0), 
        similar(x0)
    )

    # Initialize stopping parameters
    iter= 1
    ℓ₂_pri= one(ϵ_abs)
    ℓ₂_dual= one(ϵ_abs)
    ϵ_pri= zero(ϵ_abs)
    ϵ_dual= zero(ϵ_abs)
    # ADMM
    while (ℓ₂_pri > ϵ_pri || ℓ₂_dual > ϵ_dual) && iter < max_iter
        # Update states
        update_state!(state, λ, α, prox_f!, prox_g!)

        # Primal residual
        state.r.= state.x .- state.z
        ℓ₂_pri= norm(state.r)

        # Dual residual
        state.s.= -inv(λ) .* (state.z .- state.z_prev)
        ℓ₂_dual= norm(state.s)

        # Tolerance
        ϵ_pri= √n * ϵ_abs + ϵ_rel * max(norm(state.x), norm(state.z))
        ϵ_dual= √n * ϵ_abs + ϵ_rel * norm(inv(λ) .* state.u)

        # Update iteration counter
        iter+=1
    end
   
    return (state.x, state.z)
end

"""
    admm!(x, prox_f!, prox_g!; λ=1., α=1., ϵ_abs=1e-7, ϵ_rel=1e-4, max_iter=1000)

Minimize an objective function ``f(x) + g(x)``, where ``f(x)`` and ``g(x)`` can
both be nonsmooth, using alternating direction method of multipliers, also known
as Douglas-Rachford splitting, overwriting `x`. See also `admm`.
"""
function admm!(
    x::AbstractVector, 
    prox_f!::Function, 
    prox_g!::Function; 
    λ::Real=1., 
    α::Real=1., 
    ϵ_abs::Real=1e-7, 
    ϵ_rel::Real=1e-4, 
    max_iter::Integer=1000
)
    # Dimensions
    n= length(x)
    
    # Initialize state
    state= ADMMState(x, copy(x), zero(x), similar(x), similar(x), similar(x))

    # Initialize stopping parameters
    iter= 1
    ℓ₂_pri= one(ϵ_abs)
    ℓ₂_dual= one(ϵ_abs)
    ϵ_pri= zero(ϵ_abs)
    ϵ_dual= zero(ϵ_abs)
    # ADMM
    while (ℓ₂_pri > ϵ_pri || ℓ₂_dual > ϵ_dual) && iter < max_iter
        # Update states
        update_state!(state, λ, α, prox_f!, prox_g!)

        # Primal residual
        state.r.= state.x .- state.z
        ℓ₂_pri= norm(state.r)

        # Dual residual
        state.s.= -inv(λ) .* (state.z .- state.z_prev)
        ℓ₂_dual= norm(state.s)

        # Tolerance
        ϵ_pri= √n * ϵ_abs + ϵ_rel * max(norm(state.x), norm(state.z))
        ϵ_dual= √n * ϵ_abs + ϵ_rel * norm(inv(λ) .* state.u)

        # Update iteration counter
        iter+=1
    end
   
    return state.z
end