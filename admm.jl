#=
admm.jl

    Alternating direction method of multipliers (ADMM), also known as 
    Douglas-Rachford splitting, to optimize an objective function that can 
    be split into two components, where both components can be nonsmooth.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/01/20
=#

"""
    admm(x0, prox_f!, prox_g!, args, λ=1., ϵ_abs=1e-7, ϵ_rel=1e-4, max_iter=1000)

Minimize an objective function ``f(x) + g(x)``, where ``f(x)`` and ``g(x)`` can
both be nonsmooth, using alternating direction method of multipliers, also known
as Douglas-Rachford splitting.

#### Arguments
  - `x0::AbstractVector`: initial parameter values (n x 1)
  - `prox_f!::Function` : proximal operator of ``f(x)``
  - `prox_g!::Function` : proximal operator of ``g(x)``
  - `args::NamedTuple`  : arguments for `prox_f!` and `prox_g!`
  - `λ::Real`           : proximal scaling parameter
  - `ϵ_abs::Real`       : absolute tolerance
  - `ϵ_rel::Real`       : relative tolerance
  - `max_iter::Integer` : max number of iterations
  
#### Returns
  - `x::AbstractVector` : minimizer ``∈ dom f`` (n x 1)
  - `z::AbstractVector` : minimizer ``∈ dom g`` (n x 1)
"""
function admm(x0::AbstractVector, prox_f!::Function, prox_g!::Function, 
                args::NamedTuple, λ::Real=1., ϵ_abs::Real=1e-7, ϵ_rel::Real=1e-4, 
                max_iter::Integer=1000)
    # Dimensions
    n= length(x0)
    
    # Initialize containers
    x= similar(x0)
    u= zero(x0)
    z= copy(x0)
    tmp= similar(x0)

    # Initialize stopping parameters
    iter= 1
    ℓ₂_pri= one(ϵ_abs)
    ℓ₂_dual= one(ϵ_abs)
    ϵ_pri= zero(ϵ_abs)
    ϵ_dual= zero(ϵ_abs)
    # ADMM
    while ℓ₂_pri > ϵ_pri && ℓ₂_dual > ϵ_dual && iter < max_iter
        # Proximal update x
        @. x= z - u
        prox_f!(x, λ, args.prox_f...)

        # Store previous z
        copyto!(tmp, z)

        # Proximal update z
        @. z= x + u
        prox_g!(z, λ, args.prox_g...)

        # Update u
        @. u= u + x - z

        # Dual residual
        @. tmp= inv(λ)*(z - tmp)
        ℓ₂_dual= norm(tmp)

        # Primal residual
        @. tmp= x + z
        ℓ₂_pri= norm(tmp)

        # Tolerance
        ϵ_pri= sqrt(n)*ϵ_abs + ϵ_rel*max(norm(x), norm(z))
        ϵ_dual= sqrt(n)*ϵ_abs + ϵ_rel*norm(u)

        # Update iteration counter
        iter+=1
    end
   
    return (x, z)
end

"""
    admm!(x, prox_f!, prox_g!, args, λ=1., ϵ_abs=1e-7, ϵ_rel=1e-4, max_iter=1000)

Minimize an objective function ``f(x) + g(x)``, where ``f(x)`` and ``g(x)`` can
both be nonsmooth, using alternating direction method of multipliers, also known
as Douglas-Rachford splitting, overwriting `x`. See also `admm`.
"""
function admm!(x::AbstractVector, prox_f!::Function, prox_g!::Function, 
                args::NamedTuple, λ::Real=1., ϵ_abs::Real=1e-7, ϵ_rel::Real=1e-4, 
                max_iter::Integer=1000)
    # Dimensions
    n= length(x)
    
    # Initialize containers
    u= zero(x)
    z= copy(x)
    tmp= similar(x)

    # Initialize stopping parameters
    iter= 1
    ℓ₂_pri= one(ϵ_abs)
    ℓ₂_dual= one(ϵ_abs)
    ϵ_pri= zero(ϵ_abs)
    ϵ_dual= zero(ϵ_abs)
    # ADMM
    while ℓ₂_pri > ϵ_pri && ℓ₂_dual > ϵ_dual && iter < max_iter
        # Proximal update x
        @. x= z - u
        prox_f!(x, λ, args.prox_f...)

        # Store previous z
        copyto!(tmp, z)

        # Proximal update z
        @. z= x + u
        prox_g!(z, λ, args.prox_g...)

        # Update u
        @. u= u + x - z

        # Dual residual
        @. tmp= inv(λ)*(z - tmp)
        ℓ₂_dual= norm(tmp)

        # Primal residual
        @. tmp= x + z
        ℓ₂_pri= norm(tmp)

        # Tolerance
        ϵ_pri= sqrt(n)*ϵ_abs + ϵ_rel*max(norm(x), norm(z))
        ϵ_dual= sqrt(n)*ϵ_abs + ϵ_rel*norm(u)

        # Update iteration counter
        iter+=1
    end
   
    return z
end