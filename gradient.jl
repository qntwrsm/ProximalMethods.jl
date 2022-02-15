#=
gradient.jl

    Proximal gradient methods, w/ and w/o acceleration, to optimize an objective 
    function that can be split into two components, one of which is 
    differentiable.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/01/14
=#

"""
    update_state!(state, λ, prox!)

Update state using proximal gradient step with step size `λ`, storing the result
in `x`.

#### Arguments
  - `state::ProxGradState`  : state variables
  - `λ::Real`               : stepsize
  - `prox!::Function`       : proximal operator of ``g(x)``
"""
function update_state!(state::ProxGradState, λ::Real, prox!::Function)
    # Proximal gradient step
    @. state.x= state.y - λ*state.∇f
    prox!(state.x, λ)

    return nothing
end

"""
    backtrack!(ls, state, f, prox!; ϵ=1e-7)

Backtracking line search to find the optimal step size `λ`.

#### Arguments
  - `ls::BackTrack`         : line search variables
  - `state::ProxGradState`  : state variables
  - `λ::Real`               : initial stepsize
  - `f::Function`           : objective function
  - `prox!::Function`       : proximal operator of ``g(x)``
  - `ϵ::Real`               : tolerance
"""
function backtrack!(ls::BackTrack, state::ProxGradState, f::Function, 
                    prox!::Function; ϵ::Real=1e-7)
    # Store previous stepsize
    ls.λ_prev= ls.λ
    
    # Objective function value, f(y)
    f_y= f(state.y)

    # Minimal stepsize
    λ_min= ϵ*inv(minimum(abs, state.∇f))

    # Update state
    update_state!(state, ls.λ, prox!)

    # Objective function value, f(x)
    f_x= f(state.x)

    # f̂
    @. state.Δ= state.x - state.y
    f_hat= f_y + dot(state.∇f, state.Δ) + inv(ls.λ+ls.λ)*norm(state.x)^2

    # Backtracking line search
    while f_x > f_hat && ls.λ > λ_min
        # Update stepsize
        ls.λ*= ls.β

        # Update state
        update_state!(state, ls.λ, prox!)

        # Objective function value
        f_x= f(state.x)

        # Update f̂
        @. state.Δ= state.x - state.y
        f_hat= f_x_prev + dot(state.∇f, state.Δ) + inv(ls.λ+ls.λ)*norm(state.x)^2
    end

    return nothing
end

"""
    prox_grad(x0, f, ∇f!, prox!; style="none", β=.5, ϵ_abs=1e-7, ϵ_rel=1e-3, max_iter=1000)

Minimize an objective function ``f(x) + g(x)``, where ``f(x)`` is differentibale
while ``g(x)`` is not, using the proximal gradient method.

#### Arguments
  - `x0::AbstractVector`    : initial parameter values (n x 1)
  - `f::Function`           : ``f(x)``
  - `∇f!::Function`         : gradient of `f`
  - `prox!::Function`       : proximal operator of ``g(x)``
  - `style::AbstractString` : acceleration style
  - `β::Real`               : line search parameter
  - `ϵ_abs::Real`           : absolute tolerance
  - `ϵ_rel::Real`           : relative tolerance
  - `max_iter::Integer`     : max number of iterations

#### Returns
  - `x::AbstractVector` : minimizer (optimal parameter values) (n x 1)
"""
function prox_grad(x0::AbstractVector, f::Function, ∇f!::Function, prox!::Function; 
                    style::AbstractString="none", β::Real=.5, ϵ_abs::Real=1e-7, 
                    ϵ_rel::Real=1e-3, max_iter::Integer=1000)
    # Initialize state and line search
    state= ProxGradState(copy(x0), similar(x0), similar(x0), similar(x0), similar(x0))
    ls= BackTrack(one(Float64), one(Float64), β)

    # Initialize acceleration
    if style == "none"
        acc= NoAccel(zero(Float64))
    elseif style == "simple"
        acc= Simple(zero(Float64), one(Int64))
    elseif style == "nesterov"
        acc= Nesterov(zero(Float64), one(Float64), ls)
    end

    # Initialize stopping flags
    abs_change= zero(eltype(x0))
    rel_change= zero(eltype(x0))
    # Initialize iteration counter
    iter= 1
    # Proximal gradient method
    while (abs_change < ϵ_abs || rel_change < ϵ_rel) && iter < max_iter
        # Store current parameters
        copyto!(state.x_prev, state.x)

        # Extrapolation
        update_acc!(acc, state)
        @. state.y= state.x + acc.ω * state.Δ

        # Current gradient
        ∇f!(state.∇f, state.y)

        # Backtracking linesearch
        backtrack!(ls, state, f, prox!, ϵ=ϵ_abs)

        # Store change in state
        @. state.Δ= state.x - state.x_prev

        # Absolute change
        abs_change= maximum(abs, state.Δ)
        # Relative change
        rel_change= abs_change * inv(1 + maximum(abs, state.x))

        # Update iteration counter
        iter+=1
    end
    
    return state.x 
end

"""
    prox_grad!(x, f, ∇f!, prox!; style="none", β=.5, ϵ_abs=1e-7, ϵ_rel=1e-3, max_iter=1000)

Minimize an objective function ``f(x) + g(x)``, where ``f(x)`` is differentibale
while ``g(x)`` is not, using the proximal gradient method. Storing the result in
`x`. See also `prox_grad`.
"""
function prox_grad!(x::AbstractVector, f::Function, ∇f!::Function, prox!::Function; 
                    style::AbstractString="none", β::Real=.5, ϵ_abs::Real=1e-7, 
                    ϵ_rel::Real=1e-3, max_iter::Integer=1000)
    # Initialize state and line search
    state= ProxGradState(x, similar(x), similar(x), similar(x), similar(x))
    ls= BackTrack(one(Float64), one(Float64), β)

    # Initialize acceleration
    if style == "none"
        acc= NoAccel(zero(Float64))
    elseif style == "simple"
        acc= Simple(zero(Float64), one(Int64))
    elseif style == "nesterov"
        acc= Nesterov(zero(Float64), one(Float64), ls)
    end

    # Initialize stopping flag
    abs_change= zero(eltype(x0))
    rel_change= zero(eltype(x0))
    # Initialize iteration counter
    iter= 1
    # Proximal gradient method
    while (abs_change < ϵ_abs || rel_change < ϵ_rel) && iter < max_iter
        # Store current parameters
        copyto!(state.x_prev, state.x)

        # Extrapolation
        update_acc!(acc, state)
        @. state.y= state.x + acc.ω * state.Δ

        # Current gradient
        ∇f!(state.∇f, state.y)

        # Backtracking linesearch
        backtrack!(ls, state, f, prox!, ϵ=ϵ_abs)

        # Store change in state
        @. state.Δ= state.x - state.x_prev

        # Absolute change
        abs_change= maximum(abs, state.Δ)
        # Relative change
        rel_change= abs_change * inv(1 + maximum(abs, state.x))

        # Update iteration counter
        iter+=1
    end
    
    return nothing
end