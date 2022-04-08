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
    state.x.= state.y .- λ .* state.∇f
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
    λ_min= ϵ * maximum(abs, state.∇f)

    # Update state
    update_state!(state, ls.λ, prox!)

    # Objective function value, f(x)
    f_x= f(state.x)

    # f̂
    state.Δ.= state.x .- state.y
    f_hat= f_y + dot(state.∇f, state.Δ) + inv(ls.λ+ls.λ)*norm(state.Δ)^2

    # tolerance
    tol= 10 * eps(f_x) * (one(f_x) + abs(f_x))

    # Backtracking line search
    while f_x > f_hat + tol && ls.λ > λ_min
        # Update stepsize
        ls.λ*= ls.β

        # Update state
        update_state!(state, ls.λ, prox!)

        # Objective function value
        f_x= f(state.x)

        # Update f̂
        state.Δ.= state.x .- state.y
        f_hat= f_y + dot(state.∇f, state.Δ) + inv(ls.λ+ls.λ)*norm(state.Δ)^2

        # Update tolerance
        tol= 10 * eps(f_x) * (one(f_x) + abs(f_x))
    end

    return nothing
end

"""
    prox_grad(x0, f, ∇f!, prox!; style="none", β=.5, ϵ=1e-7, max_iter=1000)

Minimize an objective function ``f(x) + g(x)``, where ``f(x)`` is differentibale
while ``g(x)`` is not, using the proximal gradient method.

#### Arguments
  - `x0::AbstractVector`    : initial parameter values (n x 1)
  - `f::Function`           : ``f(x)``
  - `∇f!::Function`         : gradient of `f`
  - `prox!::Function`       : proximal operator of ``g(x)``
  - `style::AbstractString` : acceleration style
  - `β::Real`               : line search parameter
  - `ϵ::Real`               : tolerance
  - `max_iter::Integer`     : max number of iterations

#### Returns
  - `x::AbstractVector` : minimizer (optimal parameter values) (n x 1)
"""
function prox_grad(x0::AbstractVector, f::Function, ∇f!::Function, prox!::Function; 
                    style::AbstractString="none", β::Real=.5, ϵ::Real=1e-7, 
                    max_iter::Integer=1000)
    # Initialize state and line search
    state= ProxGradState(copy(x0), copy(x0), similar(x0), similar(x0), similar(x0))
    ls= BackTrack(one(Float64), one(Float64), β)

    # Initialize acceleration
    if style == "none"
        acc= NoAccel(zero(Float64))
    elseif style == "simple"
        acc= Simple(zero(Float64), one(Int64))
    elseif style == "nesterov"
        acc= Nesterov(ω= zero(Float64), θ= one(Float64), ls= ls)
    end

    # Initialize stopping flag
    rel_change= Inf
    # Initialize iteration counter
    iter= 1
    # Proximal gradient method
    while rel_change > ϵ && iter < max_iter
        # Store change in state
        state.Δ.= state.x .- state.x_prev

        # Store current parameters
        copyto!(state.x_prev, state.x)

        # Extrapolation
        update_acc!(acc, state)
        state.y.= state.x .+ acc.ω .* state.Δ

        # Current gradient
        ∇f!(state.∇f, state.y)

        # Backtracking linesearch
        backtrack!(ls, state, f, prox!, ϵ=ϵ)

        # Relative change
        rel_change= norm(state.Δ, Inf) * inv(one(Float64) + norm(state.x, Inf))

        # Update iteration counter
        iter+=1
    end
    
    return state.x 
end

"""
    prox_grad!(x, f, ∇f!, prox!; style="none", β=.5, ϵ=1e-7, max_iter=1000)

Minimize an objective function ``f(x) + g(x)``, where ``f(x)`` is differentibale
while ``g(x)`` is not, using the proximal gradient method. Storing the result in
`x`. See also `prox_grad`.
"""
function prox_grad!(x::AbstractVector, f::Function, ∇f!::Function, prox!::Function; 
                    style::AbstractString="none", β::Real=.5, ϵ::Real=1e-7, 
                    max_iter::Integer=1000)
    # Initialize state and line search
    state= ProxGradState(x, copy(x), similar(x), similar(x), similar(x))
    ls= BackTrack(one(Float64), one(Float64), β)

    # Initialize acceleration
    if style == "none"
        acc= NoAccel(zero(Float64))
    elseif style == "simple"
        acc= Simple(zero(Float64), one(Int64))
    elseif style == "nesterov"
        acc= Nesterov(ω= zero(Float64), θ= one(Float64), ls= ls)
    end

    # Initialize stopping flag
    rel_change= Inf
    # Initialize iteration counter
    iter= 1
    # Proximal gradient method
    while rel_change > ϵ && iter < max_iter
        # Store change in state
        state.Δ.= state.x .- state.x_prev

        # Store current parameters
        copyto!(state.x_prev, state.x)

        # Extrapolation
        update_acc!(acc, state)
        state.y.= state.x .+ acc.ω .* state.Δ

        # Current gradient
        ∇f!(state.∇f, state.y)

        # Backtracking linesearch
        backtrack!(ls, state, f, prox!, ϵ=ϵ)

        # Relative change
        rel_change= norm(state.Δ, Inf) * inv(one(Float64) + norm(state.x, Inf))

        # Update iteration counter
        iter+=1
    end
    
    return nothing
end