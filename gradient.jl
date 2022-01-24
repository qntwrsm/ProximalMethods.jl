#=
gradient.jl

    Proximal gradient methods, w/ and w/o acceleration, to optimize an objective 
    function that can be split into two components, one of which is 
    differentiable.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/01/14
=#

"""
    linesearch!(x, x0, f_x0, ∇f_x0, λ, f, prox!, args, β=.5, ϵ=1e-7)

Backtracking line search to find the optimal step size `λ`.

#### Arguments
  - `x0::AbstractVector`    : initial parameters
  - `f_x0::Real`            : ``f(x0)``
  - `∇f_x0::AbstractVector` : gradient of `f` at `x0`
  - `λ::Real`               : initial stepsize
  - `f::Function`           : objective function
  - `prox!::Function`       : proximal operator of ``g(x)``
  - `args::NamedTuple`      : optional arguments for `f` and `prox!`
  - `β::Real`               : line search parameter
  - `ϵ::Real`               : tolerance

#### Returns
  - `x::AbstractVector` : updated paramaters
  - `λ::Real`           : optimal stepsize
"""
function linesearch!(x::AbstractVector, x0::AbstractVector, f_x0::Real, 
                    ∇f_x0::AbstractVector, λ::Real, f::Function, prox!::Function, 
                    args::NamedTuple, β::Real=.5, ϵ::Real=1e-7)
    
    # Minimal stepsize
    @. x= abs(∇f_x0) 
    λ_min= ϵ*inv(minimum(x))

    # Proximal gradient step
    @. x= x0 - λ*∇f_x0
    prox!(x, λ, args.prox...)

    # f(x)
    f_x= f(x, args.f...)

    # f̂
    @. x= x - x0
    f_hat= f_x0 + dot(∇f_x0, x) + inv(λ+λ)*norm(x)

    # Backtracking line search
    while f_x > f_hat && λ > λ_min
        # Update stepsize
        λ*= β

        # Proximal gradient step
        @. x= x0 - λ*∇f_x0
        prox!(x, λ, args.prox...)

        # Objective function value
        f_x= f(x, args.f...)

        # Update f̂
        @. x= x - x0
        f_hat= f_x0 + dot(∇f_x0, x) + inv(λ+λ)*norm(x)
    end

    # Proximal gradient step
    @. x= x0 - λ*∇f_x0
    prox!(x, λ, args.prox...)

    return λ
end

"""
    prox_grad(x0, f, ∇f!, prox!, args, β=.5, ϵ=1e-7, max_iter=1000)

Minimize an objective function ``f(x) + g(x)``, where ``f(x)`` is differentibale
while ``g(x)`` is not, using the proximal gradient method.

#### Arguments
  - `x0::AbstractVector`: initial parameter values (n x 1)
  - `f::Function`       : ``f(x)``
  - `∇f!::Function`     : gradient of `f`
  - `prox!::Function`   : proximal operator of ``g(x)``
  - `args::NamedTuple`  : arguments for `f`, `∇f!`, and `prox!`
  - `β::Real`           : line search parameter
  - `ϵ::Real`           : tolerance
  - `max_iter::Integer` : max number of iterations

#### Returns
  - `x::AbstractVector` : minimizer (optimal parameter values) (n x 1)
"""
function prox_grad(x0::AbstractVector, f::Function, ∇f!::Function, prox!::Function, 
                    args::NamedTuple, β::Real=.5, ϵ::Real=1e-7, max_iter::Integer=1000)
    # Initialize containers
    x= copy(x0)
    x_prev= similar(x0)
    ∇f_x= similar(x0)

    # Initialize stepsize
    λ= one(Float64)

    # Initialize stopping flag
    stop= false
    # Initialize iteration counter
    iter= 1
    # Proximal gradient method
    while !stop && iter < max_iter
        # Store current parameters
        copyto!(x_prev,x)

        # Current gradient and f(x)
        ∇f!(∇f_x, x, args.∇f...)
        f_x= f(x, args.f...)

        # Backtracking linesearch
        λ= linesearch!(x, x_prev, f_x, ∇f_x, λ, f, prox!, args, β, ϵ)

        # Check for convergence
        stop= true
        @inbounds @fastmath for i in eachindex(x_prev)
            stop*= (abs(x[i] - x_prev[i]) <= ϵ*(1 + abs(x_prev[i])))
        end

        # Update iteration counter
        iter+=1
    end
    
    return x 
end

"""
    prox_grad!(x, f, ∇f!, prox!, args, β=.5, ϵ=1e-7, max_iter=1000)

Minimize an objective function ``f(x) + g(x)``, where ``f(x)`` is differentibale
while ``g(x)`` is not, using the proximal gradient method. Storing the result in
`x`. See also `prox_grad`.
"""
function prox_grad!(x::AbstractVector, f::Function, ∇f!::Function, prox!::Function, 
                    args::NamedTuple, β::Real=.5, ϵ::Real=1e-7, max_iter::Integer=1000)
    # Initialize containers
    x_prev= similar(x)
    ∇f_x= similar(x)

    # Initialize stepsize
    λ= one(Float64)

    # Initialize stopping flag
    stop= false
    # Initialize iteration counter
    iter= 1
    # Proximal gradient method
    while !stop && iter < max_iter
        # Store current parameters
        copyto!(x_prev,x)

        # Current gradient and f(x)
        ∇f!(∇f_x, x, args.∇f...)
        f_x= f(x, args.f...)

        # Backtracking linesearch
        λ= linesearch!(x, x_prev, f_x, ∇f_x, λ, f, prox!, args, β, ϵ)

        # Check for convergence 
        stop= true
        @inbounds @fastmath for i in eachindex(x_prev)
            stop*= (abs(x[i] - x_prev[i]) <= ϵ*(1 + abs(x_prev[i])))
        end

        # Update iteration counter
        iter+=1
    end
    
    return nothing
end

"""
    prox_grad_expol(x0, f, ∇f!, prox!, args, β=.5, ϵ=1e-7, max_iter=1000)

Minimize an objective function ``f(x) + g(x)``, where ``f(x)`` is differentibale
while ``g(x)`` is not, using the accelerated proximal gradient method based on
simple extrapolation.
"""
function prox_grad_expol(x0::AbstractVector, f::Function, ∇f!::Function, prox!::Function, 
                    args::NamedTuple, β::Real=.5, ϵ::Real=1e-7, max_iter::Integer=1000)
    # Initialize containers
    x= copy(x0)
    x_prev= copy(x0)
    y= similar(x0)
    ∇f_y= similar(x0)

    # Initialize stepsize
    λ= one(Float64)

    # Initialize stopping flag
    stop= false
    # Initialize iteration counters
    iter= 1
    k= 1
    # Proximal gradient method
    while !stop && iter < max_iter
        # Extrapolation
        @. y= x - x_prev
        ω= extrapolation(k, ∇f_y, y)
        @. y= x + ω * y

        # Store current parameters
        copyto!(x_prev,x)

        # Current gradient and f(y)
        ∇f!(∇f_y, y, args.∇f...)
        f_y= f(y, args.f...)

        # Backtracking linesearch
        λ= linesearch!(x, y, f_y, ∇f_y, λ, f, prox!, args, β, ϵ)

        # Check for convergence
        stop= true
        @inbounds @fastmath for i in eachindex(x_prev)
            stop*= (abs(x[i] - x_prev[i]) <= ϵ*(1 + abs(x_prev[i])))
        end

        # Update iteration counters
        iter+=1
        k+=1
    end
    
    return x 
end

"""
    prox_grad_expol!(x, f, ∇f!, prox!, args, β=.5, ϵ=1e-7, max_iter=1000)

Minimize an objective function ``f(x) + g(x)``, where ``f(x)`` is differentibale
while ``g(x)`` is not, using the accelerated proximal gradient method based on
simple extrapolation. Storing the result in `x`. See also `prox_grad`.
"""
function prox_grad_expol!(x::AbstractVector, f::Function, ∇f!::Function, prox!::Function, 
                    args::NamedTuple, β::Real=.5, ϵ::Real=1e-7, max_iter::Integer=1000)
    # Initialize containers
    x_prev= copy(x)
    y= similar(x)
    ∇f_y= similar(x)

    # Initialize stepsize
    λ= one(Float64)

    # Initialize stopping flag
    stop= false
    # Initialize iteration counters
    iter= 1
    k= 1
    # Proximal gradient method
    while !stop && iter < max_iter
        # Extrapolation
        @. y= x - x_prev
        ω= extrapolation(k, ∇f_y, y)
        @. y= x + ω * y

        # Store current parameters
        copyto!(x_prev,x)

        # Current gradient and f(y)
        ∇f!(∇f_y, y, args.∇f...)
        f_y= f(y, args.f...)

        # Backtracking linesearch
        λ= linesearch!(x, y, f_y, ∇f_y, λ, f, prox!, args, β, ϵ)

        # Check for convergence
        stop= true
        @inbounds @fastmath for i in eachindex(x_prev)
            stop*= (abs(x[i] - x_prev[i]) <= ϵ*(1 + abs(x_prev[i])))
        end

        # Update iteration counters
        iter+=1
        k+=1
    end
    
    return nothing
end

"""
    prox_grad_nest(x0, f, ∇f!, prox!, args, β=.5, ϵ=1e-7, max_iter=1000)

Minimize an objective function ``f(x) + g(x)``, where ``f(x)`` is differentibale
while ``g(x)`` is not, using the accelerated proximal gradient method based on
Nesterov momentum extrapolation.
"""
function prox_grad_nest(x0::AbstractVector, f::Function, ∇f!::Function, prox!::Function, 
                    args::NamedTuple, β::Real=.5, ϵ::Real=1e-7, max_iter::Integer=1000)
    # Initialize containers
    x= copy(x0)
    x_prev= copy(x0)
    y= similar(x0)
    ∇f_y= similar(x0)

    # Initialize stepsize
    λ= one(Float64)
    λ_prev= one(Float64)

    # Initialize momentum
    θ= one(Float64)

    # Initialize stopping flag
    stop= false
    # Initialize iteration counters
    iter= 1
    k= 1
    # Proximal gradient method
    while !stop && iter < max_iter
        # Extrapolation
        @. y= x - x_prev
        (ω, θ)= nesterov(θ, λ, λ_prev, ∇f_y, y)
        @. y= x + ω * y

        # Store current parameters
        copyto!(x_prev,x)

        # Current gradient and f(y)
        ∇f!(∇f_y, y, args.∇f...)
        f_y= f(y, args.f...)

        # Backtracking linesearch
        λ_prev= λ
        λ= linesearch!(x, y, f_y, ∇f_y, λ_prev, f, prox!, args, β, ϵ)

        # Check for convergence
        stop= true
        @inbounds @fastmath for i in eachindex(x_prev)
            stop*= (abs(x[i] - x_prev[i]) <= ϵ*(1 + abs(x_prev[i])))
        end

        # Update iteration counters
        iter+=1
        k+=1
    end
    
    return x 
end

"""
    prox_grad_nest!(x, f, ∇f!, prox!, args, β=.5, ϵ=1e-7, max_iter=1000)

Minimize an objective function ``f(x) + g(x)``, where ``f(x)`` is differentibale
while ``g(x)`` is not, using the accelerated proximal gradient method based on
Nesterov momentum extrapolation. Storing the result in `x`. See also `prox_grad`.
"""
function prox_grad_nest!(x::AbstractVector, f::Function, ∇f!::Function, prox!::Function, 
                    args::NamedTuple, β::Real=.5, ϵ::Real=1e-7, max_iter::Integer=1000)
    # Initialize containers
    x_prev= copy(x)
    y= similar(x)
    ∇f_y= similar(x)

    # Initialize stepsize
    λ= one(Float64)
    λ_prev= one(Float64)

    # Initialize momentum
    θ= one(Float64)

    # Initialize stopping flag
    stop= false
    # Initialize iteration counters
    iter= 1
    k= 1
    # Proximal gradient method
    while !stop && iter < max_iter
        # Extrapolation
        @. y= x - x_prev
        (ω, θ)= nesterov(θ, λ, λ_prev, ∇f_y, y)
        @. y= x + ω * y

        # Store current parameters
        copyto!(x_prev,x)

        # Current gradient and f(y)
        ∇f!(∇f_y, y, args.∇f...)
        f_y= f(y, args.f...)

        # Backtracking linesearch
        λ_prev= λ
        λ= linesearch!(x, y, f_y, ∇f_y, λ_prev, f, prox!, args, β, ϵ)

        # Check for convergence
        stop= true
        @inbounds @fastmath for i in eachindex(x_prev)
            stop*= (abs(x[i] - x_prev[i]) <= ϵ*(1 + abs(x_prev[i])))
        end

        # Update iteration counters
        iter+=1
        k+=1
    end
    
    return nothing
end