#=
proximal.jl

    Evaluate the proximal operator for a variety of commonly used functions in
    optimization, data science, machine learning, and econometrics

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/01/13
=#

"""
	soft_thresh(x, λ)
	
Compute soft thresholding operator with scaling parameter `λ` at `x`, proximal
operator of ``ℓ₁``-norm.

#### Arguments
  - `x::Real`	: input
  - `λ::Real`	: scaling parameter

#### Returns
  - `y::Real`	: soft thresholded value
"""
function soft_thresh(x::Real, λ::Real)
	# Soft thresholding
	y= sign(x) * max(abs(x) - λ, zero(λ))

	return y
end

"""
	block_soft_thresh(x, λ)
	
Compute block soft thresholding operator with scaling parameter `λ` at `x`,
proximal operator of the ``ℓ₂``-norm.

#### Arguments
  - `x::AbstractVector`	: input (n x 1)
  - `λ::Real`			: scaling parameter

#### Returns
  - `y::AbstractVector`	: block soft thresholded value (n x 1)
"""
function block_soft_thresh(x::AbstractVector, λ::Real)
    y= similar(x)
	block_soft_thresh!(y, x, λ)

	return y
end

"""
	block_soft_thresh!(y, x, λ)
	
Compute block soft thresholding operator with scaling parameter `λ` at `x`,
proximal operator of the ``ℓ₂``-norm, storing the results in `y`. See also
`block_soft_thresh`.
"""
function block_soft_thresh!(y::AbstractVector, x::AbstractVector, λ::Real)
	T= eltype(x)
	# Scaling
	τ= max(one(λ) - λ * inv(norm(x) + eps(T)), zero(λ))

	# Block soft thresholding
	y.= τ .* x

	return nothing
end

"""
	block_soft_thresh!(x, λ)
	
Compute block soft thresholding operator with scaling parameter `λ` at `x`,
proximal operator of the ``ℓ₂``-norm, overwriting `x`. See also
`block_soft_thresh`.
"""
function block_soft_thresh!(x::AbstractVector, λ::Real)
	T= eltype(x)
	# Scaling
	τ= max(one(λ) - λ * inv(norm(x) + eps(T)), zero(λ))

	# Block soft thresholding
	x.= τ .* x

	return nothing
end

"""
	shrinkage(x, λ)
	
Compute shrinkage operator with scaling parameter `λ` at `x`, proximal operator
of the squared ℓ₂-norm (ridge).

#### Arguments
  - `x::Real`	: input
  - `λ::Real`	: scaling parameter

#### Returns
  - `y::Real`	: shrunken value
"""
function shrinkage(x::Real, λ::Real)
	# Scaling
	τ= one(λ) * inv(one(λ) + λ)

	# Shrinkage
	y= τ * x

	return y
end

"""
    shrinkage!(x, λ, fac, b)

Compute the generalized shrinkage operator with scaling parameter `λ` at `x`,
proximal operator of a quadratic function with quadratic parameters `A` and
linear parameters `b` using a factorization `fac` of ``I + λA``, overwriting
`x`. See also `shrinkage`.
"""
function shrinkage!(
    x::AbstractVector, 
    λ::Real, 
    fac::Factorization, 
    b::AbstractVector
)
    # x - λb
    x.-= λ .* b 
    # (I + λA)⁻¹(x - λb)
    ldiv!(fac, x)

    return nothing
end

"""
    shrinkage(x, λ, fac, b)

Compute the generalized shrinkage operator with scaling parameter `λ` at `x`,
proximal operator of a quadratic function with quadratic parameters `A` and
linear parameters `b` using a factorization `fac` of ``I + λA``.

#### Arguments
  - `x::AbstractVector` : input
  - `λ::Real`           : scaling parameter
  - `fac::Factorization`: factorization of ``I + λA`` 
  - `b::AbstractVector` : linear coefficients

#### Returns
  - `y::AbstractVector` : shrunken values
"""
function shrinkage(
    x::AbstractVector, 
    λ::Real, 
    fac::Factorization, 
    b::AbstractVector
)
    y= similar(x)
    shrinkage!(y, λ, fac, b)

    return y
end

"""
    shrinkage!(x, λ, A, b)

Compute the generalized shrinkage operator with scaling parameter `λ` at `x`,
proximal operator of a quadratic function with quadratic parameters `A` and
linear parameters `b`, overwriting `x`. See also `shrinkage`.
"""
function shrinkage!(
    x::AbstractVector, 
    λ::Real, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    # scaling I + λA
    S= λ .* A
    @inbounds @fastmath for i ∈ axes(A,1)
        S[i,i]+= one(eltype(A))
    end

    # factorization
    C= cholesky!(Hermitian(S))

    shrinkage!(x, λ, C, b)

    return nothing
end

"""
    shrinkage(x, λ, A, b)

Compute the generalized shrinkage operator with scaling parameter `λ` at `x`,
proximal operator of a quadratic function with quadratic parameters `A` and
linear parameters `b`.

#### Arguments
  - `x::AbstractVector` : input
  - `λ::Real`           : scaling parameter
  - `A::AbstractMatrix` : quadratic coefficients
  - `b::AbstractVector` : linear coefficients

#### Returns
  - `y::AbstractVector` : shrunken values
"""
function shrinkage(
    x::AbstractVector, 
    λ::Real, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    y= similar(x)
    shrinkage!(y, λ, A, b)

    return y
end

"""
    smooth(x, λ, f, ∇f!, y_prev)

Compute the proximal operator of a general smooth function `f` with in-place
gradient `∇f!` and scaling parameter `λ` at `x` using L-BFGS. Warm starting is
accomodated by the use of `y_prev`, the previous solution.

#### Arguments
  - `x::AbstractVector`     : input
  - `λ::Real`               : scaling parameter
  - `f::Function`           : objective function
  - `∇f!::Function`         : gradient
  - `y_prev::AbstractVector`: previous output

#### Returns
  - `y::AbstractVector` : output
"""
function smooth(
    x::AbstractVector, 
    λ::Real, 
    f::Function, 
    ∇f!::Function, 
    y_prev::AbstractVector
)
    y= similar(x)
    smooth!(y, x, λ, f, ∇f!, y_prev)

    return y
end

"""
    smooth!(y, x, λ, f, ∇f!, y_prev)

Compute the proximal operator of a general smooth function `f` with in-place
gradient `∇f!` and scaling parameter `λ` at `x` using L-BFGS, storing the
results in `y`. See also `smooth!`. Warm starting is accomodated by the use of
`y_prev`, the previous solution.
"""
function smooth!(
    y::AbstractVector, 
    x::AbstractVector, 
    λ::Real, 
    f::Function, 
    ∇f!::Function, 
    y_prev::AbstractVector
)
    # adjust objective function and gradient
    # f(y) + (1/2λ)‖y - x‖₂² 
    g(y::AbstractVector)= f(y) + inv(λ + λ) * ( sum(abs2, y) + sum(abs2, x) - 2 * dot(y, x) )
    # ∇f(y) + (1/λ)(y - x)
    ∇g!(∇g::AbstractVector, y::AbstractVector)= begin
                                                    # gradient    
                                                    ∇f!(∇g, y)
                                                    # quadratic term
                                                    ∇g.+= inv(λ) .* (y .- x)
                                                end
    
    # solve using L-BFGS
    res= optimize(g, ∇g!, y_prev, LBFGS(), Optim.Options(g_tol = 1e-4))
    y.= Optim.minimizer(res)

    # store current estimate to warm start next iteration
    y_prev.= y

    return nothing
end

"""
    smooth!(x, λ, f, ∇f!, y_prev)

Compute the proximal operator of a general smooth function `f` with in-place
gradient `∇f!` and scaling parameter `λ` at `x` using L-BFGS, overwriting `x`.
See also `smooth!`. Warm starting is accomodated by the use of `y_prev`, the
previous solution.
"""
function smooth!(
    x::AbstractVector, 
    λ::Real, 
    f::Function, 
    ∇f!::Function, 
    y_prev::AbstractVector
)
    # adjust objective function and gradient
    # f(y) + (1/2λ)‖y - x‖₂² 
    g(y::AbstractVector)= f(y) + inv(λ + λ) * ( sum(abs2, y) + sum(abs2, x) - 2 * dot(y, x) )
    # ∇f(y) + (1/λ)(y - x)
    ∇g!(∇g::AbstractVector, y::AbstractVector)= begin
                                                    # gradient    
                                                    ∇f!(∇g, y)
                                                    # quadratic term
                                                    ∇g.+= inv(λ) .* (y .- x)
                                                end
    
    # solve using L-BFGS
    res= optimize(g, ∇g!, y_prev, LBFGS(), Optim.Options(g_tol = 1e-4))
    x.= Optim.minimizer(res)

    # store current estimate to warm start next iteration
    y_prev.= x

    return nothing
end