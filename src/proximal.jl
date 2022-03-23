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
	y= sign(x)*max(abs(x) - λ, zero(λ))

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
	T= eltype(x)
	# Scaling
	τ= max(one(λ) - λ*inv(norm(x) + eps(T)), zero(λ))

	# Block soft thresholding
	y= τ*x

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
	τ= max(one(λ) - λ*inv(norm(x) + eps(T)), zero(λ))

	# Block soft thresholding
	@. y= τ*x

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
	τ= max(one(λ) - λ*inv(norm(x) + eps(T)), zero(λ))

	# Block soft thresholding
	@. x= τ*x

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
	τ= one(λ)*inv(one(λ) + λ)

	# Shrinkage
	y= τ*x

	return y
end