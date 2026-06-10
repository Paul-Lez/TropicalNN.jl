"""
    TropicalNumber{Convention,T<:Integer}

Tropical number with exact rational storage. `Convention` is `typeof(min)` or
`typeof(max)`; `T` is the integer type of the stored `Rational{T}`.
"""
struct TropicalNumber{Convention, T <: Integer}
    value::Rational{T}
end

# ==============================================================================
# Type aliases for convenience
# ==============================================================================

"""
    TropicalMin{T}

Min-plus tropical semiring: addition is min, multiplication is +
"""
const TropicalMin{T} = TropicalNumber{typeof(min), T}

"""
    TropicalMax{T}

Max-plus tropical semiring: addition is max, multiplication is +
"""
const TropicalMax{T} = TropicalNumber{typeof(max), T}

# ==============================================================================
# Constructors
# ==============================================================================

# From numerator and denominator
function TropicalMin(num::T, den::T = one(T)) where {T <: Integer}
    TropicalMin{T}(Rational{T}(num, den))
end

function TropicalMax(num::T, den::T = one(T)) where {T <: Integer}
    TropicalMax{T}(Rational{T}(num, den))
end

# From Rational
TropicalMin(r::Rational{T}) where {T <: Integer} = TropicalMin{T}(r)
TropicalMax(r::Rational{T}) where {T <: Integer} = TropicalMax{T}(r)

# Promote constructor
function TropicalNumber{C, T}(n::Integer) where {C, T <: Integer}
    TropicalNumber{C, T}(Rational{T}(n))
end

# ==============================================================================
# Basic accessors
# ==============================================================================

"""
    convention(x::TropicalNumber)

Return the convention (min or max) of the tropical number.
"""
Oscar.convention(::TropicalNumber{typeof(min), T}) where {T} = min
Oscar.convention(::TropicalNumber{typeof(max), T}) where {T} = max
Oscar.convention(::Type{TropicalNumber{typeof(min), T}}) where {T} = min
Oscar.convention(::Type{TropicalNumber{typeof(max), T}}) where {T} = max

# ==============================================================================
# Tropical arithmetic (overload standard operators)
# ==============================================================================

"""
    +(x::TropicalMin, y::TropicalMin)

Tropical addition in min-plus semiring: returns minimum.
"""
Base.@inline Base.:+(x::TropicalMin{T}, y::TropicalMin{T}) where {T} = TropicalMin{T}(min(x.value, y.value))

"""
    +(x::TropicalMax, y::TropicalMax)

Tropical addition in max-plus semiring: returns maximum.
"""
Base.@inline Base.:+(x::TropicalMax{T}, y::TropicalMax{T}) where {T} = TropicalMax{T}(max(x.value, y.value))

"""
    *(x::TropicalNumber, y::TropicalNumber)

Tropical multiplication: standard addition of values.
"""
Base.@inline Base.:*(x::TropicalNumber{C, T}, y::TropicalNumber{
    C, T}) where {C, T} = TropicalNumber{C, T}(x.value + y.value)

"""
    /(x::TropicalNumber, y::TropicalNumber)

Tropical division: standard subtraction of values.
"""
Base.@inline Base.:/(x::TropicalNumber{C, T}, y::TropicalNumber{
    C, T}) where {C, T} = TropicalNumber{C, T}(x.value - y.value)

"""
    ^(x::TropicalNumber, n)

Tropical exponentiation: standard scalar multiplication.
"""
Base.@inline Base.:^(x::TropicalNumber{C, T}, n::Integer) where {C, T} = TropicalNumber{
    C, T}(x.value * n)

Base.@inline Base.:^(x::TropicalNumber{C, T}, n::Rational) where {C, T} = TropicalNumber{
    C, T}(x.value * n)

# Unary minus (only makes sense for tropical division by self, but included for completeness)
Base.:-(x::TropicalNumber{C, T}) where {C, T} = TropicalNumber{C, T}(-x.value)

# ==============================================================================
# Infinity handling
# ==============================================================================

"""
    tropical_inf(::Type{TropicalMin{T}})

Return the tropical zero (additive identity) for min-plus semiring: +∞
"""
tropical_inf(::Type{TropicalMin{T}}) where {T} = TropicalMin{T}(typemax(Rational{T}))
tropical_inf(::Type{TropicalMax{T}}) where {T} = TropicalMax{T}(typemin(Rational{T}))

"""
    isinf(x::TropicalNumber)

Check if tropical number represents infinity (tropical zero).
"""
Base.isinf(x::TropicalMin{T}) where {T} = x.value == typemax(Rational{T})
Base.isinf(x::TropicalMax{T}) where {T} = x.value == typemin(Rational{T})

"""
    isfinite(x::TropicalNumber)

Check if tropical number is finite (not tropical zero).
"""
Base.isfinite(x::TropicalNumber) = !isinf(x)

# ==============================================================================
# Identity elements
# ==============================================================================

"""
    zero(::Type{TropicalNumber})

Tropical zero (additive identity): +∞ for min, -∞ for max
"""
Base.zero(::Type{TropicalMin{T}}) where {T} = tropical_inf(TropicalMin{T})
Base.zero(::Type{TropicalMax{T}}) where {T} = tropical_inf(TropicalMax{T})
Base.zero(x::TropicalNumber) = zero(typeof(x))

"""
    one(::Type{TropicalNumber})

Tropical one (multiplicative identity): 0
"""
function Base.one(::Type{TropicalNumber{C, T}}) where {C, T}
    TropicalNumber{C, T}(zero(Rational{T}))
end
Base.one(x::TropicalNumber) = one(typeof(x))

"""
    iszero(x::TropicalNumber)

Check if x is the tropical zero (infinity).
"""
Base.iszero(x::TropicalNumber) = isinf(x)

"""
    isone(x::TropicalNumber)

Check if x is the tropical one (classical zero).
"""
Base.isone(x::TropicalNumber) = x.value == zero(Rational{typeof(x).parameters[2]})

# ==============================================================================
# Comparison operations
# ==============================================================================

function Base.:(==)(x::TropicalNumber{C, T}, y::TropicalNumber{C, T}) where {C, T}
    x.value == y.value
end

function Base.isless(x::TropicalNumber{C, T}, y::TropicalNumber{C, T}) where {C, T}
    isless(x.value, y.value)
end

Base.:<(x::TropicalNumber{C, T}, y::TropicalNumber{C, T}) where {C, T} = x.value < y.value

Base.:<=(x::TropicalNumber{C, T}, y::TropicalNumber{C, T}) where {C, T} = x.value <= y.value

Base.hash(x::TropicalNumber, h::UInt) = hash(x.value, hash(typeof(x), h))

# ==============================================================================
# Conversions
# ==============================================================================

function Base.convert(::Type{TropicalNumber{C, T}}, r::Rational{T}) where {C, T}
    TropicalNumber{C, T}(r)
end

function Base.convert(::Type{TropicalNumber{C, T}}, n::Integer) where {C, T}
    TropicalNumber{C, T}(Rational{T}(n))
end

Base.convert(::Type{Rational{T}}, x::TropicalNumber{C, T}) where {C, T} = x.value

# Promote tropical numbers with same convention
function Base.promote_rule(::Type{TropicalNumber{C, T1}}, ::Type{TropicalNumber{
        C, T2}}) where {C, T1, T2}
    TropicalNumber{C, promote_type(T1, T2)}
end

# ==============================================================================
# Array operations (for performance)
# ==============================================================================

"""
    tropical_sum(arr::AbstractVector{TropicalMin})

Efficient tropical sum (minimum) over array.
"""
function tropical_sum(arr::AbstractVector{TropicalMin{T}}) where {T}
    isempty(arr) && return zero(TropicalMin{T})
    result = arr[1].value
    @inbounds @simd for i in 2:length(arr)
        result = min(result, arr[i].value)
    end
    return TropicalMin{T}(result)
end

function tropical_sum(arr::AbstractVector{TropicalMax{T}}) where {T}
    isempty(arr) && return zero(TropicalMax{T})
    result = arr[1].value
    @inbounds @simd for i in 2:length(arr)
        result = max(result, arr[i].value)
    end
    return TropicalMax{T}(result)
end

"""
    tropical_prod(arr::AbstractVector{TropicalNumber})

Efficient tropical product (sum) over array.
"""
function tropical_prod(arr::AbstractVector{TropicalNumber{C, T}}) where {C, T}
    isempty(arr) && return one(TropicalNumber{C, T})
    result = zero(Rational{T})
    @inbounds @simd for i in Base.eachindex(arr)
        result += arr[i].value
    end
    return TropicalNumber{C, T}(result)
end

# ==============================================================================
# Display
# ==============================================================================

function Base.show(io::IO, x::TropicalMin{T}) where {T}
    if isinf(x)
        print(io, "TropicalMin{$T}(∞)")
    else
        print(io, "TropicalMin{$T}($(x.value))")
    end
end

function Base.show(io::IO, x::TropicalMax{T}) where {T}
    if isinf(x)
        print(io, "TropicalMax{$T}(-∞)")
    else
        print(io, "TropicalMax{$T}($(x.value))")
    end
end

# ==============================================================================
# Special functions
# ==============================================================================

"""
    abs(x::TropicalNumber)

Absolute value in tropical semiring (identity function).
"""
Base.abs(x::TropicalNumber) = x

"""
    min(x::TropicalNumber, y::TropicalNumber)

Classical minimum of underlying values.
"""
function Base.min(x::TropicalNumber{C, T}, y::TropicalNumber{C, T}) where {C, T}
    TropicalNumber{C, T}(min(x.value, y.value))
end

"""
    max(x::TropicalNumber, y::TropicalNumber)

Classical maximum of underlying values.
"""
function Base.max(x::TropicalNumber{C, T}, y::TropicalNumber{C, T}) where {C, T}
    TropicalNumber{C, T}(max(x.value, y.value))
end

# ==============================================================================
# Exports
# ==============================================================================

export TropicalNumber, TropicalMin, TropicalMax
export tropical_inf, tropical_sum, tropical_prod
