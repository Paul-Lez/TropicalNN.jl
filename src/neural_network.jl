# Data structures for neural networks with tropical rational activations.

"""
    _max_activation(exponents)

Return the maximum of the linear forms with coefficient vectors in `exponents`
as a `RationalSignomial`.
"""
function _max_activation(exponents::AbstractVector{<:AbstractVector{T}}) where {T}
    # Use one tropical monomial for each linear form in the maximum.
    coefficients = [_coefficient_from_scalar(T, 0) for _ in exponents]
    return signomial_to_rational(
        Signomial(coefficients, exponents; sorted = false)
    )
end

"""
    relu()
    relu(T)

Return `x -> max(0, x)` as a `RationalSignomial`. Use exponent type `T`. The
default type is `Rational{BigInt}`.
"""
relu() = relu(Rational{BigInt})
relu(::Type{T}) where {T} = _max_activation([T[zero(T)], T[one(T)]])

"""
    leaky_relu([slope=1 // 100])

Return `x -> max(slope * x, x)` as a `RationalSignomial`. The exponent type is
`typeof(slope)`. The slope must be in the closed interval `[0, 1]`.
"""
leaky_relu() = leaky_relu(Rational{BigInt}(1, 100))

function leaky_relu(slope::T) where {T <: Real}
    zero(slope) <= slope <= one(slope) || throw(DomainError(
        slope,
        "The leaky ReLU slope must be between zero and one"
    ))
    return _max_activation([[slope], [one(slope)]])
end

"""
    maxout(input_dimension)
    maxout(T, input_dimension)

Return `(x₁, ..., xₙ) -> max(x₁, ..., xₙ)` as a `RationalSignomial`, where
`n` is `input_dimension`. Use exponent type `T`. The default type is
`Rational{BigInt}`. The input dimension must be positive.
"""
maxout(input_dimension::Integer) = maxout(Rational{BigInt}, input_dimension)

function maxout(::Type{T}, input_dimension::Integer) where {T}
    input_dimension > 0 || throw(ArgumentError(
        "A maxout activation requires a positive input dimension"
    ))

    # Use each coordinate projection as one term in the tropical maximum.
    exponents = [zeros(T, input_dimension) for _ in 1:input_dimension]
    for coordinate in 1:input_dimension
        exponents[coordinate][coordinate] = one(T)
    end
    return _max_activation(exponents)
end

"""
    identity_activation()
    identity_activation(T)

Return `x -> x` as a `RationalSignomial`. Use exponent type `T`. The default
type is `Rational{BigInt}`.
"""
identity_activation() = maxout(1)
identity_activation(::Type{T}) where {T} = maxout(T, 1)

"""
    AbstractNeuralNetworkLayer{T}

Abstract type for neural-network layers with scalar type `T`.
"""
abstract type AbstractNeuralNetworkLayer{T} end

"""
    AffineLayer(weight, bias)

Store the affine layer `x -> weight * x + bias`. The weight and bias element
types must be `T`. The number of weight rows must equal the bias length.
"""
struct AffineLayer{
    T,
    W <: AbstractMatrix{T},
    B <: AbstractVector{T}
} <: AbstractNeuralNetworkLayer{T}
    weight::W
    bias::B

    function AffineLayer(weight::W, bias::B) where {
            T,
            W <: AbstractMatrix{T},
            B <: AbstractVector{T}
    }
        size(weight, 1) == length(bias) || throw(DimensionMismatch(
            "The affine layer has $(size(weight, 1)) outputs but bias length " *
            "$(length(bias))"
        ))
        return new{T, W, B}(weight, bias)
    end
end

"""
    ActivationLayer(activations...)
    ActivationLayer(activations)
    ActivationLayer(activation, repeats)

Store a layer of tropical rational activations. Split the layer input into
consecutive segments, one segment for each activation. The length of a segment
is the number of variables in its activation. Each activation returns one
scalar. All activations must use scalar type `T`. Use `repeats` to apply one
activation to multiple consecutive segments.
"""
struct ActivationLayer{
    T,
    A <: Tuple{RationalSignomial{T}, Vararg{RationalSignomial{T}}}
} <: AbstractNeuralNetworkLayer{T}
    activations::A
end

ActivationLayer(activation::RationalSignomial{T}) where {T} = ActivationLayer((activation,))

function ActivationLayer(
        first_activation::RationalSignomial{T},
        second_activation::RationalSignomial{T},
        remaining_activations::RationalSignomial{T}...
) where {T}
    return ActivationLayer((first_activation, second_activation, remaining_activations...))
end

function ActivationLayer(
        activations::AbstractVector{<:RationalSignomial{T}}
) where {T}
    return ActivationLayer(tuple(activations...))
end

function ActivationLayer(activation::RationalSignomial{T}, repeats::Integer) where {T}
    repeats > 0 || throw(ArgumentError(
        "The activation repeat count must be positive"
    ))
    return ActivationLayer(ntuple(_ -> activation, repeats))
end

"""
    input_dimension(layer)

Return the number of inputs accepted by a neural-network layer or network.
"""
input_dimension(layer::AffineLayer) = size(layer.weight, 2)

function input_dimension(layer::ActivationLayer)
    return sum(
        activation -> nvars(activation),
        layer.activations;
        init = 0
    )
end

"""
    output_dimension(layer)

Return the number of outputs produced by a neural-network layer or network.
"""
output_dimension(layer::AffineLayer) = size(layer.weight, 1)
output_dimension(layer::ActivationLayer) = length(layer.activations)

"""
    NeuralNetwork(layers...)
    NeuralNetwork(layers)

Store an ordered tuple of compatible neural-network layers. All layers must
use scalar type `T`. The network must contain at least one layer. The output
dimension of each layer must equal the input dimension of the next layer. A
network is also a layer and can contain another network.
"""
struct NeuralNetwork{
    T,
    L <: Tuple{AbstractNeuralNetworkLayer{T}, Vararg{AbstractNeuralNetworkLayer{T}}}
} <: AbstractNeuralNetworkLayer{T}
    layers::L

    function NeuralNetwork(layers::L) where {
            T,
            L <:
            Tuple{AbstractNeuralNetworkLayer{T}, Vararg{AbstractNeuralNetworkLayer{T}}}
    }
        # Check each interface between consecutive layers.
        for layer_index in 2:length(layers)
            previous_dimension = output_dimension(layers[layer_index - 1])
            next_dimension = input_dimension(layers[layer_index])
            previous_dimension == next_dimension || throw(DimensionMismatch(
                "Layer $(layer_index - 1) has output dimension $previous_dimension, " *
                "but layer $layer_index has input dimension $next_dimension"
            ))
        end
        return new{T, L}(layers)
    end
end

function NeuralNetwork(
        layer::AbstractNeuralNetworkLayer{T},
        layers::AbstractNeuralNetworkLayer{T}...
) where {T}
    return NeuralNetwork((layer, layers...))
end

function NeuralNetwork(
        layers::AbstractVector{<:AbstractNeuralNetworkLayer{T}}
) where {T}
    return NeuralNetwork(tuple(layers...))
end

input_dimension(network::NeuralNetwork) = input_dimension(first(network.layers))
output_dimension(network::NeuralNetwork) = output_dimension(last(network.layers))

Base.length(network::NeuralNetwork) = length(network.layers)
Base.getindex(network::NeuralNetwork, index::Integer) = network.layers[index]
Base.firstindex(network::NeuralNetwork) = firstindex(network.layers)
Base.lastindex(network::NeuralNetwork) = lastindex(network.layers)
Base.iterate(network::NeuralNetwork, state...) = iterate(network.layers, state...)
