# Data structures for neural networks with tropical rational activations.

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
    ActivationLayer(activations)

A neural-network layer of tropical rational activations.

Each activation receives one consecutive block of the input and returns one
scalar. The size of a block is the number of variables in its activation.
"""
struct ActivationLayer{
    T,
    A <: Tuple{RationalSignomial{T}, Vararg{RationalSignomial{T}}}
} <: AbstractNeuralNetworkLayer{T}
    activations::A
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
    NeuralNetwork(layers::Tuple)

An ordered, tuple-backed sequence of compatible neural-network layers.

All layers must have the same scalar type. The network must contain at least
one layer. The output dimension of each layer must equal the input dimension
of the next layer. A `NeuralNetwork` is also a layer, so a network can contain
another network.
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

input_dimension(network::NeuralNetwork) = input_dimension(first(network.layers))
output_dimension(network::NeuralNetwork) = output_dimension(last(network.layers))
