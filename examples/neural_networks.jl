# Example: neural networks with ReLU and maxout activations

using TropicalNN

# This ReLU network has two inputs, three hidden units, and one output.
relu_network = NeuralNetwork(
    AffineLayer(
        [1 0; 0 1; 1 -1],
        [0, 0, 1]
    ),
    ActivationLayer(relu(Int), 3),
    AffineLayer([1 -1 1], [0])
)

# This maxout network has two inputs, three maxout units, and one output.
# The first affine layer returns six values. Each maxout unit receives two
# adjacent values. The activation layer then returns three values.
maxout_network = NeuralNetwork(
    AffineLayer(
        [1 0; -1 0; 0 1; 0 -1; 1 1; -1 -1],
        [0, 0, 0, 0, 1, -1]
    ),
    ActivationLayer(maxout(Int, 2), 3),
    AffineLayer([1 1 1], [0])
)

println("ReLU network:")
println("  dimensions: ", input_dimension(relu_network), " -> ",
    output_dimension(relu_network[1]), " -> ", output_dimension(relu_network))
println("  number of layers: ", length(relu_network))

println("\nMaxout network:")
println("  dimensions: ", input_dimension(maxout_network), " -> ",
    output_dimension(maxout_network[2]), " -> ", output_dimension(maxout_network))
println("  maxout blocks: ", output_dimension(maxout_network[2]), " blocks of size 2")

relu_function = only(tropicalize(relu_network))
maxout_function = only(tropicalize(maxout_network))

println("\nTropicalized outputs:")
println("  ReLU network: ", relu_function)
println("  Maxout network: ", maxout_function)
