using Test, TropicalNN, Oscar

function _test_activation(exponents)
    numerator = Signomial(zeros(Int, length(exponents)), exponents; sorted = false)
    return signomial_to_rational(numerator)
end

@testset verbose = true "Neural network data structures" begin
    unary = _test_activation([[0 // 1], [1 // 1]])
    binary = _test_activation([[1 // 1, 0 // 1], [0 // 1, 1 // 1]])

    @testset "Standard activations" begin
        R = tropical_semiring(max)
        relu_activation = relu()
        leaky_activation = leaky_relu(1 // 4)
        maxout_activation = maxout(3)
        linear_activation = identity_activation()

        @test TropicalNN.evaluate(relu_activation, [-2]) == R(0)
        @test TropicalNN.evaluate(relu_activation, [3]) == R(3)
        @test TropicalNN.evaluate(leaky_activation, [-4]) == R(-1)
        @test TropicalNN.evaluate(leaky_activation, [3]) == R(3)
        @test TropicalNN.evaluate(maxout_activation, [1, 5, 2]) == R(5)
        @test TropicalNN.evaluate(linear_activation, [-3]) == R(-3)
        @test_throws DomainError leaky_relu(-1 // 4)
        @test_throws DomainError leaky_relu(5 // 4)
        @test_throws ArgumentError maxout(0)
    end

    @testset "Affine layers" begin
        R = tropical_semiring(max)
        weight = [1 0; 0 1; 1 -1]
        bias = [1, 2, 3]
        layer = AffineLayer(weight, bias)

        @test input_dimension(layer) == 2
        @test output_dimension(layer) == 3

        weight_view = @view weight[1:2, :]
        bias_view = @view bias[1:2]
        view_layer = AffineLayer(weight_view, bias_view)
        @test TropicalNN.evaluate(tropicalize(view_layer), [2, 3]) ==
              [R(3), R(5)]

        @test_throws DimensionMismatch AffineLayer(
            zeros(Int, 2, 1),
            zeros(Int, 1)
        )
        @test_throws MethodError AffineLayer(zeros(Int, 1, 1), zeros(Float64, 1))
        unsigned_layer = AffineLayer(ones(UInt, 1, 1), zeros(UInt, 1))
        unsigned_function = only(tropicalize(unsigned_layer))
        @test TropicalNN.evaluate(unsigned_function, [2]) == R(2)
    end

    @testset "Activation layers" begin
        relu_layer = ActivationLayer(unary, 3)
        @test input_dimension(relu_layer) == 3
        @test output_dimension(relu_layer) == 3

        mixed_layer = ActivationLayer(unary, binary)
        @test input_dimension(mixed_layer) == 3
        @test output_dimension(mixed_layer) == 2
        @test output_dimension(ActivationLayer([unary, binary])) == 2
        @test input_dimension(ActivationLayer(unary)) == 1

        @test_throws ArgumentError ActivationLayer(unary, 0)
        @test_throws MethodError ActivationLayer(())
        @test_throws MethodError ActivationLayer((unary, relu(Float32)))
    end

    @testset "Network construction" begin
        scalar_type = Rational{Int}
        affine_input = AffineLayer(
            scalar_type.([1 0; 0 1]),
            zeros(scalar_type, 2)
        )
        activation = ActivationLayer(unary, 2)
        affine_output = AffineLayer(scalar_type.([1 1]), zeros(scalar_type, 1))

        network = NeuralNetwork(affine_input, activation, affine_output)
        @test length(network) == 3
        @test collect(network) == [affine_input, activation, affine_output]
        @test network[1] === affine_input
        @test input_dimension(network) == 2
        @test output_dimension(network) == 1
        @test collect(NeuralNetwork([affine_input, activation, affine_output])) ==
              collect(network)

        hidden = NeuralNetwork(affine_input, activation)
        nested = NeuralNetwork(hidden, affine_output)
        @test input_dimension(nested) == 2
        @test output_dimension(nested) == 1

        incompatible_layer = AffineLayer(
            zeros(scalar_type, 1, 3),
            zeros(scalar_type, 1)
        )
        @test_throws DimensionMismatch NeuralNetwork(activation, incompatible_layer)
        @test_throws MethodError NeuralNetwork()

        float_affine = AffineLayer(reshape(Float32[1], 1, 1), Float32[0])
        exact_activation = ActivationLayer(leaky_relu(123456789 // 1000000000))
        @test_throws MethodError NeuralNetwork(float_affine, exact_activation)
    end

    @testset "Layer tropicalization" begin
        R = tropical_semiring(max)
        affine_layer = AffineLayer([1 -1; 0 2], [2, -1])
        affine_functions = only(tropicalize_layers(affine_layer))
        @test TropicalNN.evaluate(affine_functions, [3, 1]) ==
              TropicalNN.evaluate(tropicalize(affine_layer), [3, 1]) ==
              [R(4), R(1)]

        activation_layer = ActivationLayer(relu(), maxout(2))
        activation_functions = only(tropicalize_layers(activation_layer))
        @test TropicalNN.evaluate(activation_functions, [-2, 3, 1]) ==
              TropicalNN.evaluate(tropicalize(activation_layer), [-2, 3, 1]) ==
              [R(0), R(3)]
    end

    @testset "Network tropicalization" begin
        first_weight = Rational{BigInt}.([1 -1; 0 1])
        first_bias = Rational{BigInt}.([1, -1])
        second_weight = Rational{BigInt}.([2 -1])
        second_bias = Rational{BigInt}.([1])

        hidden = NeuralNetwork(
            AffineLayer(first_weight, first_bias),
            ActivationLayer(relu(), 2)
        )
        network = NeuralNetwork(hidden, AffineLayer(second_weight, second_bias))
        layer_maps = tropicalize_layers(network)
        @test length.(layer_maps) == [2, 2, 1]
        @test nvars.(first.(layer_maps)) == [2, 2, 2]
        network_output = only(tropicalize(network))
        legacy_output = only(tropicalize(
            [first_weight, second_weight],
            [first_bias, second_bias]
        ))
        for point in ([0, 0], [2, -1], [-2, 3])
            @test TropicalNN.evaluate(network_output, point) ==
                  TropicalNN.evaluate(legacy_output, point)
        end

        maxout_network = NeuralNetwork(
            AffineLayer(
                [1 0; -1 0; 0 1; 0 -1],
                zeros(Int, 4)
            ),
            ActivationLayer(maxout(Int, 2), 2),
            AffineLayer([1 1], [0])
        )
        maxout_output = only(tropicalize(
            maxout_network;
            quicksum = true,
            dedup = true
        ))
        @test TropicalNN.evaluate(maxout_output, [2, -3]) == tropical_semiring(max)(5)
        @test TropicalNN.evaluate(maxout_output, [-4, 1]) == tropical_semiring(max)(5)

        mixed_network = NeuralNetwork(
            AffineLayer([1 0; 0 1; 1 1], zeros(Int, 3)),
            ActivationLayer(relu(Int), maxout(Int, 2))
        )
        mixed_output = tropicalize(mixed_network)
        @test TropicalNN.evaluate(mixed_output, [-2, 3]) ==
              [tropical_semiring(max)(0), tropical_semiring(max)(3)]

        float_network = NeuralNetwork(
            AffineLayer(reshape([1.0, -1.0], 2, 1), [0.0, 0.0]),
            ActivationLayer(maxout(Float64, 2))
        )
        float_output = only(tropicalize(float_network))
        @test TropicalNN.evaluate(float_output, [-2.0]) == tropical_semiring(max)(2)

        float_leaky_network = NeuralNetwork(
            AffineLayer(reshape(Float32[1], 1, 1), Float32[0]),
            ActivationLayer(leaky_relu(0.125f0))
        )
        float_leaky_output = only(tropicalize(float_leaky_network))
        @test TropicalNN.evaluate(float_leaky_output, Float32[-1]) ==
              tropical_semiring(max)(-1 // 8)
    end
end
