using Test, TropicalNN, Oscar, Random

@testset verbose = true "Layerwise Linear Regions" begin
    Q = Rational{BigInt}
    modes = (OscarMode(), HiGHSMode())

    function all_full_dimensional(regions, mode)
        return all(
            TropicalNN.is_full_dimensional(piece; mode = mode)
        for region in regions for piece in region
        )
    end

    function pieces_have_mutual_full_dimensional_overlap(left, right, mode)
        left_pieces = [piece for region in left for piece in region]
        right_pieces = [piece for region in right for piece in region]
        left_covered = all(left_pieces) do left_piece
            any(right_pieces) do right_piece
                intersection = TropicalNN.region_intersection(
                    left_piece,
                    right_piece;
                    mode = mode
                )
                TropicalNN.is_full_dimensional(intersection; mode = mode)
            end
        end
        right_covered = all(right_pieces) do right_piece
            any(left_pieces) do left_piece
                intersection = TropicalNN.region_intersection(
                    left_piece,
                    right_piece;
                    mode = mode
                )
                TropicalNN.is_full_dimensional(intersection; mode = mode)
            end
        end
        return left_covered && right_covered
    end

    function compare_workflows(W, b, thresholds; expected = nothing)
        global_expression = mlp_to_trop(W, b, thresholds)
        for mode in modes
            oracle = linear_regions(global_expression; mode = mode)
            layerwise, stats = linear_regions(
                W,
                b,
                thresholds;
                mode = mode,
                return_stats = true
            )

            @test length(layerwise) == length(oracle)
            expected === nothing || @test length(layerwise) == expected
            @test all_full_dimensional(layerwise, mode)
            @test pieces_have_mutual_full_dimensional_overlap(layerwise, oracle, mode)
            @test length(stats) == length(W)
            @test all(stage -> stage.layer_cells > 0, stats)
            @test all(
                stage -> stage.pullback_candidates_tested >=
                         stage.full_dimensional_candidates_retained,
                stats)
            @test all(stage -> stage.affine_map_groups > 0, stats)
            @test all(stage -> stage.glued_components > 0, stats)
            @test all(stage -> stage.elapsed_seconds >= 0, stats)
            @test stats[end].affine_layer_fast_path
        end
    end

    @testset "Single affine layer" begin
        W = [Q.([2 -1; 0 3])]
        b = [Q.([1, -2])]
        compare_workflows(W, b, Vector{Vector{Q}}(); expected = 1)

        layer = TropicalNN.affine_to_trop(W[1], b[1])
        for mode in modes
            @test length(linear_regions([layer]; mode = mode)) == 1
        end
    end

    @testset "One-hidden-layer ReLU MLP" begin
        W = [Q.([1; -1;;]), Q.([1 1])]
        b = [Q.([0, 0]), Q.([0])]
        thresholds = [Q.([0, 0])]
        compare_workflows(W, b, thresholds; expected = 2)
    end

    @testset "Deep vector-valued hidden layers and nonzero thresholds" begin
        W = [
            Q.([1; -1;;]),
            Q.([1 1; 1 -1]),
            Q.([2 -1])
        ]
        b = [Q.([0, 1]), Q.([-1, 0]), Q.([1])]
        thresholds = [Q.([-1, 0]), Q.([0, -2])]
        compare_workflows(W, b, thresholds)
    end

    @testset "Rank-deficient prefix on a later activation boundary" begin
        W = [Q.([0;;]), Q.([1;;]), Q.([1;;])]
        b = [Q.([0]), Q.([0]), Q.([0])]
        thresholds = [Q.([0]), Q.([0])]
        global_expression = mlp_to_trop(W, b, thresholds)

        for mode in modes
            oracle = linear_regions(global_expression; mode = mode)
            layerwise, stats = linear_regions(
                W,
                b,
                thresholds;
                mode = mode,
                return_stats = true
            )
            @test length(oracle) == 1
            @test length(layerwise) == 1
            @test stats[2].full_dimensional_candidates_retained == 1
            @test stats[2].subdivision_strategy == :prefix_conditioned
            @test stats[end].affine_map_groups == 1
            @test stats[end].glued_components == 1
            @test all_full_dimensional(layerwise, mode)
        end

        # The middle monomial is dominant only at y = 0. Its layer cell is
        # lower dimensional in hidden space, but its pullback through y = 0 is
        # the whole original input line and therefore must be tested.
        R = tropical_semiring(max)
        numerator = Signomial(
            [R(0), R(0), R(0)],
            [[0 // 1], [1 // 1], [2 // 1]];
            sorted = false
        )
        denominator = Signomial([R(0)], [[0 // 1]]; sorted = false)
        constant_layer = TropicalNN.affine_to_trop(Q.([0;;]), Q.([0]))
        boundary_layer = [numerator / denominator]
        globally_composed = comp(boundary_layer, constant_layer)

        for mode in modes
            oracle = linear_regions(globally_composed; mode = mode)
            layerwise, stats = TropicalNN._linear_regions_composition(
                [constant_layer, boundary_layer];
                mode = mode
            )
            @test length(oracle) == length(layerwise) == 1
            @test stats[2].layer_cells == 1
            @test stats[2].full_dimensional_candidates_retained == 1
        end
    end

    @testset "Affine pullback preserves layer size and canonicalizes ties" begin
        R = tropical_semiring(max)
        signomial = Signomial(
            [R(0), R(0), R(0)],
            [[0 // 1], [1 // 1], [2 // 1]];
            sorted = false
        )
        pulled = TropicalNN._affine_pullback_signomial(
            signomial,
            zeros(Q, 1, 2),
            zeros(Q, 1)
        )

        @test nvars(pulled) == 2
        @test length(pulled) == 1
        @test collect(TropicalNN.get_exp(pulled, 1)) == zeros(Q, 2)
        @test Rational(TropicalNN.get_coeff(pulled, 1)) == 0

        nonconstant_pullback = TropicalNN._affine_pullback_signomial(
            signomial,
            Q.([1 2]),
            Q.([3])
        )
        @test length(nonconstant_pullback) == length(signomial)
        @test nvars(nonconstant_pullback) == 2
    end

    @testset "Exact boundary provenance is scale invariant" begin
        positive = TropicalNN._canonical_boundary_side(Q.([2, 4]), Q(6))
        scaled = TropicalNN._canonical_boundary_side(Q.([1, 2]), Q(3))
        opposite = TropicalNN._canonical_boundary_side(Q.([-1, -2]), Q(-3))
        @test positive[1] == scaled[1] == opposite[1]
        @test positive[2] == scaled[2]
        @test positive[2] != opposite[2]
        @test TropicalNN._canonical_boundary_side(Q.([0, 0]), Q(1)) === nothing
    end

    @testset "Generic signomial boundaries glue without facet discovery" begin
        R = tropical_semiring(max)
        numerator = Signomial(
            [R(0), R(0), R(-1)],
            [[0 // 1], [1 // 1], [2 // 1]];
            sorted = false
        )
        denominator = Signomial([R(0)], [[0 // 1]]; sorted = false)
        nonlinear_layer = [numerator / denominator]
        zero_layer = TropicalNN.affine_to_trop(Q.([0;;]), Q.([0]))

        for mode in modes
            regions, stats = TropicalNN._linear_regions_composition(
                [nonlinear_layer, zero_layer];
                mode = mode
            )
            @test stats[1].layer_cells == 3
            @test stats[2].affine_map_groups == 1
            @test stats[2].glued_components == 1
            @test length(regions) == 1
        end
    end

    @testset "Zero final affine map cancels all preceding regions" begin
        W = [Q.([1; -1;;]), Q.([0 0])]
        b = [Q.([0, 0]), Q.([3])]
        thresholds = [Q.([0, 0])]
        compare_workflows(W, b, thresholds; expected = 1)
    end

    @testset "Repeated composite map glues across a facet" begin
        W = [Q.([1; -1;;]), Q.([1 -1])]
        b = [Q.([0, 0]), Q.([0])]
        thresholds = [Q.([0, 0])]
        compare_workflows(W, b, thresholds; expected = 1)
    end

    @testset "Codimension-two contact does not glue" begin
        negative_quadrant_A = Q.([1 0; 0 1])
        positive_quadrant_A = Q.([-1 0; 0 -1])
        rhs = Q.([0, 0])
        matrix = zeros(Q, 1, 2)
        offset = zeros(Q, 1)

        for mode in modes
            negative_quadrant = TropicalNN.make_polyhedron(
                negative_quadrant_A,
                rhs;
                mode = mode
            )
            positive_quadrant = TropicalNN.make_polyhedron(
                positive_quadrant_A,
                rhs;
                mode = mode
            )
            pieces = TropicalNN._LabelledAffinePiece[
                TropicalNN._LabelledAffinePiece(
                    negative_quadrant,
                    TropicalNN._region_constraint_data(negative_quadrant; mode = mode)...,
                    matrix,
                    offset
                ),
                TropicalNN._LabelledAffinePiece(
                    positive_quadrant,
                    TropicalNN._region_constraint_data(positive_quadrant; mode = mode)...,
                    matrix,
                    offset
                )
            ]
            _, regions,
            group_count, component_count = TropicalNN._group_labelled_affine_pieces(pieces; mode = mode)
            @test group_count == 1
            @test component_count == 2
            @test length(regions) == 2
        end
    end

    @testset "Small seeded random MLPs" begin
        Random.seed!(20260804)
        for dims in ([1, 2, 1], [2, 2, 1], [1, 2, 2, 1])
            W = [Q.(rand(-2:2, dims[k + 1], dims[k])) for k in 1:(length(dims) - 1)]
            b = [Q.(rand(-1:1, dims[k + 1])) for k in 1:(length(dims) - 1)]
            thresholds = [Q.(rand(-1:1, dims[k + 1])) for k in 1:(length(dims) - 2)]
            compare_workflows(W, b, thresholds)
        end
    end

    @testset "Dimension validation" begin
        mode = HiGHSMode()
        empty_layers = Vector{Vector{RationalSignomial}}()
        empty_weights = Matrix{Q}[]
        empty_biases = Vector{Q}[]
        @test_throws ArgumentError linear_regions(empty_layers; mode = mode)
        @test_throws ArgumentError linear_regions(empty_weights, empty_biases; mode = mode)
        @test_throws DimensionMismatch linear_regions(
            [Q.([1 0])],
            [Q.([0, 0])];
            mode = mode
        )

        first_layer = single_to_trop(Q.([1 0]), Q.([0]), Q.([0]))
        incompatible_layer = TropicalNN.affine_to_trop(Q.([1 0]), Q.([0]))
        @test_throws DimensionMismatch linear_regions(
            [first_layer, incompatible_layer];
            mode = mode
        )
    end
end
