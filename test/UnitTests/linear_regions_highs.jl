using Test, TropicalNN, Oscar, JuMP, Graphs

@testset verbose = true "Linear Regions HiGHS Mode" begin
    R = tropical_semiring(max)

    @testset verbose = true "HiGHS thread count" begin
        threaded_mode = HiGHSMode(threads = 2)
        @test threaded_mode.threads == 2

        model = TropicalNN.create_highs_model(
            solver = threaded_mode.solver,
            threads = threaded_mode.threads
        )
        @test JuMP.get_attribute(model, TropicalNN.MOI.NumberOfThreads()) == 2

        @test_throws ArgumentError TropicalNN.create_highs_model(threads = 0)
    end

    @testset verbose = true "Non-rationalizable Float64 constraints" begin
        coefficient = -0.0009592368123730325
        @test_throws InexactError Rational(coefficient)

        f = Signomial(
            [R(Rational{BigInt}(coefficient)), R(0)],
            [[coefficient], [0.0]];
            sorted = false
        )
        mode = HiGHSMode(threads = 1)

        region = TropicalNN.polyhedron(f, 1, mode)
        @test TropicalNN.get_matrix(region; mode = mode) == [-coefficient;;]
        @test TropicalNN.get_vector(region; mode = mode) == [coefficient]
        @test length(TropicalNN.prune(f; mode = mode)) == length(f)
    end

    @testset verbose = true "Empty polyhedron detection" begin
        A = [1.0 0.0; -1.0 0.0]
        b = [0.0; -1.0]
        @test TropicalNN.highs_is_empty(A, b) == true

        A_feasible = [1.0 0.0; -1.0 0.0]
        b_feasible = [1.0; 0.0]
        @test TropicalNN.highs_is_empty(A_feasible, b_feasible) == false
    end

    @testset verbose = true "Dimension mismatch" begin
        @test_throws DimensionMismatch TropicalNN.highs_intersect_is_full_dimensional(
            zeros(Float64, 0, 1),
            Float64[],
            zeros(Float64, 0, 2),
            Float64[]
        )
    end

    @testset verbose = true "Full dimensional check" begin
        A = [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
        b = [1.0; 1.0; 1.0; 1.0]
        @test TropicalNN.highs_is_full_dimensional(A, b) == true

        A_line = [1.0 0.0; -1.0 0.0]
        b_line = [0.0; 0.0]
        @test TropicalNN.highs_is_full_dimensional(A_line, b_line) == false

        A_infeasible_zero = zeros(Float64, 1, 1)
        b_infeasible_zero = [-1.0]
        @test TropicalNN.highs_is_empty(A_infeasible_zero, b_infeasible_zero) == true
        @test TropicalNN.highs_is_full_dimensional(
            A_infeasible_zero, b_infeasible_zero) == false

        A_tiny_line = [1e-7; -1e-7;;]
        b_tiny_line = [0.0; 0.0]
        @test TropicalNN.highs_is_full_dimensional(A_tiny_line, b_tiny_line) == false
    end

    @testset verbose = true "Tolerance validation" begin
        f1 = Signomial([R(0), R(0)], [[0//1], [1//1]]; sorted = false)
        f2 = Signomial([R(0), R(0)], [[0//1], [-1//1]]; sorted = false)

        for tol in (-1.0, NaN)
            @test_throws ArgumentError linear_regions(
                [f1, f2]; mode = HiGHSMode(tol = tol))
        end
    end

    @testset verbose = true "Codimension one check" begin
        oscar_mode = OscarMode()
        highs_mode = HiGHSMode()
        A = [1.0 0.0;
             -1.0 0.0]
        b = [1.0;
             -1.0]
        @test TropicalNN.codimension_le_one(A, b; mode = oscar_mode) == true
        @test TropicalNN.codimension_le_one(A, b; mode = highs_mode) == true

        # only one point here
        A_point = [1.0 0.0;
                   0.0 1.0;
                   -1.0 0.0;
                   0.0 -1.0]
        b_point = [0.0; 0.0; 0.0; 0.0]
        @test TropicalNN.codimension_le_one(A_point, b_point; mode = highs_mode) == false
        @test TropicalNN.codimension_le_one(A_point, b_point; mode = oscar_mode) == false

        A = [1.0 0.0;
             -1.0 0.0;
             -1.0 0.0]
        b = [1.0;
             -1.0;
             1.0]
        @test TropicalNN.codimension_le_one(A, b; mode = highs_mode) == true
        @test TropicalNN.codimension_le_one(A, b; mode = oscar_mode) == true
    end

    @testset verbose = true "Pruning with threaded HiGHS" begin
        mode = HiGHSMode(threads = 2)
        u = Signomial([R(0), R(0), R(0)], [[0//1], [1//1], [2//1]]; sorted = false)

        @test TropicalNN.prune(u; mode = mode) == TropicalNN.prune(u)

        v = Signomial([R(0)], [[0//1]]; sorted = false)
        q_highs = TropicalNN.prune(u / v; mode = mode)
        q_oscar = TropicalNN.prune(u / v)
        @test q_highs.num == q_oscar.num
        @test q_highs.den == q_oscar.den

        vector_highs = TropicalNN.prune([u / v]; mode = mode)
        @test length(vector_highs) == 1
        @test vector_highs[1].num == q_oscar.num
        @test vector_highs[1].den == q_oscar.den

        W = [Rational{BigInt}.([1 0; 0 1]), Rational{BigInt}.([1 1])]
        b = [Rational{BigInt}.([0, 0]), Rational{BigInt}.([0])]
        t = [Rational{BigInt}.([0, 0])]
        highs_output = tropicalize(
            W,
            b,
            t;
            quicksum = true,
            prune = true,
            elim_mode = mode
        )
        oscar_output = tropicalize(W, b, t; quicksum = true, prune = true)
        @test collect(monomial_pairs(highs_output[1].num)) ==
              collect(monomial_pairs(oscar_output[1].num))
        @test collect(monomial_pairs(highs_output[1].den)) ==
              collect(monomial_pairs(oscar_output[1].den))
    end

    @testset verbose = true "Hoffman calculation with threaded HiGHS" begin
        mode = HiGHSMode(threads = 1)
        f = Signomial(
            [R(0), R(1), R(-1)],
            [[0//1, 0//1], [1//1, 0//1], [0//1, 1//1]];
            sorted = false
        )

        @test exact_er(f; mode = mode) ≈ exact_er(f)
    end

    @testset verbose = true "Graph construction with threaded HiGHS" begin
        mode = HiGHSMode(threads = 1)
        f = Signomial(
            [R(0), R(0), R(0)],
            [[0//1, 0//1], [1//1, 0//1], [0//1, 1//1]];
            sorted = false
        )

        g_highs = TropicalNN.get_graph(f; mode = mode)
        @test Graphs.ne(g_highs) == edge_count(f)
    end
end
