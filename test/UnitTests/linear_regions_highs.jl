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

    @testset verbose = true "GLPK fallback" begin
        calls = 0
        solve_lp = function (model)
            calls += 1
            if calls == 1
                return nothing, TropicalNN.MOI.OTHER_ERROR
            end
            @test JuMP.solver_name(model) == "GLPK"
            return true, TropicalNN.MOI.OPTIMAL
        end

        highs_model = TropicalNN.create_highs_model()
        @test TropicalNN._solve_with_glpk_fallback(
            solve_lp, highs_model, "test check")
        @test calls == 2
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
        for row_scale in (1.0, 1.0e-10)
            @test TropicalNN.highs_is_empty(row_scale .* A, row_scale .* b) == true
            @test TropicalNN.codimension_le_one(
                row_scale .* A, row_scale .* b; mode = HiGHSMode()) == false
        end

        A_feasible = [1.0 0.0; -1.0 0.0]
        b_feasible = [1.0; 0.0]
        @test TropicalNN.highs_is_empty(A_feasible, b_feasible) == false

        @test TropicalNN.highs_is_empty(zeros(Float64, 1, 1), [0.0]) == false
    end

    @testset verbose = true "Boundedness check" begin
        mode = HiGHSMode(solver = "simplex", threads = 1)
        cases = (
            ([1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0], ones(4), true),
            ([1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0], [1.0, 0.0, 0.0, 0.0], true),
            ([1.0 0.0; -1.0 0.0], ones(2), false),
            ([-1.0 0.0; 0.0 1.0; 0.0 -1.0], [0.0, 1.0, 1.0], false),
            ([1.0 0.0; -1.0 0.0; 0.0 -1.0], zeros(3), false),
            (zeros(Float64, 0, 2), Float64[], false),
            ([1.0 0.0; -1.0 0.0], [0.0, -1.0], true),
            (zeros(Float64, 1, 2), [-1.0], true),
            (zeros(Float64, 0, 0), Float64[], true),
            (zeros(Float64, 1, 0), [-1.0], true)
        )

        for (A, b, expected) in cases
            @test TropicalNN.highs_is_bounded(A, b) == expected
            region = TropicalNN.make_polyhedron(A, b; mode = mode)
            @test TropicalNN.is_bounded(region; mode = mode) == expected
        end

        A = [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
        b = ones(4)
        row_scales = [1.0e-10, 1.0, 1.0e10, 1.0]
        @test TropicalNN.highs_is_bounded(row_scales .* A, row_scales .* b)
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

        A_tiny_line = [1e-10; -1e-10;;]
        b_tiny_line = [0.0; 0.0]
        @test TropicalNN.highs_is_full_dimensional(A_tiny_line, b_tiny_line) == false

        A_interval = [1.0; -1.0;;]
        b_interval = [2.0; 2.0]
        @test TropicalNN.highs_is_full_dimensional(
            A_interval, b_interval; tol = 1.5) == true

        @test TropicalNN.highs_is_full_dimensional([1.0;;], [0.0])
    end

    @testset verbose = true "Row scaling invariance" begin
        modes = (OscarMode(), HiGHSMode())
        A = [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
        b = fill(0.01, 4)
        full_dimensional = (matrix, vector, mode) -> TropicalNN.is_full_dimensional(
            TropicalNN.make_polyhedron(matrix, vector; mode = mode); mode = mode)
        baseline_results = [full_dimensional(A, b, mode) for mode in modes]
        @test baseline_results == [true, true]

        for i in axes(A, 1)
            A_scaled = copy(A)
            b_scaled = copy(b)
            A_scaled[i, :] .*= 1.0e-10
            b_scaled[i] *= 1.0e-10

            scaled_results = [full_dimensional(A_scaled, b_scaled, mode) for mode in modes]
            @test scaled_results == baseline_results
            @test TropicalNN.highs_check_implicit_equality(A_scaled, b_scaled, i) ==
                  TropicalNN.highs_check_implicit_equality(A, b, i)
        end
    end

    @testset verbose = true "Implicit equality check" begin
        A_line = [1.0; -1.0;;]
        b_line = [0.0; 0.0]
        row_scale = 1.0e-10
        @test TropicalNN.highs_check_implicit_equality(
            row_scale .* A_line, row_scale .* b_line, 1) ==
              TropicalNN.highs_check_implicit_equality(A_line, b_line, 1)

        A_constant = zeros(Float64, 1, 1)
        @test TropicalNN.highs_check_implicit_equality(A_constant, [0.0], 1)
        @test !TropicalNN.highs_check_implicit_equality(A_constant, [1.0], 1)
    end

    @testset verbose = true "Tolerance validation" begin
        f1 = Signomial([R(0), R(0)], [[0//1], [1//1]]; sorted = false)
        f2 = Signomial([R(0), R(0)], [[0//1], [-1//1]]; sorted = false)

        for tol in (-1.0, NaN, Inf)
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

        A_point_scaled = copy(A_point)
        b_point_scaled = copy(b_point)
        A_point_scaled[3:4, :] .*= 1.0e-20
        b_point_scaled[3:4] .*= 1.0e-20
        @test TropicalNN.codimension_le_one(
            A_point_scaled, b_point_scaled; mode = highs_mode) == false

        A = [1.0 0.0;
             -1.0 0.0;
             -1.0 0.0]
        b = [1.0;
             -1.0;
             1.0]
        @test TropicalNN.codimension_le_one(A, b; mode = highs_mode) == true
        @test TropicalNN.codimension_le_one(A, b; mode = oscar_mode) == true
    end

    @testset verbose = true "Facet tolerance" begin
        matrix = zeros(Float64, 1, 2)
        offset = [0.0]
        cell = (A, b) -> TropicalNN._Cell(A, b, matrix, offset, ())
        hyperplane_key = first(TropicalNN._canonical_halfspace_key([1.0, 0.0], 0.0))

        left = cell(
            [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0],
            [0.0, 3.0, 2.0, 2.0]
        )
        right = cell(
            [-1.0 0.0; 1.0 0.0; 0.0 1.0; 0.0 -1.0],
            [0.0, 3.0, 2.0, 2.0]
        )
        for tol in (0.5, 1.5)
            @test TropicalNN._highs_cells_share_facet(
                left, right, hyperplane_key, HiGHSMode(tol = tol))
        end

        unbounded_left = cell([1.0 0.0], [0.0])
        unbounded_right = cell([-1.0 0.0], [0.0])
        @test TropicalNN._highs_cells_share_facet(
            unbounded_left, unbounded_right, hyperplane_key, HiGHSMode(tol = 1.5))
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
