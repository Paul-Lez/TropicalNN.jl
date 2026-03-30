using Test, TropicalNN, Graphs, MetaGraphsNext

@testset "Statistics" begin

    # one-dimensional tropical polynomial

    f=Signomial(Rational{BigInt}.([0,1,1]),[Rational{BigInt}.([0]),Rational{BigInt}.([1]),Rational{BigInt}.([2])],false)

    bds=bounds(f)
    @test bds==Dict{Any, Any}(Any[0//1, Rational{BigInt}[0]] => Any[Any[false]], Any[1//1, Rational{BigInt}[1]] => Any[Any[true]], Any[1//1, Rational{BigInt}[2]] => Any[Any[false]])

    vols=volumes(f)
    @test vols==Dict{Any, Any}(Any[0//1, Rational{BigInt}[0]] => Any[Inf], Any[1//1, Rational{BigInt}[1]] => Any[1.0], Any[1//1, Rational{BigInt}[2]] => Any[Inf])

    poly_counts=polyhedron_counts(f)
    @test poly_counts==Dict{Any, Any}(Any[0//1, Rational{BigInt}[0]] => [1], Any[1//1, Rational{BigInt}[1]] => [1], Any[1//1, Rational{BigInt}[2]] => [1])

    g=get_graph(f)
    @test typeof(g)<:MetaGraph

    v_collection=vertex_collection(f)
    @test v_collection==Dict{Any, Any}(QQFieldElem[-1] => 1, QQFieldElem[0] => 1)

    v_count=vertex_count(f)
    @test v_count==2

    # two-dimensional tropical polynomial

    f=Signomial(Rational{BigInt}.([7915717918548363//9007199254740992,-7126386568116357//36028797018963968,5429561850506053//18014398509481984,1797871556439715//9007199254740992,-7597859609031347//4503599627370496,6086636570648303//4503599627370496]),[Rational{BigInt}.([6247149566212205//36028797018963968, 6402402109461593//9007199254740992]),Rational{BigInt}.([1560644898352435//1125899906842624, 308028398761065//562949953421312]),Rational{BigInt}.([5123970192597481//18014398509481984, 8573812228218511//9007199254740992]),Rational{BigInt}.([8850358768454271//72057594037927936, 6433642560136419//9007199254740992]),Rational{BigInt}.([3082055910080279//36028797018963968, -2092896630110503//2251799813685248]),Rational{BigInt}.([8780149460643975//18014398509481984, 5087317266127709//36028797018963968])],false)

    bds=bounds(f)
    @test bds==Dict{Any, Any}(Any[5429561850506053//18014398509481984, Rational{BigInt}[5123970192597481//18014398509481984, 8573812228218511//9007199254740992]] => Any[Any[false]], Any[-7126386568116357//36028797018963968, Rational{BigInt}[1560644898352435//1125899906842624, 308028398761065//562949953421312]] => Any[Any[false]], Any[-7597859609031347//4503599627370496, Rational{BigInt}[3082055910080279//36028797018963968, -2092896630110503//2251799813685248]] => Any[Any[false]], Any[7915717918548363//9007199254740992, Rational{BigInt}[6247149566212205//36028797018963968, 6402402109461593//9007199254740992]] => Any[Any[true]], Any[6086636570648303//4503599627370496, Rational{BigInt}[8780149460643975//18014398509481984, 5087317266127709//36028797018963968]] => Any[Any[true]], Any[1797871556439715//9007199254740992, Rational{BigInt}[8850358768454271//72057594037927936, 6433642560136419//9007199254740992]] => Any[Any[false]])

    vols=volumes(f)
    @test sort(collect(values(vols)))≈sort(collect(values(Dict{Any, Any}(Any[5429561850506053//18014398509481984, Rational{BigInt}[5123970192597481//18014398509481984, 8573812228218511//9007199254740992]] => Any[Inf], Any[-7126386568116357//36028797018963968, Rational{BigInt}[1560644898352435//1125899906842624, 308028398761065//562949953421312]] => Any[Inf], Any[-7597859609031347//4503599627370496, Rational{BigInt}[3082055910080279//36028797018963968, -2092896630110503//2251799813685248]] => Any[Inf], Any[7915717918548363//9007199254740992, Rational{BigInt}[6247149566212205//36028797018963968, 6402402109461593//9007199254740992]] => Any[83.85041171269546], Any[6086636570648303//4503599627370496, Rational{BigInt}[8780149460643975//18014398509481984, 5087317266127709//36028797018963968]] => Any[17.689920806367773], Any[1797871556439715//9007199254740992, Rational{BigInt}[8850358768454271//72057594037927936, 6433642560136419//9007199254740992]] => Any[Inf]))))

    g=get_graph(f)
    @test typeof(g)<:MetaGraph

    e_count=edge_count(f)
    @test e_count==11

    e_gradients=edge_gradients(f)
    @test Set(e_gradients[QQFieldElem[114891024932105469462376023557000 // 100773436731723900304245122318991, 251266862377778891172137271729659 // 134364582308965200405660163091988]])==Set(QQFieldElem[43693487181065715//5895790917138212, 19846348181041479//7290715696082942, -4000790818982757//8685640475027672])

    e_lengths=edge_lengths(f)
    @test sort(e_lengths["full"])≈sort(Any[15.414689920279622, 9.544189179644512, 8.094242939512798, 0.4488256365358039, 9.182503972830268, 5.7534317602946095, 6.168847902426598])

    e_directions=edge_directions(f)
    @test sort([sum(e) for e in e_directions["full"]])≈sort([7.551730544461154, 1.283493034112865, 0.26445009382604767, 9.02008836281743, 0.08945139417642001, 0.9770688580278131, 4.73369023556487, 0.5048130632704826, 9.789298309463648, 7.81612742783689, -3.082437192272021])

    v_collection=vertex_collection(f)
    @test v_collection==Dict{Any, Any}(QQFieldElem[114891024932105469462376023557000//100773436731723900304245122318991, 251266862377778891172137271729659//134364582308965200405660163091988] => 3, QQFieldElem[3754412091190030378527856566377417//1037262225301565248686277949717669, -4353036857862963077097275417717051//1037262225301565248686277949717669] => 3, QQFieldElem[-3128137486873165258348604979472//231902538239625862028042208759, -195234271380496802632013468819//231902538239625862028042208759] => 3, QQFieldElem[896451073629643768901810677911431//829990503025703895676636601071105, 236587682011834540634606097327537//165998100605140779135327320214221] => 3, QQFieldElem[-52487495520323989250464295511660//4081230997476490799572483811083, 135808702710554091227834878243691//16324923989905963198289935244332] => 3, QQFieldElem[-483935954748453973516071434347234//122252722226343400701668229762973, -495975158221546378313136293612894//366758166679030202105004689288919] => 3)

    v_count=vertex_count(f)
    @test v_count==6

    # two-dimensional tropical rational map

    w,b,t=random_mlp([2,4,1])
    f=mlp_to_trop_with_strong_elim(w,b,t)[1]
    g=get_graph(f)
    @test typeof(g)<:MetaGraph

    # interior_points tests
    # Use f = max(0, x+1, 2x+1) in 1D.
    # Exponents [0], [1], [2] with coefficients R(0), R(1), R(1).
    # - Monomial [0] (coeff 0): region x ≤ -1        — unbounded ray, one vertex at x=-1
    # - Monomial [1] (coeff 1): region -1 ≤ x ≤ 0   — bounded interval, vertices at x=-1 and x=0
    # - Monomial [2] (coeff 1): region x ≥ 0         — unbounded ray, one vertex at x=0

    @testset "interior_points(Array) — single bounded polyhedron" begin
        R = tropical_semiring(max)
        f_1d = Signomial([R(0), R(1), R(1)], [[0//1], [1//1], [2//1]], false)
        # polyhedron for monomial index 2 (exponent [1//1]) is the interval [-1, 0]
        poly = TropicalNN.polyhedron(f_1d, 2)
        pts = TropicalNN.interior_points([poly])
        @test length(pts) == 1
        # centroid of vertices {-1, 0} is -1/2
        @test Float64.(pts[1]) ≈ [-0.5]
    end

    @testset "interior_points(Array) — multiple polyhedra" begin
        R = tropical_semiring(max)
        f_1d = Signomial([R(0), R(1), R(1)], [[0//1], [1//1], [2//1]], false)
        regions = enum_linear_regions(f_1d)
        polys = [r[1] for r in regions if r[2]]  # all three regions are non-empty
        pts = TropicalNN.interior_points(polys)
        # one interior point per polyhedron
        @test length(pts) == length(polys)
    end

    @testset "interior_points(Dict) — exercises the fixed code path" begin
        R = tropical_semiring(max)
        f_1d = Signomial([R(0), R(1), R(1)], [[0//1], [1//1], [2//1]], false)
        # interior_points(Signomial) routes through interior_points(Dict)
        # via map_statistic → separate_components → interior_points(Dict)
        result = interior_points(f_1d)
        @test result isa Dict
        # All three monomials have full-dimensional regions, so three keys
        @test length(result) == 3
        # Each value is a list of connected components (each component is a list of points)
        for v in values(result)
            @test v isa Vector
            @test all(c isa Vector for c in v)
        end
    end

    @testset "interior_points(Dict) — centroid of bounded region" begin
        R = tropical_semiring(max)
        f_1d = Signomial([R(0), R(1), R(1)], [[0//1], [1//1], [2//1]], false)
        result = interior_points(f_1d)
        # Collect all interior points across all regions and components
        all_pts = [Float64.(pt)
                   for (_, comps) in result
                   for comp in comps
                   for pt in comp]
        # The bounded region [-1, 0] has centroid -0.5; check it appears in the results
        @test any(p ≈ [-0.5] for p in all_pts)
    end

    @testset "interior_points(RationalSignomial) — end-to-end" begin
        # Verify the function runs without error on a rational function
        W, b, t = random_mlp([1, 2, 1])
        trop = mlp_to_trop(W, b, t)[1]
        result = interior_points(trop)
        @test result isa Dict
        @test length(result) > 0
    end

end