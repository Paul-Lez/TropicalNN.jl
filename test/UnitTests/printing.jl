using Test, TropicalNN, Oscar

@testset "show / string representations" begin
    R = tropical_semiring(max)

    f0 = Signomial([R(0)], [[0//1, 0//1]]; sorted=false)
    @test sprint(show, f0) == "max(0)"

    f3 = Signomial([R(3)], [[0//1, 0//1]]; sorted=false)
    @test sprint(show, f3) == "max(3)"

    fx = Signomial([R(0)], [[1//1, 0//1]]; sorted=false)
    @test sprint(show, fx) == "max(x₁)"

    f2x = Signomial([R(2)], [[1//1, 0//1]]; sorted=false)
    @test sprint(show, f2x) == "max(2 + x₁)"

    f2 = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]]; sorted=false)
    s2 = sprint(show, f2)
    @test startswith(s2, "max(")
    @test contains(s2, "x₁")
    @test contains(s2, "x₂")

    f_empty = Signomial(Dict{Vector{Rational{Int64}}, eltype(values(f0.coeff))}(),
                        Vector{Vector{Rational{Int64}}}())
    @test sprint(show, f_empty) == "max()"

    den = Signomial([R(0)], [[0//1, 0//1]]; sorted=false)
    q = RationalSignomial(f2, den)
    qs = sprint(show, q)
    @test contains(qs, "⊘")
    @test startswith(qs, "(max(")

    qvec = [q, q]
    sv = sprint(show, qvec)
    @test contains(sv, "⊘")
end
