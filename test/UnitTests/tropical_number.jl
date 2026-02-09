"""
Unit tests for TropicalNumber type
"""

using Test

# Load the implementation
include("../../src/tropical_number.jl")

@testset "TropicalNumber" begin

    @testset "Constructors" begin
        # Min-plus
        x = TropicalMin(3, 2)
        @test x.value == 3//2
        @test typeof(x) == TropicalMin{Int64}

        y = TropicalMin(5//3)
        @test y.value == 5//3

        z = TropicalMin(2)
        @test z.value == 2//1

        # Max-plus
        a = TropicalMax(1, 2)
        @test a.value == 1//2
        @test typeof(a) == TropicalMax{Int64}
    end

    @testset "Tropical Addition (Min)" begin
        x = TropicalMin(3, 2)  # 3/2
        y = TropicalMin(5, 3)  # 5/3

        # 3/2 < 5/3, so min is 3/2
        z = x + y
        @test z.value == 3//2

        # Commutativity
        @test (x + y).value == (y + x).value

        # With zero (infinity)
        @test (x + zero(TropicalMin{Int64})).value == x.value
    end

    @testset "Tropical Addition (Max)" begin
        x = TropicalMax(3, 2)  # 3/2
        y = TropicalMax(5, 3)  # 5/3

        # 3/2 < 5/3, so max is 5/3
        z = x + y
        @test z.value == 5//3

        # Commutativity
        @test (x + y).value == (y + x).value

        # With zero (negative infinity)
        @test (x + zero(TropicalMax{Int64})).value == x.value
    end

    @testset "Tropical Multiplication" begin
        x = TropicalMin(3, 2)  # 3/2
        y = TropicalMin(2, 1)  # 2

        # 3/2 + 2 = 7/2
        z = x * y
        @test z.value == 7//2

        # Commutativity
        @test (x * y).value == (y * x).value

        # Associativity
        w = TropicalMin(1, 3)
        @test ((x * y) * w).value == (x * (y * w)).value

        # With one (zero)
        @test (x * one(TropicalMin{Int64})).value == x.value
    end

    @testset "Tropical Division" begin
        x = TropicalMin(7, 2)  # 7/2
        y = TropicalMin(2, 1)  # 2

        # 7/2 - 2 = 3/2
        z = x / y
        @test z.value == 3//2

        # Division undoes multiplication
        @test (x * y / y).value == x.value
    end

    @testset "Tropical Exponentiation" begin
        x = TropicalMin(3, 2)  # 3/2

        # (3/2)^3 = 3 * 3/2 = 9/2
        y = x^3
        @test y.value == 9//2

        # (3/2)^(2/3) = 2/3 * 3/2 = 1
        z = x^(2//3)
        @test z.value == 1//1

        # x^0 should be one (tropical)
        @test (x^0).value == 0//1
    end

    @testset "Identity Elements" begin
        # Tropical zero (additive identity)
        z_min = zero(TropicalMin{Int64})
        @test isinf(z_min)
        x = TropicalMin(3, 2)
        @test (x + z_min).value == x.value

        z_max = zero(TropicalMax{Int64})
        @test isinf(z_max)
        y = TropicalMax(3, 2)
        @test (y + z_max).value == y.value

        # Tropical one (multiplicative identity)
        o = one(TropicalMin{Int64})
        @test o.value == 0//1
        @test isone(o)
        @test (x * o).value == x.value
    end

    @testset "Comparison" begin
        x = TropicalMin(3, 2)
        y = TropicalMin(5, 3)
        z = TropicalMin(3, 2)

        @test x == z
        @test x != y
        @test x < y
        @test x <= y
        @test x <= z

        # Hash consistency
        @test hash(x) == hash(z)
        @test hash(x) != hash(y)
    end

    @testset "Infinity Handling" begin
        # Min-plus infinity is +∞
        inf_min = tropical_inf(TropicalMin{Int64})
        @test isinf(inf_min)
        @test !isfinite(inf_min)
        @test iszero(inf_min)

        # Max-plus infinity is -∞
        inf_max = tropical_inf(TropicalMax{Int64})
        @test isinf(inf_max)
        @test !isfinite(inf_max)
        @test iszero(inf_max)

        # Regular numbers are finite
        x = TropicalMin(3, 2)
        @test isfinite(x)
        @test !isinf(x)
        @test !iszero(x)
    end

    @testset "Array Operations" begin
        # Min-plus sum
        arr_min = [TropicalMin(i, 1) for i in 1:10]
        result = tropical_sum(arr_min)
        @test result.value == 1//1  # minimum is 1

        # Max-plus sum
        arr_max = [TropicalMax(i, 1) for i in 1:10]
        result = tropical_sum(arr_max)
        @test result.value == 10//1  # maximum is 10

        # Tropical product
        arr = [TropicalMin(i, 1) for i in 1:5]
        result = tropical_prod(arr)
        @test result.value == (1+2+3+4+5)//1  # sum is 15
    end

    @testset "Convention" begin
        x_min = TropicalMin(3, 2)
        x_max = TropicalMax(3, 2)

        @test convention(x_min) == min
        @test convention(x_max) == max
        @test convention(typeof(x_min)) == min
        @test convention(typeof(x_max)) == max
    end

    @testset "Distributivity (tropical)" begin
        # Tropical multiplication distributes over tropical addition
        x = TropicalMin(1, 1)
        y = TropicalMin(2, 1)
        z = TropicalMin(3, 1)

        # x * (y + z) = (x * y) + (x * z)
        lhs = x * (y + z)
        rhs = (x * y) + (x * z)
        @test lhs.value == rhs.value
    end

    @testset "Conversions" begin
        r = 3//2
        x = convert(TropicalMin{Int64}, r)
        @test x.value == r

        y = TropicalMin(3, 2)
        r2 = convert(Rational{Int64}, y)
        @test r2 == 3//2

        # From integer
        z = TropicalMin(5)
        @test z.value == 5//1
    end

    @testset "Type Stability" begin
        x = TropicalMin(3, 2)
        y = TropicalMin(5, 3)

        # Operations should return same type
        @test typeof(x + y) == typeof(x)
        @test typeof(x * y) == typeof(x)
        @test typeof(x / y) == typeof(x)
        @test typeof(x^2) == typeof(x)
    end

    @testset "Edge Cases" begin
        # Very large numbers
        x = TropicalMin(typemax(Int64) ÷ 2, 1)
        y = TropicalMin(typemax(Int64) ÷ 2, 1)
        @test (x + y).value == x.value  # min of same values

        # Negative numbers
        x = TropicalMin(-5, 2)
        y = TropicalMin(3, 1)
        @test (x + y).value == -5//2  # min is negative
    end
end
