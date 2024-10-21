# This file contains the code necessary to obtain the Hoffman constant, with upper and lower bound, for 
# matrices, tropical polynomials and tropical rational functions.

@doc"""
    pmap_exponent_matrix(f)

Returns the matrix of coefficients of the linear maps operating on the polyhedra of the tropical polynomial or tropical rational map.
"""
function pmap_exponent_matrix(f)
    linear_maps=[]
    linearmap_coefficients=[]
    for i in TropicalNN.eachindex(f)
        A=mapreduce(permutedims,vcat,[Float64.(f.exp[j])-Float64.(f.exp[i]) for j in TropicalNN.eachindex(f)])
        b=[Float64(Rational(f.coeff[f.exp[i]]))-Float64(Rational(f.coeff[j])) for j in f.exp]
        p=Oscar.polyhedron(A,b)
        # we only want the linear map that are realised
        if Oscar.is_fulldimensional(p)
            linear_map=[Rational(f.coeff[f.exp[i]]),f.exp[i]]
            # we are only interested in the unique linear map
            if !(linear_map in linear_maps)
                push!(linearmap_coefficients,linear_map[2])
                push!(linear_maps,linear_map)
            end
        end
    end
    A=mapreduce(permutedims, vcat, [Float64.(row) for row in linearmap_coefficients])
    return A
end

@doc"""
    get_tilde_matrices(f)

Finds the transformed 'tilde' matrices whose Hoffman constants are considered when obtaining the Hoffman constant of the corresponding tropical polynomial or tropical rational map.
"""
function get_tilde_matrices(f)
    function tilde_matrix(A,row)
        return A-ones(size(A)[1],1)*reshape(A[row,:],(1,size(A)[2]))
    end
    if typeof(f) <: TropicalPuiseuxPoly{Rational{BigInt}}
        A=pmap_exponent_matrix(f)
        tilde_matrices=[tilde_matrix(A,row) for row in 1:size(A)[1]]
    elseif typeof(f) <: TropicalNN.TropicalPuiseuxRational{Rational{BigInt}}
        A_num=pmap_exponent_matrix(f.num)
        A_den=pmap_exponent_matrix(f.den)
        # we consider each of the possible intersections of polyhedra
        tilde_matrices=[vcat(tilde_matrix(A_num,row_num),tilde_matrix(A_den,row_den)) for row_den in 1:size(A_den)[1], row_num in size(A_num)[1]]
    else
        error("Provide a tropical polynomial or a tropical rational map.")
    end
    return tilde_matrices
end

@doc"""
    surjectivity_test(A)

Solves the optimisation problem which determines whether the matrix has surjectivity with respect to the matrix from which it was sampled.
"""
function surjectivity_test(A)
    n = size(A, 2)
    m = size(A, 1)

    # setting up the model
    model = Model(GLPK.Optimizer)
    @variable(model,x[1:m]>=0)
    @variable(model,t)
    @objective(model,Min,t)
    @constraint(model,[t;A'*x] in MOI.NormOneCone(1+n))
    @constraint(model,sum(x)==1)

    # solving the model
    optimize!(model)
    
    x_val=value.(x)
    t_val=value(t)

    # accounting for any numerical errors
    x_val=map(v->abs(v)<1e-10 ? 0.0 : v, x_val)
    t_val=abs(t_val)<1e-10 ? 0.0 : t_val
    
    return x_val,t_val
end

@doc"""
    mat_exact_hoff(A)

Computes the Hoffman constant of the matrix `A` using a brute force approach.
"""
function mat_exact_hoff(A)
    m = size(A, 1)
    H = 0.0
    # iterating over sub-matrices of A
    for j in 1:m
        subsets = collect(combinations(1:m, j))
        for subset in subsets
            AA=A[subset,:]
            # solving the optimisation problem
            y,t=surjectivity_test(AA) 
            if t > 0
                # in this case the subset is A-surjective
                H = max(H, 1/t)
            end
        end
    end
    return H
end

@doc"""
    mat_upper_hoff(A)

Computes an upper bound on Hoffman constant of the matrix `A` by using the lowest singular value as a proxy for the optimal value of the optimisation problem for A-surjectivity.
"""
function mat_upper_hoff(A)
    m,n=size(A)
    HU=0.0
    # iterating over sub-matrices of A
    for j in 1:m
        subsets = collect(combinations(1:m, j))
        for subset in subsets
            AJ=A[subset,:]
            # only considering full rank sub-matrices
            if rank(AJ)==min(j,n)
                # compute lowest singular value of the sub-matrix
                p_J=minimum(svdvals(AJ))
                if p_J>0
                    HU=max(HU,1/p_J)
                end
            end
        end
    end
    return HU
end

@doc"""
    mat_lower_hoff(A,B=10)

Computes a lower bound on Hoffman constant of the matrix `A` by only considering a fixed number of random sub-matrices of A.
"""
function mat_lower_hoff(A,num_samples=10)
    m,n=size(A)
    HL=0.0
    # if the number of sub-matrices we are considering exceeds the total number of sub-matrices in A
    # we can just use the exact method with no additional computational resources
    if num_samples>=2^m
        HL=mat_exact_hoff(A)
    else
        for i in 1:num_samples
            # consider random sub-matrices
            K=rand(1:m)
            J=rand(1:m,K)
            AJ=A[J,:]
            x,t=surjectivity_test(AJ)
            if t>0
                HL=max(HL,1/t)
            end
        end
    end
    return HL
end

@doc"""
    map_exact_hoff(f)

Returns the exact value of the Hoffman constant of a given tropical polynomial or tropical rational map.
"""
function map_exact_hoff(f)
    hoff_const=0
    for tilde_matrix in get_tilde_matrices(f)
        # constant is taken to be the maximum over each of the tilde matrices
        hoff_const=max(hoff_const,mat_exact_hoff(tilde_matrix))
    end
    return hoff_const
end

@doc"""
    map_upper_hoff(f)

Returns an upper bound on the exact value of the Hoffman constant of a given tropical polynomial or tropical rational map.
"""
function map_upper_hoff(f)
    hoff_upper=0
    for tilde_matrix in get_tilde_matrices(f)
        # to ensure we have an upper bound we need to take the maximum across all upper bounds
        hoff_upper=max(hoff_upper,mat_exact_hoff(tilde_matrix))
    end
    return hoff_upper
end

@doc"""
    map_upper_hoff(f)

Returns a lower bound on the exact value of the Hoffman constant of a given tropical polynomial or tropical rational map.
"""
function map_lower_hoff(f,num_samples=10)
    hoff_lower=Inf
    for tilde_matrix in get_tilde_matrices(f)
        # to ensure we have a lower bound we must take the minimum over all lower bounds
        hoff_lower=min(hoff_lower,mat_lower_hoff(tilde_matrix,num_samples))
    end
    return hoff_lower
end