############### Utilities ###############

@doc raw"""
    linearmap_matrices(f::TropicalPuiseuxPoly)

Returns the matrix of coefficients of the linear maps operating on the polyhedra of a tropical polynomial.
"""
function linearmap_matrices(f::TropicalPuiseuxPoly)
    linear_maps=[]
    exponents=[]
    coefficients=[]
    for i in eachindex(f)
        A=mapreduce(permutedims,vcat,[Float64.(f.exp[j])-Float64.(f.exp[i]) for j in eachindex(f)])
        b=[Float64(Rational(f.coeff[f.exp[i]]))-Float64(Rational(f.coeff[j])) for j in f.exp]
        p=Oscar.polyhedron(A,b)
        # we only want the linear map that are realised
        if Oscar.is_fulldimensional(p)
            linear_map=[Rational(f.coeff[f.exp[i]]),f.exp[i]]
            # we are only interested in the unique linear map
            if !(linear_map in linear_maps)
                push!(exponents,linear_map[2])
                push!(coefficients,linear_map[1])
                push!(linear_maps,linear_map)
            end
        end
    end
    A=mapreduce(permutedims, vcat, [Float64.(row) for row in exponents])
    b=vec(coefficients)
    return A,b
end

@doc raw"""
    linearmap_matrices(f::TropicalPuiseuxRational)

Returns the matrix of coefficients of the linear maps operating on the polyhedra of a tropical rational map.
"""
function linearmap_matrices(f::TropicalPuiseuxRational)
    Anum,bnum=linearmap_matrices(f.num)
    Aden,bden=linearmap_matrices(f.den)
    return (Anum,Aden),(bnum,bden)
end

@doc raw"""
    tilde_matrices(A::Matrix)

Finds all of the transformed 'tilde' matrices whose Hoffman constants are considered when obtaining the Hoffman constant of the corresponding tropical polynomial.
"""
function tilde_matrices(A::Matrix)
    m,n=size(A)
    ones_vector=ones(m,1)
    return [A-ones_vector*reshape(A[row,:],(1,n)) for row in 1:m]
end

@doc raw"""
    tilde_matrices(As::Tuple{Matrix, Matrix})

Finds all of the transformed 'tilde' matrices whose Hoffman constants are considered when obtaining the Hoffman constant of the corresponding tropical rational map.
"""
function tilde_matrices(As::Tuple{Matrix, Matrix})
    m_1,n=size(As[1])
    m_2=size(As[2])[1]
    ones_matrix=ones(m_1+m_2,2)
    return [vcat(As[1],As[2])-ones_matrix*vcat(As[1][row_num:row_num,:],As[2][row_den:row_den,:]) for row_den in 1:m_2, row_num in 1:m_1]
end

@doc raw"""
    tilde_vectors(b::Vector)

Find the transformed vectors used to determine the effective radius of a tropical polynomial
"""
function tilde_vectors(b::Vector)
    return [b-b[row]*ones(length(b)) for row in 1:length(b)]
end

@doc raw"""
    positive_component(b::Vector)

Returns the vector with its negative entries set to zero.
"""
function positive_component(b::Vector)
    return vec([max(0,entry) for entry in b])
end

############### Hoffman Algorithms ###############

@doc raw"""
    surjectivity_test(A)

Solves the optimisation problem which determines whether the matrix has surjectivity with respect to the matrix from which it was sampled.
"""
function surjectivity_test(A::Matrix)
    n = size(A, 2)
    m = size(A, 1)

    # setting up the model
    model=Model(GLPK.Optimizer)
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

@doc raw"""
    exact_hoff(A::Matrix)

Computes the Hoffman constant of the matrix `A` using a brute force approach.
"""
function exact_hoff(A::Matrix)
    m=size(A, 1)
    H=0.0
    # iterating over sub-matrices of A
    for j in 1:m
        subsets = collect(combinations(1:m, j))
        for subset in subsets
            AA=A[subset,:]
            # solving the optimisation problem
            y,t=surjectivity_test(AA) 
            if t>0
                # in this case the subset is A-surjective
                H=max(H,1/t)
            end
        end
    end
    return H
end

@doc raw"""
    upper_hoff(A::Matrix)

Computes an upper bound on Hoffman constant of the matrix `A` by using the lowest singular value as a proxy for the optimal value of the optimisation problem for A-surjectivity.
"""
function upper_hoff(A::Matrix)
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

@doc raw"""
    lower_hoff(A::Matrix,num_samples::Int=10)

Computes a lower bound on Hoffman constant of the matrix `A` by only considering a fixed number of random sub-matrices of A.
"""
function lower_hoff(A::Matrix,num_samples::Int=10)
    m,n=size(A)
    HL=0.0
    # if the number of sub-matrices we are considering exceeds the total number of sub-matrices in A
    # we can just use the exact method with no additional computational resources
    if num_samples>=2^m
        HL=exact_hoff(A)
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


@doc raw"""
    exact_hoff(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational};return_matrices::Bool=false)

Returns the exact value of the Hoffman constant of a given tropical polynomial or tropical rational map.
"""
function exact_hoff(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational};return_matrices::Bool=false)
    hoff_const=0
    A,b=linearmap_matrices(f)
    for tilde_matrix in tilde_matrices(A)
        # constant is taken to be the maximum over each of the tilde matrices
        hoff_const=max(hoff_const,exact_hoff(tilde_matrix))
    end
    if return_matrices
        return hoff_const,A,b
    else
        return hoff_const
    end
end

@doc raw"""
    upper_hoff(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational};return_matrices::Bool=false)

Returns an upper bound on the exact value of the Hoffman constant of a given tropical polynomial or tropical rational map.
"""
function upper_hoff(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational};return_matrices::Bool=false)
    hoff_upper=0
    A,b=linearmap_matrices(f)
    for tilde_matrix in tilde_matrices(A)
        # to ensure we have an upper bound we need to take the maximum across all upper bounds
        hoff_upper=max(hoff_upper,upper_hoff(tilde_matrix))
    end
    if return_matrices
        return hoff_upper,A,b
    else
        return hoff_upper
    end
end

@doc raw"""
    lower_hoff(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational},num_samples::Int=10)

Returns a lower bound on the exact value of the Hoffman constant of a given tropical polynomial or tropical rational map.
"""
function lower_hoff(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational},num_samples::Int=10;return_matrices::Bool=false)
    A,b=linearmap_matrices(f)
    t_matrices=tilde_matrices(A)
    # if we are taking more samples than there are submatrices we are using exact
    # computations so we can take a maximum over the Hoffman constants
    if num_samples>=2^(size(t_matrices[1])[1])
        hoff_lower=0.0
        for tilde_matrix in t_matrices
            hoff_lower=max(hoff_lower,lower_hoff(tilde_matrix,num_samples))
        end
    # otherwise, to ensure we have a lower bound we must take the minimum 
    # over all lower bounds
    else
        hoff_lower=Inf
        for tilde_matrix in t_matrices
            hoff_lower=min(hoff_lower,lower_hoff(tilde_matrix,num_samples))
        end
    end
    if return_matrices
        return hoff_lower,A,b
    else
        return hoff_lower
    end
end

############### Effective Radius ###############

@doc raw"""
    exact_er(f::TropicalPuiseuxPoly)

Provides an upper bound on the effective radius of a tropical polynomial using exact Hoffman constant computations.
"""
function exact_er(f::TropicalPuiseuxPoly)
    hoff_const,A,b=exact_hoff(f,return_matrices=true)
    tilde_bs=tilde_vectors(b)
    return hoff_const*maximum([norm(positive_component(tilde_b),Inf) for tilde_b in tilde_bs])
end

@doc raw"""
    upper_er(f::TropicalPuiseuxPoly)

Provides an upper bound on the effective radius of a tropical polynomial using upper bound approximations of the Hoffman constant.
"""
function upper_er(f::TropicalPuiseuxPoly)
    hoff_upper,A,b=upper_hoff(f,return_matrices=true)
    tilde_bs=tilde_vectors(b)
    return hoff_upper*maximum([norm(positive_component(tilde_b),Inf) for tilde_b in tilde_bs])
end

@doc raw"""
    exact_er(f::TropicalPuiseuxRational)

Provides an upper bound on the effective radius of a tropical rational map using exact Hoffman constant computations.
"""
function exact_er(f::TropicalPuiseuxRational)
    hoff_const,A,b=exact_hoff(f,return_matrices=true)
    return hoff_const*max(maximum(b[1])-minimum(b[1]),maximum(b[2])-minimum(b[2]))
end

@doc raw"""
    upper_er(f::TropicalPuiseuxRational)

Provides an upper bound on the effective radius of a tropical rational map using upper bound approximations of the Hoffman constant.
"""
function upper_er(f::TropicalPuiseuxRational)
    hoff_upper,A,b=upper_hoff(f,return_matrices=true)
    return hoff_upper*max(maximum(b[1])-minimum(b[1]),maximum(b[2])-minimum(b[2]))
end