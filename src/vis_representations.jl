# This file contains code to
# obtain the representations of the polyhedra.

@doc"""
    pmap_reps(f)

Computes the matrix representation of the polyhedron corresponding to each monomial in a tropical polynomial.
"""
function pmap_reps(f)
    reps=Dict("m_reps" => [], "f_indices" => [])
    for i in TropicalNN.eachindex(f)
        A=mapreduce(permutedims,vcat,[Float64.(f.exp[j])-Float64.(f.exp[i]) for j in TropicalNN.eachindex(f)])
        b=[Float64(Rational(f.coeff[f.exp[i]]))-Float64(Rational(f.coeff[j])) for j in f.exp]

        p_oscar=Oscar.polyhedron(A,b)
        # Only full dimensional polyhedra will be relevant to the plots
        if Oscar.is_fulldimensional(p_oscar)
            push!(reps["m_reps"],[A,b])
            push!(reps["f_indices"],i)
        end
    end
    return reps
end

@doc""""
    bound_reps(reps,bounding_box)

Takes as input a set of matrix representations for polyhedra, and bounds them in a box determined by `bounding_box`.
"""
function bound_reps(reps,bounding_box)
    function bound_rep(rep,bounding_box)
        n=length(collect(keys(bounding_box)))
        A,b=rep[1],rep[2]
        # bound representations by introducing half spaces to the matrix representation
        for k in 1:n
            # adding the upper bound
            A=vcat(A,[j==k ? 1 : 0 for j in 1:n]')
            push!(b,bounding_box[k][2])
            # adding the lower bound
            A=vcat(A,[j==k ? -1 : 0 for j in 1:n]')
            push!(b,-bounding_box[k][1])
        end
        return [A,b]
    end
    return Dict("m_reps" => [bound_rep(m_rep,bounding_box) for m_rep in reps["m_reps"]], "f_indices" => reps["f_indices"])
end

@doc""""
    bound_reps(reps,rot_matrix)

Identifies the polyhedra, as given by their matrix representations, that intersect the plane defined by the first two-coordinates and rotated according to `rot_matrix`. Returned are the matrix reprsentations of the two-dimensional polyhedra obtained on this intersecting plane.
"""
function project_reps(reps,rot_matrix)
    inv_rot_matrix=inv(rot_matrix)
    projected_reps=Dict("m_reps" => [], "f_indices" => [])
    for (m_rep,f_idx) in zip(reps["m_reps"],reps["f_indices"])
        # rotate the coordinates
        A=m_rep[1]*inv_rot_matrix
        # equivalent to setting the higher dimensions to zero, so that we obtain
        # the polyhedron obtained on the two-dimensional plane of the first two dimensions.
        p_oscar=Oscar.polyhedron(A[:,1:2],m_rep[2])
        if Oscar.is_feasible(p_oscar)
            push!(projected_reps["m_reps"],[A[:,1:2],m_rep[2]])
            push!(projected_reps["f_indices"],f_idx)
        end
    end
    return projected_reps
end

@doc""""
    intersect_reps(rep_1,rep_2)

Returns the matrix representation of a polyhedron obtained by intersecting the polyhedra given by the matrix representations `rep_1` and `rep_2`.
"""
function intersect_reps(rep_1,rep_2)
    # intersect matrix representations by just appending the inequalities
    return [vcat(rep_1[1],rep_2[1]),vcat(rep_1[2],rep_2[2])]
end

@doc""""
    m_reps(f,bounding_box=nothing,rot_matrix=nothing)

Returned are the matrix representations, and the corresponding index of the monomial, of the polyhedron corresponding to each monomial in a tropical polynomial or a tropical rational map. If bounding box is not provided, then the polyhedra are bounded by a region that encompases all the intersections between the polyhedra. If `f` has more than two variables, then a rotation matrix should be supplied so that the returned representations are two-dimensional.
"""
function m_reps(f,bounding_box=nothing,rot_matrix=nothing)
    if typeof(f) <: TropicalNN.TropicalPuiseuxPoly{Rational{BigInt}}
        reps=pmap_reps(f)
    elseif typeof(f) <: TropicalNN.TropicalPuiseuxRational{Rational{BigInt}}
        numerator,denominator=f.num,f.den
        n_reps=pmap_reps(numerator)
        d_reps=pmap_reps(denominator)
    
        reps=Dict("m_reps" => [], "f_indices" => [])
        for (n_m_rep,n_f_idx) in zip(n_reps["m_reps"],n_reps["f_indices"])
            for (d_m_rep,d_f_idx) in zip(d_reps["m_reps"],d_reps["f_indices"])
                # polyhedra of tropical rational maps are obtained by intersecting those of the monomials of
                # the numerator with those of the monomials of the denominator
                int_rep=intersect_reps(n_m_rep,d_m_rep)
                p_oscar=Oscar.polyhedron(int_rep[1],int_rep[2])
                if Oscar.is_fulldimensional(p_oscar)
                    push!(reps["m_reps"],[int_rep[1],int_rep[2]])
                    push!(reps["f_indices"],[n_f_idx,d_f_idx])
                end
            end
        end
    else
        error("The function of focus must be a tropical polynomial or a tropical rational map")
    end
    if bounding_box!=nothing
        # bounding the representations if a box is provided
        reps=bound_reps(reps,bounding_box)
    else
        # bounding the representations by the fully encapsulating box
        reps=bound_reps(reps,get_full_bounding_box(f,reps))
    end
    if rot_matrix!=nothing
        # getting the projected representations
        reps=project_reps(reps,rot_matrix)
    end
    return reps
end

@doc""""
    polyhedra_from_reps(reps)

Given a set of matrix representations of polyhedra, returned are the `Polyhedra.jl` polyhedron representations.
"""
function polyhedra_from_reps(reps)
    return [Polyhedra.polyhedron(Polyhedra.hrep(m_rep[1],m_rep[2]),CDDLib.Library(:exact)) for m_rep in reps["m_reps"]]
end