# This code contains the functions necessary to visualise the linear regions
# of tropical polynomials and tropical rational maps.

####################### Utility Functions ##################################

@doc"""
    random_pmap(n_vars,n_mons)

Returns a random tropical polynomial in `n_vars` variables with `n_mons` monomials.
"""
function random_pmap(n_vars,n_mons) # move to main package
    return TropicalPuiseuxPoly(Rational{BigInt}.(rand(Normal(0,1/sqrt(2)),n_mons)),[Rational{BigInt}.(rand(Normal(0,1/sqrt(2)),n_vars)) for _ in 1:n_mons],true)
end

@doc"""
    update_bounding_box(bounding_box,p)

Expands the bounding box to contain the vertices of the polyhedron `p`.
"""
function update_bounding_box(bounding_box,p)
    vertices=collect(Polyhedra.points(p))
    for vertex in vertices
        for k in collect(keys(bounding_box))
            # updates the max and min values along each dimension
            bounding_box[k][1]=min(bounding_box[k][1],vertex[k])
            bounding_box[k][2]=max(bounding_box[k][2],vertex[k])
        end
    end
    return bounding_box
end

@doc"""
    get_full_bounding_box(f,reps)

Computes the bounding box containing all intersection between the polyhedra of a tropical polynomial or tropical rational map.
"""
function get_full_bounding_box(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational},reps)
    n=nvars(f)
    bounding_box=Dict(dimension => [0.0,0.0] for dimension in 1:n)
    for m_rep in reps["m_reps"]
        p_oscar=Oscar.polyhedron(m_rep[1],m_rep[2])
        # only full dimensional polyhedra will be visible
        if Oscar.is_fulldimensional(p_oscar)
            p=Polyhedra.polyhedron(Polyhedra.hrep(m_rep[1],m_rep[2]),CDDLib.Library())
            # updates the bounding box with the polyhedron
            bounding_box=update_bounding_box(bounding_box,p)
        end
    end
    # slightly enlarging the bounding box such that all vertice are visible in the plots
    for k in collect(keys(bounding_box))
        bounding_box[k][1]-=1
        bounding_box[k][2]+=1
    end
    return bounding_box
end

@doc"""
    apply_linear_map(point,linear_map)

Returns the output of `linear_map` at `point`.
"""
function apply_linear_map(point,linear_map)
    return sum([linear_map[2][j]*point[j] for j in 1:length(point)])+linear_map[1]
end

@doc"""
    get_level_set_component(poly,linear_map,value)

Returns components of points along the vertices and edges of `poly` that form the level set of value `value` for the map `linear_map`.
"""
function get_level_set_component(poly,linear_map,value)
    component=[]
    vertices=collect(Polyhedra.points(poly))
    functional_values=[apply_linear_map(vertex,linear_map) for vertex in vertices]
    for pair in Combinatorics.combinations(range(1,length(functional_values)),2)
        values=functional_values[pair]
        vertexs=vertices[pair]
        # using the continuity of the linear map over the one-dimensional edges to note when the
        # level set is crossed
        if (values[1]-value)*(values[2]-value)<0
            push!(component,vertexs[1]+(vertexs[2]-vertexs[1])*(value-values[1])/(values[2]-values[1]))
        elseif (values[1]-value)*(values[2]-value)==0 && value!=0
            # in this case the entire edge is on the level set so we add the vertex such that
            # the edges are connected in the plot
            push!(component,vertexs[1])
        # we deal with the case when the value is zero as the above condition could be acheived
        # when either one of values is zero, rather than both
        elseif value==0 
            if values[1]==0
                push!(component,vertexs[1])
            elseif values[2]==0
                push!(component,vertexs[2])
            end
        end
    end
    return component
end

@doc"""
    get_surface_points(poly,linear_map)

Returns the coordinates of the vertices of `poly` and the output of `linear_map` applied to those points, such that the action of `linear_map` on `poly` can be visualised.
"""
function get_surface_points(poly,linear_map)
    poly_vertices=collect(Polyhedra.points(poly))
    input_dim=length(poly_vertices[1])
    # splitting the vertices of the polyhedron into dimensions
    input_vertices=[[poly_vertex[k] for poly_vertex in poly_vertices] for k in 1:input_dim]
    # applying the linear map to each vertex of the polyhedron
    output_vertices=[apply_linear_map(poly_vertex,linear_map) for poly_vertex in poly_vertices]
    return input_vertices,output_vertices
end

@doc"""
    get_linear_maps(f,f_indices)

Comptues the linear maps corresponding to the monomials with indices `f_indices` in `f`.
"""
function get_linear_maps(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational},f_indices)
    linear_maps=[]
    for f_idx in f_indices
        if length(f_idx)==1
            push!(linear_maps,[Rational(f.coeff[f.exp[f_idx]]),f.exp[f_idx]])
        else
            numr,denr=f.num,f.den
            push!(linear_maps,[Rational(numr.coeff[numr.exp[f_idx[1]]])-Rational(denr.coeff[denr.exp[f_idx[2]]]),numr.exp[f_idx[1]]-denr.exp[f_idx[2]]])
        end
    end
    return linear_maps
end

@doc"""
    get_linear_regions(polyhedra,linear_maps,with_colors::Bool=true)

For each unique linear map, the polyhedra that are acted on by this linear map are identified. Moreover, a color is attributed to each unique linear map such that they can be distinguished in the visualisations.
"""
function get_linear_regions(polyhedra,linear_maps)
    # color coding each distinct linear map
    num_distinct_linear_maps=length(Set(linear_maps))
    colors=shuffle([ColorSchemes.get(ColorSchemes.Paired_8,(i-1)/(max(1,num_distinct_linear_maps-1))) for i in 1:num_distinct_linear_maps])
    linear_regions=Dict()
    relative_index=1
    for (linear_map,poly) in zip(linear_maps,polyhedra)
        # add a region if the linear map is distinct from the previous
        # otherwise append the polyhedron to the existing region
        if !haskey(linear_regions,linear_map)
            linear_regions[linear_map]=Dict("color" => colors[relative_index], "polyhedra" => [poly])
            relative_index+=1
        else
            push!(linear_regions[linear_map]["polyhedra"],poly)
        end
    end
    return linear_regions
end

####################### Representation Functions ##################################

@doc""""
    bound_reps(reps,bounding_box)

Takes as input a set of matrix representations for polyhedra, and bounds them in a box determined by `bounding_box`.
"""
function bound_reps(reps,bounding_box)
    n=length(collect(keys(bounding_box)))
    bounded_reps=Dict("m_reps" => [], "f_indices" => [])
    for (m_rep,f_idx) in zip(reps["m_reps"],reps["f_indices"])
        A,b=m_rep[1],m_rep[2]
        for k in 1:n
            # adding the upper bound
            A=vcat(A,[j==k ? 1 : 0 for j in 1:n]')
            push!(b,bounding_box[k][2])
            # adding the lower bound
            A=vcat(A,[j==k ? -1 : 0 for j in 1:n]')
            push!(b,-bounding_box[k][1])
        end
        if Oscar.is_feasible(Oscar.polyhedron(A,b))
            push!(bounded_reps["m_reps"],[A,b])
            push!(bounded_reps["f_indices"],f_idx)
        end
    end
    return bounded_reps
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
        A=Float64.(m_rep[1])*inv_rot_matrix
        b=Float64.(m_rep[2])
        # equivalent to setting the higher dimensions to zero, so that we obtain
        # the polyhedron obtained on the two-dimensional plane of the first two dimensions.
        p_oscar=Oscar.polyhedron(A[:,1:2],b)
        if Oscar.is_feasible(p_oscar)
            push!(projected_reps["m_reps"],[A[:,1:2],b])
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

@doc"""
    m_reps(f::TropicalPuiseuxPoly)

Computes the matrix representation of the polyhedron corresponding to each monomial in a tropical polynomial.
"""
function m_reps(f::TropicalPuiseuxPoly)
    reps=Dict("m_reps" => [], "f_indices" => [])
    for i in eachindex(f)
        A=mapreduce(permutedims,vcat,[f.exp[j]-f.exp[i] for j in eachindex(f)])
        b=[Rational(f.coeff[f.exp[i]])-Rational(f.coeff[j]) for j in f.exp]

        p_oscar=Oscar.polyhedron(A,b)
        # Only full dimensional polyhedra will be relevant to the plots
        if Oscar.is_fulldimensional(p_oscar)
            push!(reps["m_reps"],[A,b])
            push!(reps["f_indices"],i)
        end
    end
    return reps
end

@doc"""
    m_reps(f::TropicalPuiseuxPoly)

Computes the matrix representation of the polyhedron corresponding to each monomial in a tropical rational map.
"""
function m_reps(f::TropicalPuiseuxRational)
    numerator,denominator=f.num,f.den
    n_reps=m_reps(numerator)
    d_reps=m_reps(denominator)

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
    return reps
end

@doc""""
    format_reps(reps,bounding_box=nothing,rot_matrix=nothing)

Returned are the matrix representations, and the corresponding index of the monomial, of the polyhedron formatted according to a bounding box and a rotation matrix. If bounding box is not provided, then the polyhedra are bounded by a region that encompases all the intersections between the polyhedra. If `f` has more than two variables, then a rotation matrix should be supplied so that the returned representations are two-dimensional.
"""
function formatted_reps(f;bounding_box=nothing,rot_matrix=nothing)
    reps=m_reps(f)
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
    polyhedra_from_reps(reps,oscar::Bool=false)

Given a set of matrix representations of polyhedra, returned by default are the `Polyhedra.jl` polyhedron representations, otherwise the `Oscar.jl` polyhedron representations can be returned.
"""
function polyhedra_from_reps(reps,oscar::Bool=false)
    if oscar
        return [Oscar.polyhedron(m_rep[1],m_rep[2]) for m_rep in reps["m_reps"]]
    else
        return [Polyhedra.polyhedron(Polyhedra.hrep(m_rep[1],m_rep[2]),CDDLib.Library(:exact)) for m_rep in reps["m_reps"]]
    end
end

####################### Plotting Functions ##################################

@doc raw"""
    plotpoly(polyhedron,color,ax)

Plots a polyhedorn in a certain color on a given GLMakie axis.
"""
function plotpoly(poly,color,ax)
    if Polyhedra.fulldim(poly)==1
        # one-dimensional polyhedron are just lines
        vertices=[point[1] for point in collect(Polyhedra.points(poly))]
        GLMakie.lines!(ax,vertices,zeros(length(vertices)),color=color,linewidth=2)
    else
        # higher (two) dimensional polyhedra
        m=Polyhedra.Mesh(poly)
        try
            GLMakie.mesh!(ax,m,color=color,alpha=0.5)
        catch
        end
    end
end

@doc raw"""
    plotsurface(poly,linear_map,color,ax)

Given a polyhedron and its corresponding linear map, this function plots that the action of the linear map given the polyhedron as its input.
"""
function plotsurface(poly,linear_map,color,ax)
    input_vertices,output_vertices=get_surface_points(poly,linear_map)
    if length(input_vertices)==1
        GLMakie.lines!(ax,input_vertices[1],output_vertices,color=color,linewidth=2)
    else
        xs,ys=input_vertices[1],input_vertices[2]
        # the polyhedron is convex so we can plot the triangles formed by its vertices
        for triangle in Combinatorics.combinations(range(1,length(xs)),3)
            GLMakie.mesh!(ax,xs[triangle],ys[triangle],output_vertices[triangle],color=(color,0.9),overdraw=true,shading=NoShading)
        end
    end
end

@doc raw"""
    plotlevelset(f,poly,linear_map,level_set_value,ax)

Given a polyhedron, its corresponding linear map, and a the level set of focus, this function plots the region on the polyhedron where the level set is attained. 
"""
function plotlevelset(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational},poly,linear_map,level_set_value,ax)
    component=get_level_set_component(poly,linear_map,level_set_value)
    # if component is empty then the level set does not cross f in the region poly
    if length(component)>0
        if nvars(f)==1
            GLMakie.scatter!(ax,[point[1] for point in component],zeros(length(component)),color=:red)
        else
            GLMakie.lines!(ax,[point[1] for point in component],[point[2] for point in component],color=:red,linewidth=2)
        end
    end
end

@doc raw"""
    plot_linear_regions(f;bounding_box=nothing,rot_matrix=nothing,level_set_value=nothing)

Plots the linear regions of a tropical polynomial or tropical rational map, with each region color coordinated with respect to the linear map that operates on that region. If bounding box is not provided, then the regions are bounded by a region that encompases all the intersections between the polyhedra. A rotation matrix should be provided when `f` has more than two variables, such that the regions intersecting the rotated plane of the first two variables are plotted. If a level set value is provided, then the level set of f is depicted on the plot in red.
"""
function plot_linear_regions(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational};bounding_box=nothing,rot_matrix=nothing,level_set_value=nothing)
    if nvars(f)>2 && rot_matrix==nothing
        error("Please supply a rotation matrix, even if it is the identity, when there are more than two input dimensions.")
    end
    fig=GLMakie.Figure()
    ax=GLMakie.Axis(fig[1,1])

    reps=formatted_reps(f,bounding_box=bounding_box,rot_matrix=rot_matrix)
    polys=polyhedra_from_reps(reps)
    linear_maps=get_linear_maps(f,reps["f_indices"])
    linear_regions=get_linear_regions(polys,linear_maps)

    # plot the polyhedra for each linear region
    for key in collect(keys(linear_regions))
        region_color=linear_regions[key]["color"]
        region_polyhedra=linear_regions[key]["polyhedra"]
        for poly in region_polyhedra
            plotpoly(poly,region_color,ax)
        end
    end

    # if a level set value is specified we add the level set to the plot
    if level_set_value!=nothing
        for (linear_map,value) in linear_regions
            for poly in value["polyhedra"]
                plotlevelset(f,poly,linear_map,level_set_value,ax)
            end
        end
    end
    return fig,ax
end

@doc raw"""
    plot_linear_maps(f;bounding_box=nothing,xreversed=false,yreversed=false)

Plots the functional value of a tropical polynomial or tropical rational map, with the regions colored with respect to the polyhedra. If bounding box is not provided, then the input domain of the plot is bounded by a region that encompases all the intersections between the polyhedra. When `f` has two variables, xreversed and yreversed control the perspective of the resulting three-dimensional plot. This function is only support for tropical polynomials and tropical rational maps in at most two variables.
"""
function plot_linear_maps(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational};bounding_box=nothing,xreversed=false,yreversed=false)
    if nvars(f)>2
        error("Only supported for rational maps with at most two input dimensions")
    end

    reps=formatted_reps(f,bounding_box=bounding_box)
    polys=polyhedra_from_reps(reps)
    linear_maps=get_linear_maps(f,reps["f_indices"])
    linear_regions=get_linear_regions(polys,linear_maps)

    fig=GLMakie.Figure()
    if nvars(f)==1
        ax=GLMakie.Axis(fig[1,1])
    else
        ax=GLMakie.Axis3(fig[1,1],xreversed=xreversed,yreversed=yreversed)
    end

    # plot the polyhedra for each linear region
    for (linear_map,value) in linear_regions
        region_color=value["color"]
        for poly in value["polyhedra"]
            plotsurface(poly,linear_map,region_color,ax)
        end
    end
    return fig,ax
end