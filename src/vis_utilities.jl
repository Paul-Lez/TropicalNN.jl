# This file provides some utility functions
# for the visualisation functions.

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
function get_full_bounding_box(f,reps)
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
        elseif (values[1]-value)*(values[2]-value)==0
            # in this case the entire edge is on the level set so we add the vertex such that
            # the edges are connected in the plot
            push!(component,vertexs[1])
        end
    end
    if length(component)>0
        # connecting the components, such that if an edge lies on the level then the
        # edge is highlighted in the plot
        push!(component,component[1])
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
function get_linear_maps(f,f_indices)
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
    get_linear_regions(polyhedra,linear_maps)

For each unique linear map, the polyhedra that are acted on by this linear map are identified. Moreover, a color is attributed to each unique linear map such that they can be distinguished in the visualisations.
"""
function get_linear_regions(polyhedra,linear_maps)
    num_distinct_linear_maps=length(Set(linear_maps))
    # color coding each distinct linear map
    colors=[ColorSchemes.get(ColorSchemes.Paired_8,rand()) for _ in 1:num_distinct_linear_maps]
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