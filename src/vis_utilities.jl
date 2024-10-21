function random_pmap(n_vars,n_mons)
    return TropicalPuiseuxPoly(Rational{BigInt}.(rand(Normal(0,1/sqrt(2)),n_mons)),[Rational{BigInt}.(rand(Normal(0,1/sqrt(2)),n_vars)) for _ in 1:n_mons],true)
end

function update_bounding_box(bounding_box,p)
    vertices=collect(Polyhedra.points(p))
    for vertex in vertices
        for k in collect(keys(bounding_box))
            bounding_box[k][1]=min(bounding_box[k][1],vertex[k])
            bounding_box[k][2]=max(bounding_box[k][2],vertex[k])
        end
    end
    return bounding_box
end

function get_full_bounding_box(f,reps)
    n=nvars(f)
    bounding_box=Dict(dimension => [0.0,0.0] for dimension in 1:n)
    for m_rep in reps["m_reps"]
        p_oscar=Oscar.polyhedron(m_rep[1],m_rep[2])
        if Oscar.is_fulldimensional(p_oscar)
            p=Polyhedra.polyhedron(Polyhedra.hrep(m_rep[1],m_rep[2]),CDDLib.Library())
            bounding_box=update_bounding_box(bounding_box,p)
        end
    end
    for k in collect(keys(bounding_box))
        bounding_box[k][1]-=1
        bounding_box[k][2]+=1
    end
    return bounding_box
end

function apply_linear_map(point,linear_map)
    return sum([linear_map[2][j]*point[j] for j in 1:length(point)])+linear_map[1]
end

function get_level_set(value,polyhedra,linear_maps)
    level_set_components=[]
    for (poly,linear_map) in zip(polyhedra,linear_maps)
        vertices=collect(Polyhedra.points(poly))
        functional_values=[apply_linear_map(vertex,linear_map) for vertex in vertices]
        poly_component=[]
        for pair in Combinatorics.combinations(range(1,length(functional_values)),2)
            (value_one,value_two)=(functional_values[pair[1]],functional_values[pair[2]])
            (vertex_one,vertex_two)=(vertices[pair[1]],vertices[pair[2]])
            if (value_one-value)*(value_two-value)<0
                push!(poly_component,vertex_one+(vertex_two-vertex_one)*(value-value_one)/(value_two-value_one))
            elseif (value_one-value)*(value_two-value)==0
                push!(poly_component,vertex_one)
            end
        end
        if length(poly_component)>0
            push!(poly_component,poly_component[1])
            push!(level_set_components,poly_component)
        end
    end
    return level_set_components
end

function get_surface_points(poly,linear_map)
    poly_points=collect(Polyhedra.points(poly))
    input_dim=length(poly_points[1])
    input_vertices=[[poly_point[k] for poly_point in poly_points] for k in 1:input_dim]
    output_vertices=[apply_linear_map(poly_point,linear_map) for poly_point in poly_points]
    return input_vertices,output_vertices
end

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

function get_linear_regions(linear_maps,polyhedra)
    num_distinct_linear_maps=length(Set(linear_maps))
    colors=[ColorSchemes.get(ColorSchemes.Paired_8,rand()) for _ in 1:num_distinct_linear_maps]
    linear_regions=Dict()
    relative_index=1
    for (linear_map,poly) in zip(linear_maps,polyhedra)
        if !haskey(linear_regions,linear_map)
            linear_regions[linear_map]=Dict("color" => colors[relative_index], "polyhedra" => [poly])
            relative_index+=1
        else
            push!(linear_regions[linear_map]["polyhedra"],poly)
        end
    end
    return linear_regions
end