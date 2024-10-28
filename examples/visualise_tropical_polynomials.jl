# In this file we visualise a tropical polynomials in one, two and three variables.

using TropicalNN
using GLMakie

# One Variable
### Generate a tropical polynomial in one variable with four monomials
pmap=random_pmap(1,4)
## Linear Regions
### Visualise linear regions in the region [-1,1]
fig,ax=plot_linear_regions(pmap,bounding_box=Dict(1 => [-1,1]))
GLMakie.save("./examples/outputs/one_variable_tropical_polynomial_regions_bounded.png",fig)
### Visualise linear regions in a region encompasing all of the polynomial's linear regions
fig,ax=plot_linear_regions(pmap)
GLMakie.save("./examples/outputs/one_variable_tropical_polynomial_regions_full.png",fig)
## Linear Maps
### Visualise linear maps acting on the linear regions in the region [-1,1]
fig,ax=plot_linear_maps(pmap,bounding_box=Dict(1 => [-1,1]))
GLMakie.save("./examples/outputs/one_variable_tropical_polynomial_maps_bounded.png",fig)
### Visualise linear maps acting on the linear regions in a region encompasing all of the polynomial's linear regions
fig,ax=plot_linear_maps(pmap)
GLMakie.save("./examples/outputs/one_variable_tropical_polynomial_maps_full.png",fig)

# Two Variables
### Generate a tropical polynomial in two variables with four monomials
pmap=random_pmap(2,4)
## Linear Regions
### Visualise linear regions in the region [-1,1] for the first dimension and [0,1] for the second dimension
fig,ax=plot_linear_regions(pmap,bounding_box=Dict(1 => [-1,1], 2 => [0,1]))
GLMakie.save("./examples/outputs/two_variables_tropical_polynomial_regions_bounded.png",fig)
### Visualise linear regions in a region encompasing all of the polynomial's linear regions
fig,ax=plot_linear_regions(pmap)
GLMakie.save("./examples/outputs/two_variables_tropical_polynomial_regions_full.png",fig)
## Linear Maps
### Visualise linear maps acting on the linear regions in the region [-1,1] for the first dimension and [0,1] for the second dimension
fig,ax=plot_linear_maps(pmap,bounding_box=Dict(1 => [-1,1], 2 => [0,1]))
GLMakie.save("./examples/outputs/two_variables_tropical_polynomial_maps_bounded.png",fig)
### Visualise linear maps acting on the linear regions in a region encompasing all of the polynomial's linear regions
fig,ax=plot_linear_maps(pmap)
GLMakie.save("./examples/outputs/two_variables_tropical_polynomial_maps_full.png",fig)

# Three Variables
### As we are consider tropical polynomials in more than two dimensions we need to provide a rotation matrix, such that we can the consider two-dimensional polyhedra that intersected the plane constructed by the first two dimensions rotated by this matrix. For simplicitly we will choose the identity matrix.
### Generate a tropical polynomial in three variables with four monomials
pmap=random_pmap(3,4)
## Linear Regions
### Visualise linear regions in the region [-1,1]^3
fig,ax=plot_linear_regions(pmap,bounding_box=Dict(1 => [-1,1], 2 => [-1,1], 3 => [-1,1]),rot_matrix=[1 0 0;0 1 0;0 0 1])
GLMakie.save("./examples/outputs/three_variables_tropical_polynomial_regions_bounded.png",fig)
### Visualise linear regions in a region encompasing all of the polynomial's linear regions
fig,ax=plot_linear_regions(pmap,rot_matrix=[1 0 0;0 1 0;0 0 1])
GLMakie.save("./examples/outputs/three_variables_tropical_polynomial_regions_full.png",fig)
## Linear Maps
### The package currently does not support visualising the operation of the linear maps on the linear regions