# In this file we visualise the effective radius of a tropical polynomial.
# The effective radius of a tropical polynomial is determined from its Hoffman constant.
# It represents a radius, under the infinity norm, in which all the linear regions enter. 

using TropicalNN, GLMakie

# Generate a random tropical polynomial in two variables with four monomials
pmap=random_pmap(2,4)

# Compute the effective radius of the tropical polynomial
er=exact_er(pmap)
println(er)

# Create a two-dimensional bounding box slighly larger than the effective radius
bounding_box=Dict(1 => [-er-0.5,er+0.5], 2 => [-er-0.5,er+0.5])

# Obtain the GLMakie figure and axis containing the linear regions of the
# tropical polynomial bounded using the bounding box
fig,ax=plot_linear_regions(pmap,bounding_box=bounding_box)

# Identify the region bounded by the effective radius on the plot
GLMakie.lines!(ax,[er,er,-er,-er,er],[er,-er,-er,er,er],color=:blue)

# save the plot
GLMakie.save("./examples/outputs/effective_radius.png",fig)