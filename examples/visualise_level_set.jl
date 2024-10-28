# In this file we visualise the level of a tropical polynomial

using TropicalNN
using GLMakie

# For a tropical polynomial with a scalar output, we can identify
# its level set, that is the set of points whose outputs are a certain value.
# We can do this for tropical polynomials (and tropical rational maps) in arbitrary dimensions,
# however, for simplicitly we will consider tropical polynomials in two variables.

# Initialise a random tropical polynomial in two variables and four monomials
f=TropicalPuiseuxPoly(Rational{BigInt}.([-3723308421248693//288230376151711744,73114227503219//140737488355328,-747885935752887//1125899906842624,4648013016134511//4503599627370496]),[Rational{BigInt}.([3378591440196385//4503599627370496, 2316848943565215//4503599627370496]),Rational{BigInt}.([-1429525263752149//2251799813685248, -896638549896579//9007199254740992]),Rational{BigInt}.([-2508142725340491//4503599627370496, -4491348586605881//9007199254740992]),Rational{BigInt}.([1437914608837039//18014398509481984, 8253797924747951//9007199254740992])],false)

# Plot the linear regions with the level set of 0.5 identified in red
fig,ax=plot_linear_regions(f,level_set_value=0.5)
GLMakie.save("./examples/outputs/tropical_polynomial_nonzero_level_set.png",fig)

# We identify that an entire polyhedron lies in the level set by
# highlighting each of its edges in red
f=TropicalPuiseuxPoly(Rational{BigInt}.([0,73114227503219//140737488355328,-747885935752887//1125899906842624,4648013016134511//4503599627370496]),[Rational{BigInt}.([0,0]),Rational{BigInt}.([-1429525263752149//2251799813685248,-896638549896579//9007199254740992]),Rational{BigInt}.([-2508142725340491//4503599627370496,-4491348586605881//9007199254740992]),Rational{BigInt}.([1437914608837039//18014398509481984,8253797924747951//9007199254740992])],false)
fig,ax=plot_linear_regions(f,level_set_value=0.0)
GLMakie.save("./examples/outputs/tropical_polynomial_zero_level_set.png",fig)