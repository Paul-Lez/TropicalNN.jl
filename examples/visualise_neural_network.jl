# In this file we visualise the linear regions of a neural network

using TropicalNN, GLMakie

# A neural network is just a tropical rational map. 
# For simplicitly we will consider a neural network with architecture [2,4,1]

# Initialise the weight, biases of a ReLU neural network randomly, that is
# the thresholds will all be set to zero
weights,biases,thresholds=random_mlp([2,4,1])

# Obtain the tropical rational map corresponding to this neural network
f=mlp_to_trop_with_quicksum_with_strong_elim(weights,biases,thresholds)[1]

# Plot the linear regions
fig,ax=plot_linear_regions(f)
GLMakie.save("./examples/outputs/neural_network_linear_regions.png",fig)

# Plot the linear maps
fig,ax=plot_linear_maps(f)
GLMakie.save("./examples/outputs/neural_network_linear_maps.png",fig)