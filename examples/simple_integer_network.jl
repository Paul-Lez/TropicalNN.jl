# Example: small neural network with integer weights
#
# Architecture [2, 2, 1]:
#   Layer 1: W = [0 1; 1 0], b = [1; 1], threshold = [0; 0]  (ReLU)
#   Layer 2: W = [1 -1],     b = [0],    threshold = [0]      (ReLU)

using TropicalNN

# Define weights, biases and ReLU thresholds
W1 = Rational{BigInt}[0 1; 1 0]
b1 = Rational{BigInt}[1, 1]
t1 = Rational{BigInt}[0, 0]

W2 = Rational{BigInt}[1 -1]
b2 = Rational{BigInt}[0]
t2 = Rational{BigInt}[0]

weights    = [W1, W2]
biases     = [b1, b2]
thresholds = [t1, t2]

# Convert to tropical representation
f = mlp_to_trop(weights, biases, thresholds)

println("Tropical rational function for the network output:")
println(f[1])
