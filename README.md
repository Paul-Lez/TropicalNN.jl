# TropicalNN

[![Build Status](https://github.com/Paul-Lez/TropicalNN.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Paul-Lez/TropicalNN.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Content 
This library implements some basic symbolic tropical geometric tools for computing with neural networks in Julia. 

### Tropical Expressions of Neural Networks
Every fully connected multilayer perceptron (MLP) with ReLU activation can be written in terms of tropical algebra [1]. More precisely, any such MLP can be expressed as a *tropical rational function*, if one allows the exponents appearing in such expressions to be non-integral (e.g. rational or real numbers). We provide tools for symbolically manipulating such objects, and for finding the tropical expression of arbitrary MLPs with ReLU activation. 

### Tropical Measures of Expressivity of Neural Networks
There are various ways of measuring how *complicated* the function represented by a given neural network is. A large body of litterature has formed around using the number of *linear regions* of a neural network as such a measure. This library provides tools for computing this quantity, analysing the geometry of these linear regions, and studying another algebraic measure of expressivity, namely the number of irredundant monomials that appear in the tropical expression of a neural network. 

## Examples of usage: 

### Manipulating Tropical Expressions
```
# Write down the tropical polynomial f = 1*X^1*Y^0 + 2*X^0*Y^1 + 3*X^1*Y^1
f = TropicalPuiseuxPoly([1, 2, 3], [[1, 0], [0, 1], [1, 1]], false)
# Write down the tropical polynomial 0*X^1*Y^7 + 4*X^0*Y^1 + (-5)*X^9*Y^1
g = TropicalPuiseuxPoly([0, 4, -5], [[1, 7], [0, 1], [9, 1]], false) 
@show f + g # outputs the sum of f and g 
@show f * g # outputs the product of f and g
```

### Computing Tropical Expressions of MLPs
```
# Generate a random neural network, expressed as a tuple (W, b, t) where W is an array of weight 
# matrices, b an array of biases and t an array of activation threshholds
W, b, t = random_mlp([3, 2, 2])
# Compute a tropical expression of this network
@show trop1 = mlp_to_trop(W, b, t)
# Compute a reduced tropical expression of this network 
@show trop2 = mlp_to_trop_with_quicksum_with_monomial_strong_elim(W, b, t)
```

### Computing Linear Regions of Tropical Rational Functions
```
f = TropicalPuiseuxPoly([1, 2, 3], [[1, 0], [0, 1], [1, 1]], false)
g = TropicalPuiseuxPoly([0, 4, -5], [[1, 7], [0, 1], [9, 1]], false) 
# Compute the linear regions of f/g, expressed as an array containing polyhedra or arrays of polyhedra
linear_regions = enum_linear_regions_rat(f, g)
@show lenth(linear_regions)
```

If this code was useful, please consider citing [this paper](https://arxiv.org/abs/2405.20174), using
```
@article{lezeau2024tropical,
  title={Tropical Expressivity of Neural Networks},
  author={Lezeau, Paul and Walker, Thomas and Cao, Yueqi and Bhatia, Shiv and Monod, Anthea},
  journal={arXiv preprint arXiv:2405.20174},
  year={2024}
}
```

## References
[1] [*Tropical Geometry of Deep Neural Networks*](https://arxiv.org/pdf/1805.07091), Liwen Zhang, Gregory Naitzat and Lek-Heng Lim.