# In this file we compute the Hoffman constant of matrices, tropical polynomials and
# tropical rational maps. Moreover, we obtain lower and upper bounds on these values
# and compare the time taken to get these values

using TropicalNN
using Random

# Computing the Hoffman constant of a matrix, a tropical polynomial 
# or a tropical rational map works all in the same way

matrix=rand(7,7)
pmap=random_pmap(2,7)
w,b,t=random_mlp([2,4,1])
rmap=mlp_to_trop_with_quicksum_with_strong_elim(w,b,t)[1]

hoff_const_matrix=round(exact_hoff(matrix),digits=4)
hoff_const_pmap=round(exact_hoff(pmap),digits=4)
hoff_const_rmap=round(exact_hoff(rmap),digits=4)
println("Hoffman Constant")
println("Matrix $hoff_const_matrix")
println("Tropical Polynomial $hoff_const_pmap")
println("Tropical Rational Map $hoff_const_rmap\n")

# Similarly one can upper bounds as follows

hoff_upper_matrix=round(upper_hoff(matrix),digits=4)
hoff_upper_pmap=round(upper_hoff(pmap),digits=4)
hoff_upper_rmap=round(upper_hoff(rmap),digits=4)
println("Hoffman Constant Upper Bound")
println("Matrix $hoff_upper_matrix (>= $hoff_const_matrix)")
println("Tropical Polynomial $hoff_upper_pmap (>= $hoff_const_pmap)")
println("Tropical Rational Map $hoff_upper_rmap (>= $hoff_const_rmap)\n")

# Computing a lower bound works through random sampling, and 
# so choosing more samples should lead to a tighter bound, 
# by default the number of samples is 10

hoff_lower_matrix=round(lower_hoff(matrix,2),digits=4)
hoff_lower_pmap=round(lower_hoff(pmap,2),digits=4)
hoff_lower_rmap=round(lower_hoff(rmap,2),digits=4)
println("Hoffman Constant Lower Bound (2 Samples)")
println("Matrix $hoff_lower_matrix (<= $hoff_const_matrix)")
println("Tropical Polynomial $hoff_lower_pmap (<= $hoff_const_pmap)")
println("Tropical Rational Map $hoff_lower_rmap (<= $hoff_const_rmap)\n")

hoff_lower_matrix=round(lower_hoff(matrix,20),digits=4)
hoff_lower_pmap=round(lower_hoff(pmap,20),digits=4)
hoff_lower_rmap=round(lower_hoff(rmap,20),digits=4)
println("Hoffman Constant Lower Bound (20 Samples)")
println("Matrix $hoff_lower_matrix (<= $hoff_const_matrix)")
println("Tropical Polynomial $hoff_lower_pmap (<= $hoff_const_pmap)")
println("Tropical Rational Map $hoff_lower_rmap (<= $hoff_const_rmap)\n")

# If the number of samples we choose exceeds the number of distinct
# samples possible, then we default to the exact constant computation
# (not you may not get exact values for the tropical polynomial or tropical rational map
# since it is dependent on how many linear regions they have)

hoff_lower_matrix=round(lower_hoff(matrix,200),digits=4)
hoff_lower_pmap=round(lower_hoff(pmap,200),digits=4)
hoff_lower_rmap=round(lower_hoff(rmap,200),digits=4)
println("Hoffman Constant Lower Bound (200 Samples or Exact)")
println("Matrix $hoff_lower_matrix (<= $hoff_const_matrix)")
println("Tropical Polynomial $hoff_lower_pmap (<= $hoff_const_pmap)")
println("Tropical Rational Map $hoff_lower_rmap (<= $hoff_const_rmap)\n")

# The method of exact computation works by brute force. The approximations
# mitigate some of this computational burden and so should be faster to compute.

lower_times=[]
exact_times=[]
upper_times=[]
for _ in 1:50
    local matrix=rand(5,5)

    t_start=time()
    lower=lower_hoff(matrix)
    t_elapsed=time()-t_start
    push!(lower_times,t_elapsed)

    t_start=time()
    exact=exact_hoff(matrix)
    t_elapsed=time()-t_start
    push!(exact_times,t_elapsed)

    t_start=time()
    upper=upper_hoff(matrix)
    t_elapsed=time()-t_start
    push!(upper_times,t_elapsed)
end
avg_lower_time=round(sum(lower_times)/length(lower_times),digits=6)
avg_exact_time=round(sum(exact_times)/length(exact_times),digits=6)
avg_upper_time=round(sum(upper_times)/length(upper_times),digits=6)

println("Avg time for lower bound $avg_lower_time\nAvg time for exact constant $avg_exact_time\nAvg time for upper bound $avg_upper_time\n")

# The lower and upper bound approximations for tropical polynomials and tropical rational 
# maps are not as tight, as they are are respectively taken to be the minimum or maximum
# of bounds on other matrices.

matrix_lower_tightness=[]
matrix_upper_tightness=[]
pmap_lower_tightness=[]
pmap_upper_tightness=[]
rmap_lower_tightness=[]
rmap_upper_tightness=[]
for _ in 1:5
    local matrix=rand(5,5)
    lower=lower_hoff(matrix)
    exact=exact_hoff(matrix)
    upper=upper_hoff(matrix)
    push!(matrix_lower_tightness,abs(exact-lower)/exact)
    push!(matrix_upper_tightness,abs(exact-upper)/exact)

    local pmap=random_pmap(2,5)
    lower=lower_hoff(pmap)
    exact=exact_hoff(pmap)
    upper=upper_hoff(pmap)
    push!(pmap_lower_tightness,abs(exact-lower)/exact)
    push!(pmap_upper_tightness,abs(exact-upper)/exact)

    local w,b,t=random_mlp([2,3,1])
    local rmap=mlp_to_trop_with_quicksum_with_strong_elim(w,b,t)[1]
    lower=lower_hoff(rmap)
    exact=exact_hoff(rmap)
    upper=upper_hoff(rmap)
    push!(rmap_lower_tightness,abs(exact-lower)/exact)
    push!(rmap_upper_tightness,abs(exact-upper)/exact)
end

avg_matrix_lt=round(sum(matrix_lower_tightness)/length(matrix_lower_tightness),digits=4)
avg_matrix_ut=round(sum(matrix_upper_tightness)/length(matrix_upper_tightness),digits=4)
avg_pmap_lt=round(sum(pmap_lower_tightness)/length(pmap_lower_tightness),digits=4)
avg_pmap_ut=round(sum(pmap_upper_tightness)/length(pmap_upper_tightness),digits=4)
avg_rmap_lt=round(sum(rmap_lower_tightness)/length(rmap_lower_tightness),digits=4)
avg_rmap_ut=round(sum(rmap_upper_tightness)/length(rmap_upper_tightness),digits=4)
println("Matrix\n    Lower bound tightness $avg_matrix_lt\n     Upper bound tightness $avg_matrix_ut")
println("Tropical Polynomial\n    Lower bound tightness $avg_pmap_lt\n     Upper bound tightness $avg_pmap_ut")
println("Tropical Rational Map\n    Lower bound tightness $avg_rmap_lt\n     Upper bound tightness $avg_rmap_ut")