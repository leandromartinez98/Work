
using Statistics: mean
using LinearAlgebra: norm_sqr
using DelimitedFiles
include("./align.jl")

x_read = readdlm("./1.xyz")
y_read = readdlm("./2.xyz")
x = [ SVector{3,Float64}(row) for row in eachrow(x_read) ]
y = [ SVector{3,Float64}(row) for row in eachrow(y_read) ]
xmass = [ 1. for _ in 1:length(x) ]
ymass = [ 1. for _ in 1:length(y) ]

# Align x to y (same size, correspondence given by sequence)
z = align(x, y, xmass, ymass)

# or
# xnew = similar(x)
# align!(xnew, x, y, xmass, ymass)

# Compute rmsd before and after alignment
println("rmsd before = ", sqrt(mean(norm_sqr, x - y))) 
println("rmsd after = ", sqrt(mean(norm_sqr, z - y))) 
