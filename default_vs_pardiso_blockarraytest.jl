using Pkg
Pkg.activate(".")
Pkg.instantiate()

using AIBECS
using LinearSolve
import Pardiso # that's what they do in LinearSolve.jl benchmarks (maybe to avoid name clash)
using SparseArrays
using LinearAlgebra
using Unitful
using Unitful: m, s, yr, d
using BlockArrays

# Define the model
# grd, T = OCCA.load()
# grd, T = Primeau_2x2x2.load()
grd, T = OCIM2_48L.load()
Nwet = size(T, 1)

v = grd.volume_3D[grd.wet3D];

issrf = let
    issrf3D = zeros(size(grd.wet3D))
    issrf3D[:,:,1] .= 1
    issrf3D[grd.wet3D]
end
M = sparse(Diagonal(issrf))

# Now let's imagine we have 4 seasons of OCCA where T gets more or less intense
αs = (1.0, 0.95, 1.0, 1.05)
Nα = length(αs)
δt = ustrip(s, 365d / Nα)
A = BlockArray(spzeros(Nα * Nwet, Nα * Nwet), fill(Nwet, Nα), fill(Nwet, Nα))
@time "building blocks" for (i, α) in enumerate(αs)
    A[Block(i, i)] = I + δt * (α * T + M)
    A[Block(mod1(i+1, Nα), i)] = -I(Nwet)
end
# @time "converting BlockArray to standard sparse" A = sparse(A)
A = @time "converting BlockArray to standard sparse" reduce(hcat, reduce(vcat, A[Block(i, j)] for i in 1:blocksize(A, 1)) for j in 1:blocksize(A, 2))

b = δt * ones(size(A, 1))

# Solve the system using the LinearSolve.jl package
prob = LinearProblem(A, b)

# extra test
nprocs = 48
@time "init" linsolve = init(prob, MKLPardisoFactorize(; nprocs))
@time "solve" solve!(linsolve)


fooo
