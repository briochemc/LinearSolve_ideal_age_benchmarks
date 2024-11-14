using Pkg
Pkg.activate(".")
Pkg.instantiate()

using FileIO
using LinearSolve
import Pardiso # that's what they do in LinearSolve.jl benchmarks (maybe to avoid name clash)
using SparseArrays
using LinearAlgebra
using Unitful
using Unitful: m, s, yr

# Load matrix and grid metrics
model = "ACCESS-ESM1-5"
member = "r1i1p1f1"
experiment = "historical"
time_window = "Jan1990-Dec1999"
CMIP6outputdir = "/scratch/xv83/TMIP/data/$model/$experiment/$member/$(time_window)"
inputfile = joinpath(CMIP6outputdir, "transportmatrix_frommonthlymatrices.jld2")
@info "Loading matrices + metrics as $inputfile"
input = load(inputfile)
T = input["T"]
gridmetrics = input["gridmetrics"]
indices = input["indices"]
note = input["note"]
OceanTransportMatrixBuilder_version = input["OceanTransportMatrixBuilder"]
(; v3D) = gridmetrics
(; wet3D) = indices
v = v3D[wet3D]

issrf = let
    issrf3D = zeros(size(wet3D))
    issrf3D[:,:,1] .= 1
    issrf3D[wet3D]
end

A = T + sparse(Diagonal(issrf))

b = ones(size(A, 1))

# Solve the system using the LinearSolve.jl package
prob = LinearProblem(A, b)

# Solve twice to avoid precompilation
@info "Solving for ideal age once for precompilation"
solbackslash = A \ b
@info "Solving again"
@time "backslash" solbackslash = A \ b

println()

@info "Solving for ideal age once for precompilation"
nprocs = 48 # (max 48 for normal queue)
solMKLPardisoFactorize = solve(prob, MKLPardisoFactorize(; nprocs))
@info "Solving again"
@time "MKLPardisoFactorize nprocs=$nprocs" solMKLPardisoFactorize = solve(prob, MKLPardisoFactorize())
