# qsub -I -P y99 -q normal -l mem=38GB -l storage=scratch/gh0+scratch/y99+scratch/p66 -l walltime=02:00:00 -l ncpus=10
nprocs = 10

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using MKL
using LinearAlgebra
@show BLAS.get_config().loaded_libs
import Pardiso # TODO add my fork of Pardiso here # import Pardiso instead of using (to avoid name clash?)
@show BLAS.get_config().loaded_libs
using LinearSolve
using AIBECS
using Unitful
using Unitful: m, s, yr, Myr


@show BLAS.get_config().loaded_libs
# @show Pardiso.blaslib


# grd, T = AIBECS.JLD2.load("/Users/z3319805/Downloads/OCCA.jld2", "grid", "T")
# T = ustrip.(s^-1, T)
grd, T = OCCA.load()


@info "Grid size: $(size(grd.wet3D))"
@info "Number of wet cells: $(count(grd.wet3D))"
@info "Number of nonzeros in T: $(nnz(T))"

@info "Setting up the linear problem..."

issrf = let
    issrf3D = zeros(size(grd.wet3D))
    issrf3D[:, :, 1] .= 1
    issrf3D[grd.wet3D]
end

A = T + sparse(Diagonal(issrf))

b = ones(size(A, 1))

# Solve the system using the LinearSolve.jl package
prob = LinearProblem(A, b, u0 = ustrip(s, 1000yr) * ones(size(b)))

sol0 = A \ b
@time "backslash" sol0 = A \ b

matrix_type = Pardiso.REAL_SYM
solver = MKLPardisoIterate(; nprocs, matrix_type)
prob6 = init(prob, solver, reltol = 1.0e-10)
@time "Pardiso + MKL.jl" sol6 = solve!(prob6).u
