using Pkg
Pkg.activate(".")


using AIBECS
using LinearSolve
using AlgebraicMultigrid
using IncompleteLU
using IterativeSolvers
using HYPRE
using LinearAlgebra
using Unitful
using Unitful: m, s, yr
using Plots


# Define the model
grd, T = OCCA.load()
# grd, T = Primeau_2x2x2.load()

issrf = let
    issrf3D = zeros(size(grd.wet3D))
    issrf3D[:,:,1] .= 1
    issrf3D[grd.wet3D]
end

A = T + sparse(Diagonal(issrf))

b = ones(size(A, 1))

# Solve the system using the LinearSolve.jl package
prob = LinearProblem(A, b)

sol0 = A \ b
plt0 = plothorizontalslice(sol0 * s .|> yr, grd, depth=1000m, clim=(0, 2000))
@time sol0 = A \ b

# sol1 = solve(prob)
# plt1 = plothorizontalslice((sol1 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
# @time sol1 = solve(prob)

# sol2 = solve(prob, KLUFactorization())
# plt2 = plothorizontalslice((sol2 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
# @time sol2 = solve(prob, KLUFactorization())
# sol3 = solve(prob, UMFPACKFactorization())
# plt3 = plothorizontalslice((sol3 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
# @time sol3 = solve(prob, UMFPACKFactorization())

# sol4 = solve(prob, SparspakFactorization())
# plt4 = plothorizontalslice((sol4 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
# @time sol4 = solve(prob, SparspakFactorization())

# # GMRES no preconditioner
# # sol5 = solve(prob, KrylovJL_GMRES())
# # plt5 = plothorizontalslice((sol5 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
# # @time sol5 = solve(prob)


# AlgebraicMultigrid: Implementations of the algebraic multigrid method. Must be converted to a preconditioner via AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.precmethod(A)). Requires A as a AbstractMatrix. Provides the following methods:
# Pl = AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.ruge_stuben(A; max_levels=3, coarse_solver=AlgebraicMultigrid.LinearSolveWrapper(UMFPACKFactorization())))
# sol6 = solve(prob, KrylovJL_GMRES(); Pl)
# plt6 = plothorizontalslice((sol6 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
# @time sol6 = solve(prob, KrylovJL_GMRES(); Pl)

# Pl = AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.smoothed_aggregation(A; max_levels=3, coarse_solver=AlgebraicMultigrid.LinearSolveWrapper(UMFPACKFactorization())))
# sol7 = solve(prob, IterativeSolversJL_CG(); Pl)
# plt7 = plothorizontalslice((sol7 - sol0) * s .|> yr, grd, depth=1000m, clim=(-400, 400), cmap=:balance)
# plt7 = plothorizontalslice(sol7 * s .|> yr, grd, depth=1000m, clim=(0, 2000))
# @time sol7 = solve(prob, IterativeSolversJL_CG(); Pl)



# IncompleteLU.ilu: an implementation of the incomplete LU-factorization preconditioner. This requires A as a SparseMatrixCSC.
# @time Pl = IncompleteLU.ilu(A, τ=1e-8) # arbitriry largest τ for which solver worked fast
# sol8 = solve(prob, KrylovJL_GMRES(); Pl)
# plt8 = plothorizontalslice((sol8 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
# @time sol8 = solve(prob, KrylovJL_GMRES(); Pl)



# BoomerAMG preconditioner
# Pl = HYPRE.BoomerAMG
# sol = solve(prob, HYPREAlgorithm(HYPRE.PCG); Pl)



a


# The following preconditioners match the interface of LinearSolve.jl.

#     LinearSolve.ComposePreconditioner(prec1,prec2): composes the preconditioners to apply prec1 before prec2.

#     LinearSolve.InvPreconditioner(prec): inverts mul! and ldiv! in a preconditioner definition as a lazy inverse.

#     LinearAlgera.Diagonal(s::Union{Number,AbstractVector}): the lazy Diagonal matrix type of Base.LinearAlgebra. Used for efficient construction of a diagonal preconditioner.

#     Other Base.LinearAlgera types: all define the full Preconditioner interface.


#     Preconditioners.CholeskyPreconditioner(A, i): An incomplete Cholesky preconditioner with cut-off level i. Requires A as a AbstractMatrix and positive semi-definite.


#     PyAMG: Implementations of the algebraic multigrid method. Must be converted to a preconditioner via PyAMG.aspreconditioner(PyAMG.precmethod(A)). Requires A as a AbstractMatrix. Provides the following methods:
#         PyAMG.RugeStubenSolver(A)
#         PyAMG.SmoothedAggregationSolver(A)

#     ILUZero.ILU0Precon(A::SparseMatrixCSC{T,N}, b_type = T): An incomplete LU implementation. Requires A as a SparseMatrixCSC.

#     LimitedLDLFactorizations.lldl: A limited-memory LDLᵀ factorization for symmetric matrices. Requires A as a SparseMatrixCSC. Applying F = lldl(A); F.D .= abs.(F.D) before usage as a preconditioner makes the preconditioner symmetric positive definite and thus is required for Krylov methods which are specialized for symmetric linear systems.

#     RandomizedPreconditioners.NystromPreconditioner A randomized sketching method for positive semidefinite matrices A. Builds a preconditioner P≈A+μ∗IP≈A+μ∗I for the system (A+μ∗I)x=b(A+μ∗I)x=b.

#     HYPRE.jl A set of solvers with preconditioners which supports distributed computing via MPI. These can be written using the LinearSolve.jl interface choosing algorithms like HYPRE.ILU and HYPRE.BoomerAMG.

#     KrylovPreconditioners.jl: Provides GPU-ready preconditioners via KernelAbstractions.jl. At the time of writing the package provides the following methods:
#         Incomplete Cholesky decomposition KrylovPreconditioners.kp_ic0(A)
#         Incomplete LU decomposition KrylovPreconditioners.kp_ilu0(A)
#         Block Jacobi KrylovPreconditioners.BlockJacobiPreconditioner(A, nblocks, device)