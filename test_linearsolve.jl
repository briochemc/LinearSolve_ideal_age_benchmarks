using Pkg
Pkg.activate(".")

using AIBECS
using LinearSolve
using AlgebraicMultigrid
using IncompleteLU
using KrylovPreconditioners
using GeometricMultigrid
using IterativeSolvers
using HYPRE
using MPI
using LinearAlgebra
using Unitful
using Unitful: m, s, yr
using Plots
import Pardiso # import Pardiso instead of using (to avoid name clash?)

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

sol1 = solve(prob)
plt1 = plothorizontalslice((sol1 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
@time sol1 = solve(prob)

sol2 = solve(prob, KLUFactorization())
plt2 = plothorizontalslice((sol2 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
@time sol2 = solve(prob, KLUFactorization())

sol3 = solve(prob, UMFPACKFactorization())
plt3 = plothorizontalslice((sol3 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
@time sol3 = solve(prob, UMFPACKFactorization())

# sol4 = solve(prob, SparspakFactorization())
# plt4 = plothorizontalslice((sol4 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
# @time sol4 = solve(prob, SparspakFactorization())

# GMRES no preconditioner
if false # hangs
    sol5 = solve(prob, KrylovJL_GMRES())
    plt5 = plothorizontalslice((sol5 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
    @time sol5 = solve(prob)
end

# Pardiso solver
if false # cannot use MKL Pardiso on Mac
    matrix_type = Pardiso.REAL_SYM
    solver = MKLPardisoIterate(; nprocs=6, matrix_type)
    prob6 = init(prob, solver, rtol = 1e-10)
    @time sol6 = solve!(prob6).u
end

foo

if false # does not seem to work with OCCA matrix
    @time Pl = IncompleteLU.ilu(A, τ = 0.0) # even with "full" LU, which takes ages to factorize!?
    @time sol7 = solve(prob, KrylovJL_GMRES(; restart = true), Pl = Pl)
end

if false # KrylovPreconditioner is only for GPUs
    @time Pl = KrylovPreconditioners.kp_ilu0(A)
    @time sol8 = solve(prob, KrylovJL_GMRES(; restart = true), Pl = Pl)
end

foo, baz = let
    function setup_2D(n=128,T::Type=Float64)
        L = zeros(T,n+2,n+2,2); L[3:n+1,2:n+1,1] .= 1; L[2:n+1,3:n+1,2] .= 1;
        x = T[i-1 for i ∈ 1:n+2, j ∈ 1:n+2]
        Poisson(L),FieldVector(x)
    end

    A, x = setup_2D(4)

    A, x
end

foo

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

# prob = LinearProblem(A, b)
# alg = HYPREAlgorithm(HYPRE.PCG)
# prec = HYPRE.BoomerAMG
# @time sol = solve(prob, alg; Pl = prec)

# Direct HYPRE test
HYPRE.Init()
A_h = HYPREMatrix(A)
b_h = HYPREVector(b)
Tol = 1e-12
# gmres = HYPRE.GMRES(; Tol)
# @time x_h = HYPRE.solve(gmres, A_h, b_h)

Precond = HYPRE.BoomerAMG()
gmres = HYPRE.FlexGMRES(; Tol, Precond)
@time x_ph = HYPRE.solve(gmres, A_h, b_h)
x = zeros(size(b))
copy!(x, x_ph)

amg = HYPRE.BoomerAMG(; Tol)
x_h = HYPRE.solve(amg, A_h, b_h)
copy!(x, x_h)


# Try splitting the operator first
# xs = 0
# Tii xi = 1
idx = findall(issrf .== 0)
Ai = A[idx, idx]
bi = b[idx]
Ai_h = HYPREMatrix(Ai)
bi_h = HYPREVector(bi)
# Precond = HYPRE.BoomerAMG()
Precond = HYPRE.ILU()
# gmres = HYPRE.FlexGMRES(; Tol, Precond)
amg = HYPRE.BoomerAMG()
@time xi_h = HYPRE.solve(amg, Ai_h, bi_h)
xi = zeros(size(bi))
copy!(xi, xi_h)

bicg = HYPRE.BiCGSTAB(; Precond)
@time xi_h = HYPRE.solve(bicg, Ai_h, bi_h)
xi = zeros(size(bi))
copy!(xi, xi_h)






# copy-pasted from HYPRE.jl tests
A = sprand(100, 100, 0.05); A = A + 5I
b = rand(100)
x = zeros(100)

A = T + sparse(Diagonal(issrf))
b = ones(size(A, 1))
x = zeros(size(b))

A_h = HYPREMatrix(A)
b_h = HYPREVector(b)
x_h = HYPREVector(x)
# Solve
tol = 1.0e-9
ilu = HYPRE.ILU(; Tol = tol)
HYPRE.solve!(ilu, x_h, A_h, b_h)
copy!(x, x_h)
x ≈ A \ b

x_h = HYPRE.solve(bicg, A_h, b_h)
copy!(x, x_h)
x ≈ A \ b

[A * x b]



