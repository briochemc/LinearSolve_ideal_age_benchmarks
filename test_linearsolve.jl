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
using Unitful: m, s, yr, Myr
using Plots
using PETSc
import Pardiso # import Pardiso instead of using (to avoid name clash?)
using RestrictProlong

@info "Loading the grid and transport matrix..."

# Define the model
grd, T = OCCA.load()
foo
# grd, T = OCIM2_48L.load()
# grd, T = Primeau_2x2x2.load()

@info "Grid size: $(size(grd.wet3D))"
@info "Number of wet cells: $(count(grd.wet3D))"
@info "Number of nonzeros in T: $(nnz(T))"

@info "Setting up the linear problem..."

issrf = let
    issrf3D = zeros(size(grd.wet3D))
    issrf3D[:,:,1] .= 1
    issrf3D[grd.wet3D]
end

A = T + sparse(Diagonal(issrf))

b = ones(size(A, 1))

# Solve the system using the LinearSolve.jl package
prob = LinearProblem(A, b, u0 = ustrip(s, 1000yr) * ones(size(b)))


sol0 = A \ b
plt0 = plothorizontalslice(sol0 * s .|> yr, grd, depth=1000m, clim=(0, 2000))
@time "backslash" sol0 = A \ b

sol1 = solve(prob)
plt1 = plothorizontalslice((sol1 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
@time "default LinearSolve solve" sol1 = solve(prob)

sol2 = solve(prob, KLUFactorization())
plt2 = plothorizontalslice((sol2 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
@time "KLUFactorization" sol2 = solve(prob, KLUFactorization())

sol3 = solve(prob, UMFPACKFactorization())
plt3 = plothorizontalslice((sol3 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
@time "UMFPACKFactorization" sol3 = solve(prob, UMFPACKFactorization())

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
    prob6 = init(prob, solver, reltol = 1e-10)
    @time sol6 = solve!(prob6).u
end


if false # does not seem to work with OCCA matrix
    @time Pl = IncompleteLU.ilu(A, τ = 0.0) # even with "full" LU, which takes ages to factorize!?
    @time sol7 = solve(prob, KrylovJL_GMRES(; restart = true), Pl = Pl)
end

if false # KrylovPreconditioner is only for GPUs
    @time Pl = KrylovPreconditioners.kp_ilu0(A)
    @time sol8 = solve(prob, KrylovJL_GMRES(; restart = true), Pl = Pl)
end



# Try my own lump-solve-spray as preconditioner (but with Tim Holy's anti-alising idea)
wet3D = grd.wet3D
wet = wet3D[:]
nx, ny, nz = size(wet3D)
# Create lumping matrices in each dimension
vi = repeat(1:nx, inner=2)[1:nx]
vj = repeat(1:ny, inner=2)[1:ny]
vk = repeat(1:nz, inner=1)[1:nz]
LUMPx = sparse(vi, 1:nx, true)
LUMPy = sparse(vj, 1:ny, true)
LUMPz = sparse(vk, 1:nz, true)
# Wrap along longitude
# LUMPx = sparse(restrict(I(nx + 4), 1))
# LUMPx = LUMPx[2:end-1, :] # remove ghost rows
# LUMPx_west_ghost = LUMPx[:, 2]
# LUMPx_east_ghost = LUMPx[:, end-1]
# LUMPx = LUMPx[:, 3:end-2] # remove ghost columns
# LUMPx[:, 1] .+= LUMPx_east_ghost # add ghost cell weights to first real cell
# LUMPx[:, end] .+= LUMPx_west_ghost # add ghost cell weights to last real cell
# LUMPx = nx / sum(LUMPx) * LUMPx # normalize to conserve volume? # CHECK if necessary after volume weighting below
# LUMPx

# LUMPx = sparse(restrict(I(nx), 1))
# LUMPx = nx / sum(LUMPx) * LUMPx # normalize to conserve volume? # CHECK if necessary after volume weighting below

# LUMPy = sparse(restrict(I(ny), 1))
# LUMPy = ny / sum(LUMPy) * LUMPy # normalize to conserve volume? # CHECK if necessary after volume weighting below

# LUMPz = sparse(1.0I(nz))
# kron each dimension to build whole LUMP matrix
LUMP = kron(LUMPz, kron(LUMPy, LUMPx))

# Find wet points in coarsened grid
wet_c = LUMP * wet .> 0
nx_c = size(LUMPx, 1)
ny_c = size(LUMPy, 1)
nz_c = size(LUMPz, 1)
wet3D_c = fill(false, nx_c, ny_c, nz_c)
wet3D_c[wet_c] .= true

# Extract only indices of wet grd points
LUMP = LUMP[wet_c, wet]

# Make the LUMP operator mass-conserving
# by volume integrating on the right and dividing by the coarse
# volume on the left
volume = grd.volume_3D[wet3D]
volume_c = LUMP * volume
LUMP = sparse(Diagonal(1 ./ volume_c)) * LUMP * sparse(Diagonal(volume))

# The SPRAY operator just copies the values back
# so it is sinply 1's with the transposed sparsity structure
# SPRAY = copy(LUMP')
# SPRAY.nzval .= 1
SPRAY = transpose(LUMP)

# CHeck mass conservation
xtest = rand(size(wet))
volume' * xtest ≈ volume_c' * (LUMP * xtest) # should be true

issrf_c = let
    issrf3D_c = zeros(size(wet3D_c))
    issrf3D_c[:,:,1] .= 1
    issrf3D_c[wet3D_c]
end

v = grd.volume_3D[wet3D]
e1 = ones(size(v))

τdiv = ustrip(Myr, norm(e1) / norm(T * e1) * s)
τdiv > 1e6
@info "    div: $(round(τdiv, sigdigits=2)) Myr"

τvol = ustrip(Myr, norm(v) / norm(T' * v) * s)
τvol > 1e6
@info "    vol: $(round(τvol, sigdigits=2)) Myr"

T_c = LUMP * T * SPRAY
e1_c = ones(size(T_c, 1))

τdiv_c = ustrip(Myr, norm(e1_c) / norm(T_c * e1_c) * s)
τdiv_c > 1e6
@info "    div: $(round(τdiv_c, sigdigits=2)) Myr"

v_c = volume_c
τvol_c = ustrip(Myr, norm(v_c) / norm(T_c' * v_c) * s)
τvol_c > 1e6
@info "    vol: $(round(τvol_c, sigdigits=2)) Myr"

A_c = LUMP * T * SPRAY + sparse(Diagonal(issrf_c))
A_c_factor = factorize(A_c)
A_c_factor = factorize(LUMP * A * SPRAY) # equivalent and maybe faster?


struct MyPreconditioner
    SPRAY
    LUMP
    A_c_factor
end
Base.eltype(::MyPreconditioner) = Float64
function LinearAlgebra.ldiv!(Pl::MyPreconditioner, x::AbstractVector)
    @info "applying 2-arg Pl"
    x .= Pl.SPRAY * (Pl.A_c_factor \ (Pl.LUMP * x))
end
function LinearAlgebra.ldiv!(y::AbstractVector, Pl::MyPreconditioner, x::AbstractVector)
    @info "applying 3-arg Pl"
    y .= Pl.SPRAY * (Pl.A_c_factor \ (Pl.LUMP * x))
end
Pl2 = MyPreconditioner(SPRAY, LUMP, A_c_factor)
Pr = I
precs = Returns((Pl2, Pr))


struct MySuperPreconditioner
    A_factor
end
Base.eltype(::MySuperPreconditioner) = Float64
LinearAlgebra.ldiv!(Pl::MySuperPreconditioner, x::AbstractVector) = ldiv!(Pl.A_factor, x)
LinearAlgebra.ldiv!(y::AbstractVector, Pl::MySuperPreconditioner, x::AbstractVector) = ldiv!(y, Pl.A_factor, x)
superPl = MySuperPreconditioner(factorize(A))
Pr = I
superprecs = Returns((superPl, Pr))

sol_mg1 = solve(prob, KrylovJL_GMRES(); Pl = superPl, maxiters = 100, restarts = 50, verbose = true, reltol = 1e-12)
plot(sol_mg1 * s .|> yr)
@time "KrylovJL_GMRES + lu(A) as preconditioner" sol_mg1 = solve(prob, KrylovJL_GMRES(); Pl = superPl, maxiters = 100, restarts = 50, verbose = true, reltol = 1e-12)

sol_mg2 = solve(prob, KrylovJL_GMRES(); Pl = Pl2, maxiters = 100, restarts = 50, verbose = true, reltol = 1e-12)
plot(sol_mg2 * s .|> yr)
@time "KrylovJL_GMRES + coarsened LU as preconditioner" sol_mg2 = solve(prob, KrylovJL_GMRES(); Pl = Pl2, maxiters = 100, restarts = 50, verbose = true, reltol = 1e-12)
plt_mg2 = plothorizontalslice((sol_mg2 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)


scatter(sol_mg1 * s .|> yr, sol_mg2 * s .|> yr, markersize = 1)


# AlgebraicMultigrid: Implementations of the algebraic multigrid method. Must be converted to a preconditioner via AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.precmethod(A)). Requires A as a AbstractMatrix. Provides the following methods:
# Pl = AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.smoothed_aggregation(A; max_levels=2, coarse_solver=AlgebraicMultigrid.LinearSolveWrapper(UMFPACKFactorization())));
prob_normal = LinearProblem(A' * A, A' * b)
# Pl = AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.ruge_stuben(A' * A))
Pl = AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.smoothed_aggregation(A' * A))
# sol6 = solve(prob_normal, KrylovJL_CG(); Pl, maxiters = 1000, restarts = 50, verbose = true, reltol = 1e-10)
sol6 = solve(prob_normal, KrylovJL_GMRES(); Pl, maxiters = 500, verbose = true, reltol = 1e-10)

Pl = AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.smoothed_aggregation(A))
sol6 = solve(prob, KrylovJL_GMRES(); Pl, maxiters = 500, verbose = true, reltol = 1e-10)
plt6 = plothorizontalslice((sol6 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
@time "KrylovJL_GMRES + AMG smoothed_aggregation" sol6 = solve(prob, KrylovJL_GMRES(); Pl, maxiters = 100, restarts = 50, verbose = true, reltol = 1e-12)

# Pl = AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.smoothed_aggregation(A; max_levels=3, coarse_solver=AlgebraicMultigrid.LinearSolveWrapper(UMFPACKFactorization())))
# sol7 = solve(prob, IterativeSolversJL_CG(); Pl)
# plt7 = plothorizontalslice((sol7 - sol0) * s .|> yr, grd, depth=1000m, clim=(-400, 400), cmap=:balance)
# plt7 = plothorizontalslice(sol7 * s .|> yr, grd, depth=1000m, clim=(0, 2000))
# @time sol7 = solve(prob, IterativeSolversJL_CG(); Pl)


if false # throws because A is not symmetric
    strategy = KrylovJL_CG(precs = RugeStubenPreconBuilder())
    sol = solve(prob, strategy, atol=1.0e-14)

    strategy = KrylovJL_CG(precs = SmoothedAggregationPreconBuilder())
    sol = solve(prob, strategy, atol=1.0e-14)
end

# Pl = PyAMG.aspreconditioner(PyAMG.precmethod(A))


# IncompleteLU.ilu: an implementation of the incomplete LU-factorization preconditioner. This requires A as a SparseMatrixCSC.
@time Pl = IncompleteLU.ilu(A, τ=1e-8) # arbitriry largest τ for which solver worked fast
sol8 = solve(prob, KrylovJL_GMRES(); Pl, maxiters = 1000, restarts = 50, verbose = true, reltol = 1e-12)
plt8 = plothorizontalslice((sol8 - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
@time "KrylovJL_GMRES + ILU(τ=1e-8) as preconditioner" sol8 = solve(prob, KrylovJL_GMRES(); Pl, maxiters = 1000, restarts = 50, verbose = true, reltol = 1e-12)





# Try PETSc
MPI.Initialized() || MPI.Init()
PETSc.initialize()

# don't print these below
M_PETSc = PETSc.MatSeqAIJ(A);
b_PETSc = PETSc.VecSeq(b);
x_PETSc = PETSc.VecSeq(zeros(size(b)));

ksp = PETSc.KSP(
    M_PETSc;
    ksp_monitor = false,    # set to true for output
    ksp_monitor_true_residual = false,
    ksp_view = false,
    log_view = true,
    ksp_type = "gmres",
    ksp_rtol = 1e-10,
    pc_type = "hypre",
    # mg_levels_ksp_type = "chebyshev",
    # mg_levels_ksp_max_it = 3,
    # mg_levels_pc_type = "bjacobi",
    # mg_levels_sub_pc_type = "icc",
    # mg_coarse_ksp_type = "preonly",
    # mg_coarse_pc_type = "cholesky",
);

@time PETSc.solve!(x_PETSc, ksp, b_PETSc)
sol_PETSc = x_PETSc.array
plt_PETSc = plothorizontalslice(100(sol_PETSc - sol0) ./ sol0, grd, depth=1000m, clim=(-100, 100), cmap=:balance)


foo


# BoomerAMG preconditioner
# Pl = HYPRE.BoomerAMG
# sol = solve(prob, HYPREAlgorithm(HYPRE.PCG); Pl)

# prob = LinearProblem(A, b)
# alg = HYPREAlgorithm(HYPRE.PCG)
# prec = HYPRE.BoomerAMG
# @time sol = solve(prob, alg; Pl = prec)







# Direct HYPRE test

using HYPRE.LibHYPRE: HYPRE_BigInt,
                      HYPRE_Complex, HYPRE_IJMatrixGetValues,
                      HYPRE_IJVectorGetValues, HYPRE_Int

# Convert from HYPREArrays to Julia arrays
function to_array(A::HYPREMatrix)
    i = (A.ilower):(A.iupper)
    j = (A.jlower):(A.jupper)
    nrows = HYPRE_Int(length(i))
    ncols = fill(HYPRE_Int(length(j)), length(i))
    rows = convert(Vector{HYPRE_BigInt}, i)
    cols = convert(Vector{HYPRE_BigInt}, repeat(j, length(i)))
    values = Vector{HYPRE_Complex}(undef, length(i) * length(j))
    HYPRE_IJMatrixGetValues(A.ijmatrix, nrows, ncols, rows, cols, values)
    return sparse(permutedims(reshape(values, (length(j), length(i)))))
end
function to_array(b::HYPREVector)
    i = (b.ilower):(b.iupper)
    nvalues = HYPRE_Int(length(i))
    indices = convert(Vector{HYPRE_BigInt}, i)
    values = Vector{HYPRE_Complex}(undef, length(i))
    HYPRE_IJVectorGetValues(b.ijvector, nvalues, indices, values)
    return values
end
to_array(x) = x



HYPRE.Init()
A_h = HYPREMatrix(A)
b_h = HYPREVector(b)

prob_h = LinearProblem(A_h, b_h)

Tol = 1e-12
# gmres = HYPRE.GMRES(; Tol)
# @time x_h = HYPRE.solve(gmres, A_h, b_h)
# alg = HYPREAlgorithm(HYPRE.BoomerAMG) # fails
# alg = HYPREAlgorithm(HYPRE.BiCGSTAB) # fails
# alg = HYPREAlgorithm(HYPRE.FlexGMRES) # fails
alg = HYPREAlgorithm(HYPRE.GMRES) # fails
# alg = HYPREAlgorithm(HYPRE.Hybrid) # fails (segfaults?)
# alg = HYPREAlgorithm(HYPRE.ILU) # fails
# alg = HYPREAlgorithm(HYPRE.PCG)
# Pl = HYPRE.BoomerAMG
Pl = HYPRE.ILU
y = solve(prob_h, alg; Pl, verbose=true, maxiters=5000, reltol=1e-12)
y = solve(prob_h, alg; verbose=true, maxiters=1000)
y = solve(prob_h, alg; verbose=true, maxiters=1000, Pl = HYPRE.ParaSails)
sol10 = to_array(y.u)



Precond = HYPRE.BoomerAMG()

x_ph = HYPRE.solve(gmres, A_h, b_h)

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



