using Pkg
Pkg.activate(".")
Pkg.instantiate()

using AIBECS
using LinearSolve
using MPI
using LinearAlgebra
using Unitful
using Unitful: m, s, yr, Myr
using Plots
using PETSc

MPI.Initialized() || MPI.Init()
PETSc.initialize()

@info "Loading the grid and transport matrix..."

# Define the model
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
@time sol = A \ b # reference solution


# don't print these below
M_PETSc = PETSc.MatSeqAIJ(A);
b_PETSc = PETSc.VecSeq(b);
x_PETSc = PETSc.VecSeq(copy(b));

ksp = PETSc.KSP(
    M_PETSc;
    ksp_monitor = false,    # set to true for output
    ksp_monitor_true_residual = false,
    ksp_view = false,
    ksp_type = "gmres",
    ksp_atol = 1.0e-10,
    pc_type = "ilu", # WORKS!
    # pc_type = "gamg", # Does not work well
    # pc_type = "ml", # throws
    # pc_type = "mg", # Does not work at all
    # pc_type = "hypre", # Does not work at all
    # pc_type = "amgx", # throws
    # pc_type = "hmg", # Does not work at all
    # pc_type = "fmg", # throws
    # pc_type = "hmg", # Does not work at all
    # pc_type = "hpddm", # throws
    # pc_type = "jacobi", # Does not work well
    # pc_type = "pbjacobi", # Does not work well
    # pc_type = "sor", # Does not work at all
    # pc_type = "kaczmarz", # Does not work at all
    # pc_type = "asm", # WORKS!
    # pc_type = "patch", # throws
    # pc_type = "deflation", # does not work well
    # pc_factor_levels = 0,
    # mg_levels_ksp_type = "chebyshev",
    # mg_levels_ksp_max_it = 3,
    # mg_levels_pc_type = "bjacobi",
    # mg_levels_sub_pc_type = "icc",
    # mg_coarse_ksp_type = "preonly",
    # mg_coarse_pc_type = "cholesky",
);

@time PETSc.solve!(x_PETSc, ksp, b_PETSc);
sol_PETSc = x_PETSc.array

@show norm(sol_PETSc - sol) / norm(sol)
plot(collect(extrema(sol)), collect(extrema(sol)), label = "")
plot!(sol, sol_PETSc, seriestype = :scatter, label = "")

plt0 = plothorizontalslice(sol * s .|> yr, grd, depth = 1000m, clim = (0, 2000))
plt1 = plothorizontalslice(sol_PETSc * s .|> yr, grd, depth = 1000m, clim = (0, 2000))
plt_PETSc = plothorizontalslice(100(sol_PETSc - sol) ./ sol, grd, depth = 1000m, clim = (-100, 100), cmap = :balance)
