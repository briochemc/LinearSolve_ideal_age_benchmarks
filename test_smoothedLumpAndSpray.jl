using Pkg
Pkg.activate(".")

using AIBECS
using LinearSolve
using IterativeSolvers
using LinearAlgebra
using IncompleteLU
using Unitful
using Unitful: m, s, yr, Myr
using Plots
import Pardiso # import Pardiso instead of using (to avoid name clash?)
using OceanTransportMatrixBuilder

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

M = T + sparse(Diagonal(issrf))

b = ones(size(M, 1))

prob = LinearProblem(M, b, u0 = 1000 * ones(size(b)))

@info "Solve the system using default backslash"
@time "backslash" sol0 = copy(solve(prob).u)
@show norm(M * sol0 - b) / norm(b)

@info "Now try GMRES with multigrid with smoothed preconditioner"
wet3D = grd.wet3D
v = grd.volume_3D[wet3D]
LUMP, SPRAY, v_c = OceanTransportMatrixBuilder.lump_and_spray(wet3D, v, T; di = 2, dj = 2, dk = 1)

# One implicit backward time step
# (x(t+dt) - x(t)) / dt + M x(t+dt) = 1
# (I + dt M) x(t+dt) = x(t) + dt
# One CN time step
# (x(t+dt) - x(t)) / dt + M (x(t+dt) + x(t))/2 = 1
# (I + dt/2 M) x(t+dt) = (I - dt/2 M) x(t) + dt
dt = ustrip(s, 1yr)
M⁺
TIMESTEPfac = factorize(I + dt / 2 * M)
TIMESTEP(x) = TIMESTEPfac \ ((I - dt / 2 * M) * x .+ dt)

# SPRAY is the tentative unsmoothed prolongation operator?
ω = 0.7 # smoothing parameter
SMOOTHERONES = SPRAY * LUMP
SMOOTHERONES.nzval .= 1.0
SMOOTHER = SMOOTHERONES .* M # Simpler than M so that it doesnt lump in vertical.
PROLONG = (I - ω * SMOOTHER) * SPRAY # smoothed prolongation operator
RESTRICT = transpose(PROLONG) - ω * transpose(PROLONG) * SMOOTHER # smoothed restriction operator
# source: https://trilinos.github.io/pdfs/MueLu_tutorial.pdf

@info "factorize coarse operator"
M_c = RESTRICT * M * PROLONG
M_c_factor = factorize(M_c)

@info "define preconditioner"
struct MyPreconditioner
    M_c_factor
end
Base.eltype(::MyPreconditioner) = Float64
function LinearAlgebra.ldiv!(Pl::MyPreconditioner, x::AbstractVector)
    # x .= PROLONG * (Pl.M_c_factor \ (RESTRICT * x))
    return x .= TIMESTEP(SPRAY * (Pl.M_c_factor \ (LUMP * TIMESTEP(x))))
end
function LinearAlgebra.ldiv!(y::AbstractVector, Pl::MyPreconditioner, x::AbstractVector)
    # y .= PROLONG * (Pl.M_c_factor \ (RESTRICT * x))
    return y .= TIMESTEP(SPRAY * (Pl.M_c_factor \ (LUMP * TIMESTEP(x))))
end

# @info "Solve the system using MG+GMRES"
# @time "GMRES" sol_gmres = copy(solve(prob, KrylovJL_GMRES(); maxiters = 500, restarts = 50, verbose = true, reltol = 1e-12).u)
# @show norm(M * sol_gmres - b) / norm(b)
# plt = plothorizontalslice((sol_gmres - sol0) * s .|> yr, grd, depth=1000m, clim=(-200, 200), cmap=:balance)
# savefig(plt, "GMRES_tracer_error.png")

Pl = MyPreconditioner(M_c_factor)
@time "MG + GMRES" sol_mg = copy(solve(prob, KrylovJL_GMRES(); Pl = Pl, maxiters = 500, restarts = 50, verbose = true, reltol = 1.0e-12).u)
@show norm(M * sol_mg - b) / norm(b)
plt = plothorizontalslice((sol_mg - sol0) * s .|> yr, grd, depth = 1000m, clim = (-200, 200), cmap = :balance)
savefig(plt, "sMG+GMRES_tracer_error.png")

@time "ILU fact" Pl = IncompleteLU.ilu(M, τ = 1.0e-8)
@time "ILU solve" sol_ilu = copy(solve(prob, KrylovJL_GMRES(); Pl = Pl, maxiters = 500, restarts = 50, verbose = true, reltol = 1.0e-12).u)
@show norm(M * sol_ilu - b) / norm(b)
plt = plothorizontalslice((sol_ilu - sol0) * s .|> yr, grd, depth = 1000m, clim = (-200, 200), cmap = :balance)
savefig(plt, "ILU+GMRES_tracer_error.png")

@show norm(M * sol_gmres - b) / norm(b)
@show norm(M * sol_mg - b) / norm(b)
@show norm(M * sol_ilu - b) / norm(b)
