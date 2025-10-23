# The age tracer equation is
#
#   ∂x(t)/∂t + T(t) x(t) = 1 - Ω x(t)
#
# where Ω "relaxes" x to zero in the top layer.
#
# Applying Forward Euler time step gives
#
#   x(t+δt) = (I - δt M) x(t) + δt
#
# where M = T + Ω.

# qsub -I -P y99 -q express -l mem=47GB -l storage=scratch/gh0+scratch/xv83+scratch/p66 -l walltime=01:00:00 -l ncpus=12

using Pkg
Pkg.activate(".")

const nprocs = 12 # Make sure you match ncpus in qsub

using AIBECS
using LinearSolve
using NonlinearSolve
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

δt = ustrip(s, 1yr)
N = size(M, 1)
b = ones(N)


@info "Solve the system using default backslash"
@time "backslash" sol0 = (I + δt * M) \ b
@show norm((I + δt * M) * sol0 - b) / norm(b)

@info "Now try Newton–Krylov with multigrid-preconditioned GMRES"
wet3D = grd.wet3D
v = grd.volume_3D[wet3D]
LUMP, SPRAY, v_c = OceanTransportMatrixBuilder.lump_and_spray(wet3D, v, T; di = 2, dj = 2, dk = 1)

M_c = LUMP * M * SPRAY

# Left Preconditioner needs a new type
struct MyPreconditioner
    prob
end
Base.eltype(::MyPreconditioner) = Float64
function LinearAlgebra.ldiv!(Pl::MyPreconditioner, x::AbstractVector)
    Pl.prob.b = LUMP * x
    solve!(Pl.prob)
    x .= SPRAY * Pl.prob.u .- x
    return x
end
function LinearAlgebra.ldiv!(y::AbstractVector, Pl::MyPreconditioner, x::AbstractVector)
    Pl.prob.b = LUMP * x
    solve!(Pl.prob)
    y .= SPRAY * Pl.prob.u .- x
    return y
end

@info "Setting up Pardiso solver"
matrix_type = Pardiso.REAL_SYM
@show solver = MKLPardisoIterate(; nprocs, matrix_type)

@info "Set up preconditioner problem"
Plprob = LinearProblem(-δt * M_c, ones(N))  # following Bardin et al. (M -> -M though)
Plprob = init(Plprob, solver, rtol = 1.0e-10)
Pl = MyPreconditioner(Plprob)
Pr = I
precs = Returns((Pl, Pr))


# Line below throws!?
@time "initial state solve" u0 = solve(LinearProblem(M, ones(N)), solver, rtol = 1.0e-10).u
@show norm(M̄ * u0 - ones(N)) / norm(ones(N))


# Euler forward time steps
function stepforwardoneyear!(du, u)
    du .= u .- dt * M * u .+ dt
    return du
end
function jvponeyear!(dv, v)
    dv .= v .- dt * M * v
    return dv
end
function G!(du, u)
    stepforwardoneyear!(du, u)
    du .-= u
    return du
end
function jvp!(dv, v, u)
    jvponeyear!(dv, v)
    dv .-= v
    return dv
end
f! = NonlinearFunction(G!; jvp = jvp!)
nonlinearprob! = NonlinearProblem(f!, u0)

@info "solve periodic state (here it's actually steady state)"
# @time sol = solve(nonlinearprob, NewtonRaphson(linsolve = KrylovJL_GMRES(precs = precs)), verbose = true, reltol=1e-10, abstol=Inf);
@time sol! = solve(nonlinearprob!, NewtonRaphson(linsolve = KrylovJL_GMRES(precs = precs, rtol = 1.0e-12)); show_trace = Val(true), reltol = Inf, abstol = 1.0e-10norm(u0, Inf));
