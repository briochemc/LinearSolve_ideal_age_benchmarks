# # The age tracer equation is
# #
# #   ∂x(t)/∂t + T(t) x(t) = 1 - Ω x(t)
# #
# # where Ω "relaxes" x to zero in the top layer.
# #
# # Applying Backward Euler time step gives
# #
# #   (I + δt M) x(t+δt) = x(t) + δt
# #
# # where M = T + Ω.
# #
# # So maybe I can try to use Newton to solve G(x) = (I + δt M) \ (x + δt) - x(t) = 0
# # (which is the same as solving δt - δt M x = 0    or   M x = 1)
# # But this time I only coarsen M (not I).
# #
# # So instead of solving Mc xc = 1 where Mc = L M S and then spraying x = S xc,
# # I use GMRES to solve (I + δt M) x = x0 + δt many times, preconditioned with
# # y -> S (I + δt Mc)⁻¹ L y
# # And wrap that into a nonlinear solver with the same preconditioner?
# # (I + δt Mc) xc = xc0 + δt many times, and that should be fast as I + δt Mc is factorized only once.
# #
# # So maybe the first thing is to check that GMRES works for the linear system
# # That is, solve (I + δt M) x = b
# # And then check that GMRES(I + δt M,  b) with preconditioner P(y) = (I - δt S Mc⁻¹ L) y
# # works too? My hope is that having the I not be part of the lumping/spraying will
# # help.

# using Pkg
# Pkg.activate(".")

# using AIBECS
# using LinearSolve
# using IterativeSolvers
# using LinearAlgebra
# using IncompleteLU
# using Unitful
# using Unitful: m, s, yr, Myr
# using Plots
# import Pardiso # import Pardiso instead of using (to avoid name clash?)
# using OceanTransportMatrixBuilder

# @info "Loading the grid and transport matrix..."

# # Define the model
# grd, T = OCCA.load()

# @info "Grid size: $(size(grd.wet3D))"
# @info "Number of wet cells: $(count(grd.wet3D))"
# @info "Number of nonzeros in T: $(nnz(T))"

# @info "Setting up the linear problem..."

# issrf = let
#     issrf3D = zeros(size(grd.wet3D))
#     issrf3D[:,:,1] .= 1
#     issrf3D[grd.wet3D]
# end

# M = T + sparse(Diagonal(issrf))

# δt = ustrip(s, 1yr)
# b = δt * ones(size(M, 1))


# prob = LinearProblem(I + δt * M, b, u0 = b .+ δt)

# @info "Solve the system using default backslash"
# @time "backslash" sol0 = (I + δt * M) \ b
# @show norm((I + δt * M) * sol0 - b) / norm(b)

# @info "Now try GMRES with multigrid preconditioner"
# wet3D = grd.wet3D
# v = grd.volume_3D[wet3D]
# LUMP, SPRAY, v_c = OceanTransportMatrixBuilder.lump_and_spray(wet3D, v, T; di=2, dj=2, dk=1)

# @info "factorize coarse operator"
# M_c = LUMP * M * SPRAY
# M_c_factor = factorize(M_c)

# @info "define preconditioner"
# struct MyPreconditioner
#     M_c_factor
# end
# Base.eltype(::MyPreconditioner) = Float64
# function LinearAlgebra.ldiv!(Pl::MyPreconditioner, x::AbstractVector)
#     x .= x .- δt * SPRAY * (Pl.M_c_factor \ (LUMP * x))
# end
# function LinearAlgebra.ldiv!(y::AbstractVector, Pl::MyPreconditioner, x::AbstractVector)
#     y .= x .- δt * SPRAY * (Pl.M_c_factor \ (LUMP * x))
# end

@info "Solve the system using MG+GMRES"
@time "GMRES" sol_gmres = copy(solve(prob, KrylovJL_GMRES(); maxiters = 500, restarts = 50, verbose = true, reltol = 1.0e-12).u)
@show norm((I + δt * M) * sol_gmres - b) / norm(b)
plt = plothorizontalslice((sol_gmres - sol0) * s .|> yr, grd, depth = 1000m, clim = (-200, 200), cmap = :balance)
savefig(plt, "GMRES_tracer_error.png")

Pl = MyPreconditioner(M_c_factor)
@time "MG + GMRES" sol_mg = copy(solve(prob, KrylovJL_GMRES(); Pl = Pl, maxiters = 500, restarts = 50, verbose = true, reltol = 1.0e-12).u)
@show norm((I + δt * M) * sol_mg - b) / norm(b)
plt = plothorizontalslice((sol_mg - sol0) * s .|> yr, grd, depth = 1000m, clim = (-200, 200), cmap = :balance)
savefig(plt, "MG+GMRES_tracer_error.png")

@time "ILU fact" Pl = IncompleteLU.ilu(I + δt * M, τ = 1.0e-8)
@time "ILU solve" sol_ilu = copy(solve(prob, KrylovJL_GMRES(); Pl = Pl, maxiters = 500, restarts = 50, verbose = true, reltol = 1.0e-12).u)
@show norm((I + δt * M) * sol_ilu - b) / norm(b)
plt = plothorizontalslice((sol_ilu - sol0) * s .|> yr, grd, depth = 1000m, clim = (-200, 200), cmap = :balance)
savefig(plt, "ILU+GMRES_tracer_error.png")

@show norm((I + δt * M) * sol_gmres - b) / norm(b)
@show norm((I + δt * M) * sol_mg - b) / norm(b)
@show norm((I + δt * M) * sol_ilu - b) / norm(b)
