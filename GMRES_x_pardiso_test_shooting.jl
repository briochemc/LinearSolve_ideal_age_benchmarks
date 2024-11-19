using Pkg
Pkg.activate(".")
Pkg.instantiate()

using AIBECS
using LinearSolve
import Pardiso # that's what they do in LinearSolve.jl benchmarks (maybe to avoid name clash)
using NonlinearSolve
using SparseArrays
using LinearAlgebra
using Unitful
using Unitful: m, s, yr, d
using Statistics

# Define the model
# grd, T = Primeau_2x2x2.load()
# grd, T = Archer_etal_2000.load()
# grd, T = OCCA.load()
# grd, T = OCIM0.load()
# grd, T = OCIM1.load()
grd, T = OCIM2_48L.load()
N = size(T, 1)

v = grd.volume_3D[grd.wet3D];

issrf = let
    issrf3D = falses(size(grd.wet3D))
    issrf3D[:,:,1] .= true
    issrf3D[grd.wet3D]
end
Ω = sparse(Diagonal(Float64.(issrf)))

# Now let's imagine we have 4 steps of OCCA where T gets more or less intense
# steps = ("DJF", "MAM", "JJA", "SON")
steps = 1:12
Nsteps = length(steps)
αs = @. 1 + 0.1cos(2π * steps / Nsteps)

δt = ustrip(s, 1yr / Nsteps) # TODO maybe use exact mean number of days (more important for monthly because Feb)?

# The shooting method idea is to simply perform the n implicit time steps and solve for the zero of
#   G(x₁) = x₁₃ - x₁ = ∏ₘ Aₘ⁻¹ x₁ - x₁ +  δt ∑ₘ (∏ₖ₌₁ᵐ Aₖ⁻¹) 1 = 0
# so
#   (I - ∏ₘ Aₘ⁻¹) x₁ = δt ∑ₘ (∏ₖ₌₁ᵐ Aₖ⁻¹) 1
#
#   G(x₁) = x₁₃(x₁) - x₁
#   J(x₁) = I - ∏ₘ Aₘ⁻¹ <- don't form this, only its action on x₁, which is
#
#   ∂x/∂t + (T + Ω) x = 1      (Bardin's 𝐌 = -(T + Ω) so (Mbar) 𝑇𝐌̄ = -𝑇(T̄ + Ω) and 𝐉⁻¹ = (-𝑇(T̄ + Ω))⁻¹ - 𝐈)
# So approximate year time step may be
#   x(yr+1) = (I + Δt * (T̄ + Ω))⁻¹ (x + Δt 1)
# And therefore approximate Jacobian may be
#   J x ≈ (I + Δt * (T̄ + Ω))⁻¹ x - x =
# So J ≈ (I + Δt * (T̄ + Ω))⁻¹ - I
#      ≈ -Δt (T̄ + Ω)⁻¹
#
#   (I + δt * (α * Tₘ₊₁ + Ω)) xₘ₊₁ = Aₘ₊₁ xₘ₊₁ = xₘ + δt 1
#   xₘ₊₁ = Aₘ₊₁⁻¹ (xₘ + δt 1)
#   x₁₃ = A₁⁻¹ x₁₂ + δt A₁⁻¹ 1
#              x₁₂ = A₁₂⁻¹ x₁₁ + δt A₁₂⁻¹ 1
#   x₁₃ = A₁⁻¹ A₁₂⁻¹ x₁₁ + δt A₁⁻¹ A₁₂⁻¹ 1 + δt A₁⁻¹ 1
#                    x₁₁ = A₁₁⁻¹ x₁₀ + δt A₁₁⁻¹ 1
#   x₁₃ = A₁⁻¹ A₁₂⁻¹ A₁₁⁻¹ x₁₀ + δt  A₁⁻¹ A₁₂⁻¹ A₁₁⁻¹ 1 + δt A₁⁻¹ A₁₂⁻¹ 1 + δt A₁⁻¹ 1
#   ⋮
#   x₁₃ = ∏ₘ Aₘ⁻¹ x₁ +  δt ∑ₘ (∏ₖ₌₁ᵐ Aₖ⁻¹) 1
#


# "load" the seasonal matrices
As = [I + δt * (α * T + Ω) for (α, season) in zip(αs, steps)]
Ms = [α * T + Ω for (α, season) in zip(αs, steps)]


# Left Preconditioner needs a new type
struct CycloPreconditioner
    prob
end
Base.eltype(::CycloPreconditioner) = Float64
function LinearAlgebra.ldiv!(Pl::CycloPreconditioner, x::AbstractVector)
    @info "applying Pl"
    Pl.prob.b = x
    solve!(Pl.prob)
    x .= Pl.prob.u .- x # Note the -x (following Bardin et al)
    return x
end
function LinearAlgebra.ldiv!(y::AbstractVector, Pl::CycloPreconditioner, x::AbstractVector)
    Pl.prob.b = x
    solve!(Pl.prob)
    y .= Pl.prob.u .- x # Note the -x (following Bardin et al)
    return y
end
M̄ = mean(Ms) #
Δt = sum(δt for _ in steps)

# Note: rtol can be set with MKLPardisoIterate (which probably opts in iterative refinements)
# as opposed to MKLPardisoFactorize for which rtol has no effect (no iterative refinements I guess)
# Plprob = init(Plprob, MKLPardisoFactorize(; nprocs = 48))
# Plprob = init(Plprob)
Plprob = LinearProblem(-Δt * M̄, ones(N))  # following Bardin et al. (M -> -M though)
Plprob = init(Plprob, MKLPardisoIterate(; nprocs = 48), rtol = 1e-10)
Pl = CycloPreconditioner(Plprob)
Pr = I
precs = Returns((Pl, Pr))

@time "initial state solve" u0 = solve(LinearProblem(M̄, ones(N)), MKLPardisoIterate(; nprocs = 48), rtol = 1e-10).u
@show norm(M̄ * u0 - ones(N)) / norm(ones(N))

function initstepprob(A)
    prob = LinearProblem(A, δt * ones(N))
    return init(prob, MKLPardisoIterate(; nprocs = 48), rtol = 1e-10)
    # return init(prob, MKLPardisoFactorize(; nprocs = 48))
    # return init(prob)
end

p = (;
    δt,
    stepprob = [initstepprob(I + δt * M) for M in Ms]
)
function mystep!(du, u, p, m)
    prob = p.stepprob[m]
    prob.b = u .+ p.δt # xₘ₊₁ = Aₘ₊₁⁻¹ (xₘ + δt 1) # CHECK m index is not off by 1
    du .= solve!(prob).u
    return du
end
function jvpstep!(dv, v, p, m)
    prob = p.stepprob[m]
    prob.b = v # xₘ₊₁ = Aₘ₊₁⁻¹ (xₘ + δt 1) # CHECK m index is not off by 1
    dv .= solve!(prob).u
    return dv
end
# function mystep(u, p, m)
#     # @info "  +1 season"
#     prob = p.stepprob[m]
#     prob.b = u .+ p.δt # xₘ₊₁ = Aₘ₊₁⁻¹ (xₘ + δt 1) # CHECK m index is not off by 1
#     solve!(prob)
#     out = deepcopy(prob.u)
#     return out
# end
# function jvpstep(v, p, m)
#     # @info "  +1 season"
#     prob = p.stepprob[m]
#     prob.b = v # J * v = Aₘ₊₁⁻¹ v # CHECK m index is not off by 1
#     solve!(prob)
#     out = deepcopy(prob.u)
#     return out
# end
function steponeyear!(du, u, p)
    du .= u
    for m in eachindex(p.stepprob)
        mystep!(du, du, p, m)
    end
    return du
end
function jvponeyear!(dv, v, p)
    dv .= v
    for m in eachindex(p.stepprob)
        jvpstep!(dv, dv, p, m)
    end
    return dv
end
# function steponeyear(u, p)
#     # @info "+1yr     $(u[end])"
#     um = deepcopy(u)
#     for m in eachindex(steps)
#         um = mystep(um, p, m)
#     end
#     return um
# end
# function jvponeyear(v, p)
#     vm = deepcopy(v)
#     for m in eachindex(steps)
#         vm = jvpstep(vm, p, m)
#     end
#     return vm
# end
function G!(du, u, p)
    steponeyear!(du, u, p)
    du .-= u
    return du
end
# G(u, p) = steponeyear(u, p) - u

# Function for Jacobian–vector product, J * x.
# Since G(x) = J x + b
# function jvp!(Jv, v, u, p)
#     b = G!(Jv, zero(u), p) # b <- J b
#     G!(Jv, v, p) # Jv <- J v + b
#     Jv .-= b # Jv <- Jv - b = J v + b - (J u + b) = J v
#     return Jv
# end
# jvp(v, u, p) = G(v, p) - G(zero(v), p)
function jvp!(dv, v, u, p)
    jvponeyear!(dv, v, p)
    dv .-= v
    return dv
end


# f = NonlinearFunction(G; jvp = jvp)
f! = NonlinearFunction(G!; jvp = jvp!)

# nonlinearprob = NonlinearProblem(f, u0, p)
nonlinearprob! = NonlinearProblem(f!, u0, p)


# w = (1 .- issrf) .* v
# internalnorm(u, t) = sqrt(u' * Diagonal(w) * u / sum(w))
# # internalnorm(u, t) = maximum(abs, u[.!issrf])
# # u0 = zeros(size(u0))

@info "solve seasonal steady state"
# @time sol = solve(nonlinearprob, NewtonRaphson(linsolve = KrylovJL_GMRES(precs = precs)), verbose = true, reltol=1e-10, abstol=Inf);
@time sol! = solve(nonlinearprob!, NewtonRaphson(linsolve = KrylovJL_GMRES(precs = precs, rtol=1e-12)); show_trace = Val(true), reltol=Inf, abstol=1e-10norm(u0, Inf));


du = deepcopy(u0)
norm(G!(du, sol!.u, p), Inf) / norm(sol!.u, Inf) |> u"permille"