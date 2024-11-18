using Pkg
Pkg.activate(".")
Pkg.instantiate()

using AIBECS
using LinearSolve
import Pardiso # that's what they do in LinearSolve.jl benchmarks (maybe to avoid name clash)
using SparseArrays
using LinearAlgebra
using Unitful
using Unitful: m, s, yr, d
using BlockArrays

# Define the model
# grd, T = Primeau_2x2x2.load()
# grd, T = Archer_etal_2000.load()
grd, T = OCCA.load()
# grd, T = OCIM0.load() # <- too big to GMRES-solve in 1hr it seems
# grd, T = OCIM2_48L.load()
N = size(T, 1)

v = grd.volume_3D[grd.wet3D];

issrf = let
    issrf3D = zeros(size(grd.wet3D))
    issrf3D[:,:,1] .= 1
    issrf3D[grd.wet3D]
end
M = sparse(Diagonal(issrf))

# Now let's imagine we have 4 seasons of OCCA where T gets more or less intense
seasons = ("DJF", "MAM", "JJA", "SON")
αs = (1.0, 0.95, 1.0, 1.05)
# Preconditioner for cyclostationary solver
struct CycloPreconditioner
    DJF # Pardiso factors for DJF
    MAM # Pardiso factors for MAM
    JJA # Pardiso factors for JJA
    SON # Pardiso factors for SON
end
Nseasons = length(seasons)
δt = ustrip(s, 365d / Nseasons) # TODO maybe use exact mean number of days (more important for monthly because Feb)?

# Build matrix to multiply
A = BlockArray(spzeros(Nseasons * N, Nseasons * N), fill(N, Nseasons), fill(N, Nseasons))
@time "building blocks" for (i, (α, season)) in enumerate(zip(αs, seasons))
    A[Block(i, i)] = I + δt * (α * T + M)
    A[Block(mod1(i+1, Nseasons), i)] = -I(N)
end

function initseasonlinprob(A, i)
    prob = LinearProblem(A[Block(i,i)], δt * ones(N))
    return init(prob, MKLPardisoFactorize(; nprocs = 48))
end
Pl = CycloPreconditioner(
    initseasonlinprob(A, 1),
    initseasonlinprob(A, 2),
    initseasonlinprob(A, 3),
    initseasonlinprob(A, 4),
)


# @time "converting BlockArray to standard sparse" A = sparse(A)
function initseasonlinprob(A, i)
    prob = LinearProblem(A[Block(i,i)], δt * ones(N))
    return init(prob, MKLPardisoFactorize(; nprocs = 48))
end
Pl = CycloPreconditioner(
    initseasonlinprob(A, 1),
    initseasonlinprob(A, 2),
    initseasonlinprob(A, 3),
    initseasonlinprob(A, 4),
)


Base.eltype(::CycloPreconditioner) = Float64
function LinearAlgebra.ldiv!(Pl::CycloPreconditioner, x::AbstractVector)
    # Making a blocked view into x to help with blocks
    xᵇ = BlockedVector(x, fill(N, Nseasons))
    for (i, block) in enumerate(propertynames(Pl))
        # Grab the $block seasonal linear problem
        linprob = getproperty(Pl, block)
        # update the RHS
        if i == 1 # First season is independent of other seasons
            linprob.b = xᵇ[Block(i)]
        else # Other seasons depend on the previous one
            linprob.b = xᵇ[Block(i)] + xᵇ[Block(i - 1)]
        end
        # linprob.b = xᵇ[Block(i)]
        # @time "solve $block" solve!(linprob)
        solve!(linprob)
        # Update input vector
        xᵇ[Block(i)] .= linprob.u
    end
    return x
end
function LinearAlgebra.ldiv!(y::AbstractVector, Pl::CycloPreconditioner, x::AbstractVector)
    # Making blocked views into x and y to help with blocks
    xᵇ = BlockedVector(x, fill(N, Nseasons)) # a blocked view into x to help with blocks
    yᵇ = BlockedVector(y, fill(N, Nseasons)) # a blocked view into x to help with blocks
    for (i, block) in enumerate(propertynames(Pl))
        # Grab the $block seasonal linear problem
        linprob = getproperty(Pl, block)
        # update the RHS
        if i == 1 # First season is independent of other seasons
            linprob.b = xᵇ[Block(i)]
        else # Other seasons depend on the previous one
            linprob.b = xᵇ[Block(i)] + yᵇ[Block(i - 1)]
        end
        # linprob.b = xᵇ[Block(i)]
        # @time "solve $block" solve!(linprob)
        solve!(linprob)
        # Update input vector
        yᵇ[Block(i)] .= linprob.u
    end
    return y
end

@info "solve non seasonal steady state"
prob0 = LinearProblem(T + M, ones(N))
@time "solve steady state" u0 = solve(prob0, MKLPardisoFactorize(; nprocs = 48)).u

@time "converting BlockArray to standard sparse" A = reduce(hcat, reduce(vcat, A[Block(i, j)] for i in 1:blocksize(A, 1)) for j in 1:blocksize(A, 2))
b = δt * ones(N * Nseasons)

@info "Setting up full seasonal problem"
prob = LinearProblem(A, b; u0 = repeat(u0, outer = Nseasons))

@time "initialize full problem" linsolve = init(prob, KrylovJL_GMRES(gmres_restart = 200), Pl = I)

@info "Now attempting seasonal solve"

@time "solve" solve!(linsolve)




@time "Direct solve" linsolve2 = solve(prob, MKLPardisoFactorize(; nprocs = 48))

@show norm(linsolve.u - linsolve2.u) / norm(linsolve.u)
