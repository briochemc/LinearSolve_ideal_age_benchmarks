# The age tracer equation is
#
#   ∂x(t)/∂t + T(t) x(t) = 1 - Ω x(t)
#
# where Ω "relaxes" x to zero in the top layer.
#
# Applying Backward Euler time step gives
#
#   (I + δt M) x(t+δt) = x(t) + δt
#
# where M = T + Ω.
#
# So maybe I can try to use Newton to solve G(x) = (I + δt M) \ (x + δt) - x(t) = 0
# (which is the same as solving δt - δt M x = 0    or   M x = 1)
# But this time I only coarsen M (not I).
#
# So instead of solving Mc xc = 1 where Mc = L M S and then spraying x = S xc,
# I use GMRES to solve (I + δt M) x = x0 + δt many times, preconditioned with
# y -> S (I + δt Mc)⁻¹ L y
# And wrap that into a nonlinear solver with the same preconditioner?
# (I + δt Mc) xc = xc0 + δt many times, and that should be fast as I + δt Mc is factorized only once.
#
# So maybe the first thing is to check that GMRES works for the linear system
# That is, solve (I + δt M) x = b
# And then check that GMRES(I + δt M,  b) with preconditioner P(y) = (I + δt S Mc⁻¹ L) y
# works too. My hope is that having the I not be part of the lumping/spraying will
# help.