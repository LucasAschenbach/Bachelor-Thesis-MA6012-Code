using Plots
using LinearAlgebra
using ApproxFun
using ApproxFunNewton

d = Chebyshev(0..1)
x = Fun(d)
u1_initial = 2*sin(2π*x)
u2_initial = 1+cos(2π*x)
T_initial = 2π + 0.0*x

N = (u1,u2,T) -> [
    u1(0);
    u1(1);
    u2(0) - u2(1);
    u1' - 3T*(u1 + u2 - u1^3/3 - 1.3);
    u2' + T/3*(u1 + 0.8*u2 - 0.7);
    T'
]

# Converges immediately but expensive iterations, hence omit monotonicity check
@time u1, u2, T = ApproxFunNewton.newton(N, [u1_initial, u2_initial, T_initial], θ̄=Inf, verbose=true)
u = [u1,u2]

plts = []
for (i,fun) in enumerate(["voltage", "permeability"])
    plt = plot(u[i], xlabel="t", ylabel=fun, legend=false)
    push!(plts, plt)
end
plot(plts..., layout=2, size=(800,400), margin=5Plots.mm)

savefig("graphics/experiment_3_2_nervepulse.pdf")

# Repeat with monotonicity check
@time u1, u2, T = ApproxFunNewton.newton(N, [u1_initial, u2_initial, T_initial], θ̄=1e15, verbose=true)
u = [u1,u2]
