using Plots
using LaTeXStrings
using LinearAlgebra
using ApproxFun
using ApproxFunNewton

# Domain transformed to canonical Chebyshev interval due to missing setdomain
# implementation for PiecewiseSpace
d = Chebyshev()
Ψ(x) = π/2*x # [-1..1] -> [-π/2..π/2]
Ψ′ = π/2
ϵ = 0.1

x = Fun(d)
u1_initial = 1. + 0*x
u2_initial = 1. + 0*x

N(ϵ,u1,u2,x=Fun(d)) = [
    u1(-1)-1.;
    u1(1)-1.;
    u2';
    Ψ′ * (sin(Ψ(x))^2 - u2*sin(Ψ(x))^4/u1) - ϵ*u1';
]

@time u1, u2 = ApproxFunNewton.newton((u1,u2,x=Fun(d)) -> N(1.,u1,u2,x), [u1_initial, u2_initial], damped=false, θ̄=2, maxiterations=100, verbose=true)
@time vlist = ApproxFunNewton.continuation((λ,u1,u2,x=Fun(d)) -> N(λ*ϵ+(1-λ),u1,u2,x), [u1, u2], s0=0.1, verbose=true)

vlist = [setdomain(v, -π/2..π/2) for v in vlist]

λ = vlist[end].u[2](0)
plot(vlist, vars=Tuple(1), xlabel="x", ylabel="y(x)", legend=false)

savefig("graphics/experiment_3_4_lubrication.pdf")
