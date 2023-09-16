using Plots
using ApproxFun
using ApproxFunNewton
using LinearAlgebra


R = 1e3 # Reynolds number
Pₑ = 0.7*R # Peclet number

d = Chebyshev(0..1)

L = (R,f,h,θ,A) -> [f(0); f(1)-1;
                    f'(0); f'(1);
                    h(0); h(1);
                    θ(0); θ(1)-1;
                    f'''-R*((f')^2-f*f''-A);
                    h''+R*f*h'+1;
                    θ''+0.7*R*f*θ';
                    A']

x = Fun(d)
f0 = -2x^3+3x^2; h0 = 0*x; θ0 = -(x-1)^2+1; A0 = 0*x

# SOLVE FOR R=1e3

@time f, h, θ, A = ApproxFunNewton.newton((f,h,θ,A) -> L(R,f,h,θ,A), [f0,h0,θ0,A0], θ̄=1e15, damped=false, verbose=true)
u = [f,h,θ,A]

plts1 = []
for (i,fun) in enumerate(["f(x)", "h(x)", "θ(x)", "A"])
    plt = plot(u[i], xlabel="x", ylabel=fun, legend=false)
    push!(plts1, plt)
end
plt1 = plot(plts1..., layout=4, size=(800,800))

savefig(plt1, "graphics/experiment_3_1_fluidinjection_1000.pdf")

# REPEAT FOR R=1e4

R = 1e4 # Reynolds number
Pₑ = 0.7*R # Peclet number

@time f, h, θ, A = ApproxFunNewton.newton((f,h,θ,A) -> L(R,f,h,θ,A), [f0,h0,θ0,A0], θ̄=1e15, damped=false, verbose=true)
u = [f,h,θ,A]

plts2 = []
for (i,fun) in enumerate(["f(x)", "h(x)", "θ(x)"])
    plt = plot(u[i], xlabel="x", ylabel=fun, legend=false)
    push!(plts2, plt)
end
plt2 = plot(plts2..., layout=(1,3), size=(900,275), margin=5Plots.mm)

savefig(plt2, "graphics/experiment_3_1_fluidinjection_10000.pdf")
