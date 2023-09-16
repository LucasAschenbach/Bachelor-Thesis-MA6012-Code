using Plots
using LaTeXStrings
using Random
using LinearAlgebra
using BandedMatrices
using SpecialFunctions

truncation(A::AbstractArray{T,2}, cols::AbstractArray{Int,1}) where T = size(A,1)
truncation(A::AbstractArray{T,2}, col::Int) where T = truncation(A, [col])
truncation(A::AbstractArray{T,2}) where T = truncation(A, 1:size(A,2))
# BandedMatrix
truncation(A::BandedMatrix{T}, cols::AbstractArray{Int,1}) where T = bandwidth(A,1) + max(cols...) - 1
truncation(A::BandedMatrix{T}, col::Int) where T = truncation(A, [col])

function solve(A::AbstractArray{T,2}, b::AbstractArray{T,1}; batch::Int=1, tol=eps()) where T
    solve!(copy(A), copy(b), batch=batch, tol=tol)
end

function solve!(A::AbstractArray{T,2}, b::AbstractArray{T,1}; batch::Int=1, tol=eps()) where T
    err = Inf
    k = 1 - batch
    factors = []
    errs = []
    steps = []
    # QR factorization
    while err > tol && k < size(A,2)
        k += batch
        # Determine crop for batch
        jbatch = k:min(k+batch-1, size(A,2)) # columns
        tr = truncation(A, jbatch)
        ibatch = k:min(tr, size(A, 1)) # rows
        # Apply previous Qs for current batch columns
        for (index,Ql) in enumerate(factors)
            l = (index-1)*batch+1
            iprevbatch = l:l+size(Ql,1)-1
            A[iprevbatch,jbatch] = Ql'*A[iprevbatch,jbatch]
        end
        # Local QR factorization
        F = qr!(view(A,ibatch,jbatch))
        push!(factors, F.Q)
        # Update RHS
        b[ibatch] = F.Q'*b[ibatch]
        # Compute error
        err = norm(b[k+1:end])
        push!(errs, err)
        push!(steps, UpperTriangular(A[1:k,1:k])\b[1:k])
    end
    # Backwards substitution
    k = min(k, size(A,2)) # k_opt
    UpperTriangular(A[1:k,1:k])\b[1:k], errs, steps
end

Random.seed!(1234)

n = 10_000
R(x) = BandedMatrix(1 => ones(n-1), 0 => -2/x .* (1:n), -1 => ones(n-1))
c0 = [1.0; (n->iseven(n) ? 2.0 : 0.0).(1:n-1)]'
Bessel(x) = [c0; R(x)[2:n,:]]
b = [1.0; zeros(n-1)]

A = Bessel(100)

x = A\b
η = 1/min(svd(A[1:1000,1:1000]).S...)

@time x̂, errs, steps = solve(A, b, batch=1)

isapprox(x,[x̂; zeros(n-length(x̂))])

plt = plot([eps() for i in 1:length(errs)], yscale=:log10, yticks=[1e5, 1e0, 1e-5, 1e-10, η*eps(),eps()], label=false, linestyle=:dash, color=:black, xlabel="Truncation", ylabel="Error", yaxis=(formatter = y -> 10^round(log10(y), digits=2)))
plot!(plt, [η .* eps() for i in 1:length(errs)], label=false, linestyle=:dash, color=:black)
plot!(plt, η .* errs, label="Bound", color=:black)
plot!(plt, errs, label="Estimate", color=:black, linestyle=:dash)
scatter!(plt, [norm(step - x[1:length(step)]) for step in steps], label="Error", color=:black, markersize=2)

savefig(plt, "graphics/experiment_1_bessel.pdf")
