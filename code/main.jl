include("utils.jl")

using Plots
using LaTeXStrings
using Statistics


n = 100
maxk = 50
repeats = 20

A = hilbert(n)
ks = 1:maxk
λ = 1e-4

errors = zeros(maxk, repeats, 3)

methods = [
    ridge_weights(A, λ),
    uniform_weights(A),
    squared_norms_weights(A)
]

for k in ks
    for rep = 1:repeats
        for (i, w) in enumerate(methods)
            C = samplemat(A, w, k)
            C = unique(C, dims=2)
            errors[k, rep, i] = approxerror(A, C)
        end
    end
end

best_approx_errors = zeros(maxk)

for k in ks
    C = best_approx(A, k)
    best_approx_errors[k] = approxerror(A, C)
end


means = mean(errors, dims=2)[:, 1, :]
stds = std(errors, dims=2)[:, 1, :]

plot(xlabel=L"k", ylabel=L"\textrm{error}")
plot!(ks, means[:, 2], yaxis=:log, label="uniform", w=2)
plot!(ks, means[:, 3], yaxis=:log, label="squared norm", w=2)
plot!(ks, means[:, 1], yaxis=:log, label="ridge", w=2)
p = plot!(ks, best_approx_errors, yaxis=:log, label="best", c=:gray, w=2, ls=:dashdot)
savefig(p, "./plot.pdf")
display(p)  # sometimes the display doesn't work