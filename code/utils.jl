using LinearAlgebra
using StatsBase


function ell(A, i; λ=0)
    ai = A[:, i]
    return ai' * pinv(A * A' + λ^2 * I) * ai
end


function ridge_weights(A, λ)
    _, n = size(A)
    weights = zeros(n)
    for i = 1:n
        weights[i] = ell(A, i, λ=λ)
    end
    return weights
end


function uniform_weights(A)
    _, n = size(A)
    return ones(n) / n
end


function squared_norms_weights(A)
    _, n = size(A)
    return dropdims(sum(A .* A, dims=1), dims=1)
end


function samplemat(A, weights, k)
    columns = sample(1:n, Weights(weights), k)
    return A[:, columns]
end


function ridge_sampling(A, k, λ)
    return samplemat(A, ridge_weights(A, λ), k)
end


function uniform_sampling(A, k)
    return samplemat(A, uniform_weights(A), k)
end


function squared_norms_sampling(A, k)
    return samplemat(A, squared_norms_weights(A), k)
end


function hilbert(n)
    H = zeros(n, n)
    for i = 1:n
        for j = 1:n
            H[i, j] = 1 / (i + j - 1)
        end
    end
    return H
end


function approxerror(A, C)
    return norm(A - C * pinv(C) * A)
end
