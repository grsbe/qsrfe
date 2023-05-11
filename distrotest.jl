using Plots
using Distributions
using Random
using LinearAlgebra

r = Normal(0.0, 1)
om = Normal(0.0, 1)

trials = 1000000
dim = 10
data = zeros(trials)
for i in 1:trials
    x = rand(r,dim)
    ω = rand(om,dim)
    tau = rand(Uniform(-π,π))
    data[i] = x ⋅ ω + tau
end

histogram(data, bins=100)