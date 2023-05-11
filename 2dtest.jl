include("src/qsrfe.jl")
using .qsrfe



#point cloud
using Distributions, Random
using LinearAlgebra
n, m = 20, 50

function ontoring(data, radius)
    da = zeros(size(data))
    for i in 1:size(data,1)
        da[i,:] = data[i,:] ./ norm(data[i,:]) .* radius
        println("Data: ", data[i,:], " Da: ", da[i,:])
    end
    return da
end

data1 = rand(Normal(0.,0.5),(n,2))
data2 = ontoring(rand(Normal(0.,0.5),(m,2)),4) .+ rand(Normal(0.,0.5),(m,2))



xtrain = vcat(data1,data2)
ytrain = [ones(Float64,n);zeros(Float64,m)]

scatter(data1[:,1],data1[:,2])
scatter!(data2[:,1],data2[:,2])


