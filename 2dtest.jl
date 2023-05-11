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
    end
    return da
end

data1 = rand(Normal(0.,0.5),(n,2))
data2 = ontoring(rand(Normal(0.,0.5),(m,2)),4) .+ rand(Normal(0.,0.5),(m,2))



xtrain = vcat(data1,data2)
ytrain = [ones(Float64,n);zeros(Float64,m)]



scatter(data1[:,1],data1[:,2])
scatter!(data2[:,1],data2[:,2])


srfe = qsrfe.srfeRegressor(;N=500, λ= 0.001, intercept=false)

c, ω, ζ = qsrfe.fit(srfe,xtrain,ytrain)


using Plots
plot(c)

#100 times 100 grid hack
x = range(-13, 13, length=100)
y = range(-13, 13, length=100)
z = [[i j] for i in x, j in y]
zz = zeros(Float64,(1,2))
for i in 1:100, j in 1:100
    zz = vcat(zz,z[i,j])
end
zz = zz[2:size(zz,1),:]

#prediction contour plot
zzz = qsrfe.predict(srfe,zz,c,ω,ζ)
zzz = reshape(zzz,(100,100))
contour(x,y,zzz)

scatter!(data1[:,1],data1[:,2], label="y = 1")
scatter!(data2[:,1],data2[:,2],label="y = 0")


###############################
# lasso path
using MLJ
using MLJLinearModels

λ = range(0.3,0.001,length=1000)

w = 500 #number of weights

c_collection = zeros(Float64,(1,w))

m,d = size(xtrain)
ω, ζ = qsrfe.gen_weights(w,d)
#println("compute features")
A = qsrfe.compute_featuremap(xtrain,ω, cos,ζ)
solver = FISTA(max_iter=20000)


for l in λ
    lasso = LassoRegression(l; fit_intercept=false)
    c = MLJLinearModels.fit(lasso,A,ytrain;solver)
    c_collection = vcat(c_collection,c')
end
c_collection = c_collection[2:size(c_collection,1),:]

plot(λ,c_collection;legend=false,xaxis=:log)

