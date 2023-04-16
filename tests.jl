include("srfe.jl")
using .srfe


#datasets
d = 10 # dimension of feature vectors
m = 50 #number of features in dataset

γ = 1 #x spread

X = rand(Normal(0.0,γ),(m,d))
f = function(x) 
    s = 0
    for i in 1:(length(x)-1)
        s += exp(-x[i]^2) / (x[i+1]^2 + 1)
    end
    return 1/length(x) .* s
end
Y = [f(X[i,:]) for i in 1:m]


Xtest = rand(Normal(0.0,γ),(m,d))
Ytest = [f(Xtest[i,:]) for i in 1:m]

#another dataset
bX, bY = @load_boston
bX = Matrix(DataFrame(bX))
bY = collect(bY)
bX

bX = (bX .- mean(bX)) ./ std(bX)
#bY = (bY .- mean(bY)) ./ std(bY)
(Xtrain, Xtest), (ytrain, ytest) = partition((bX, bY), 0.9, rng=123, multi=true)

##########################################################
#constants and computation
λ = 0.1
N= 20000
func = rff
c, ω = fit_srfe(Xtrain,ytrain,λ,N,func;σ2=1,q=0, quantization=0,K=1)

y_pred = compute_featuremap(Xtest,ω,func) * c
rel_error(ytest,y_pred)
mean(abs.(ytest-y_pred))

using Plots
plot(c)
prune!(c,0.01)
plot(abs.(y_pred - ytest))

plot(ytest)
plot!(y_pred)

#standard lasso fit
λ = 0.02
lasso = LassoRegression(λ; fit_intercept=true)
theta = MLJLinearModels.fit(lasso,Xtrain,ytrain)
plot(theta)
y_lasso = hcat( Xtest, ones(size(Xtest)[1])) * theta
rel_error(ytest,y_lasso)
mean(abs.(ytest-y_lasso))
plot!(y_lasso)

#