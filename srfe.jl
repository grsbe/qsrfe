using LinearAlgebra
using Random, Distributions
using Convex, SCS
using RDatasets
using MLJ
using DataFrames
using MLJLinearModels

Random.seed!(42) 


# compute the kernel <- is not a kernel ;-; its a feature matrix
function compute_featuremap(x,ω)
    m, d1 = size(x)
    N, d2 = size(ω)
    if d1 != d2 
        throw(ArgumentError("mismatching dimensions"))
    end
    A = zeros(Float64, m,N)
    for i in 1:m
        for j in 1:N
            A[i,j] = cos(x[i,:] ⋅ ω[j,:])
        end
    end
    return A
end

function compute_stable_featuremap(x,ω,ζ)
    m, d1 = size(x)
    N, d2 = size(ω)
    if d1 != d2 
        error("mismatching dimensions")
    end
    A = zeros(Float64, m,N)
    for i in 1:m
        for j in 1:N
            A[i,j] = cos(x[i,:] ⋅ ω[j,:] + ζ[j])
        end
    end
    return A
end

function compute_featuremap(x,ω,func)
    m, d1 = size(x)
    N, d2 = size(ω)
    if d1 != d2 
        throw(ArgumentError("mismatching dimensions"))
    end
    A = zeros(Float64, m,N)
    for i in 1:m
        for j in 1:N
            A[i,j] = func(x[i,:] ⋅ ω[j,:])
        end
    end
    return A
end

#feature maps
function rff(x)
    return cos(x)
end

function ReLU(x)
    return max(0,x)
end

#quantization schemes
function quantize_msq(A,K)
    K = convert(Float64,K)
    m, N = size(A)
    q = zeros(Float64,(m,N))
    function rounding_quantizer(a,K)
        return round(a * (2K - 1)) / (2K - 1)
    end
    for i in 1:m
        for j in 1:N
            q[i,j] = rounding_quantizer(A[i,j],K)
        end
    end
    return q
end

function ΣΔ_quantization(A,K)
    m, N = size(A)
    K = convert(Float64,K)
    function rounding_quantizer(a,K)
        return round(a * (2K - 1)) / (2K - 1)
    end
    q = zeros(Float64,(m,N))
    for i in 1:m
        #u = zeros(Float64,N+1) #u[i+1] = u_i
        u = 0.0
        for j in 1:N
            q[i,j] = rounding_quantizer(A[i,j] + u,K)
            u = u + A[i,j] - q[i,j]
        end
    end
    return q
end

function distr_ΣΔ_quantization(A,K,β,λ)
    m, N = size(A)
    K = convert(Float64,K)
    function rounding_quantizer(a,K)
        return round(a * (2K - 1)) / (2K - 1)
    end
    for i in 1:m
        u = zeros(Float64,N+1) #u[i+1] = u_i
        q = zeros(Float64,N)
        for j in 1:N
            q[j] = rounding_quantizer(A[i,j] + u[j],K)
            u[j+1] = u[j]+A[i,j]-q[j]
        end
    end
end

#basis pursuit

function basispursuit(A,y,η)
    m, N = size(A)
    c = Variable(N)
    p = minimize(norm(c,1), norm(A * c - y) <= η)
    solve!(p, SCS.Optimizer; silent_solver = true)
    #print(p.status)
    return Convex.evaluate(c)
end

#Lasso




function prune!(c, s=0.2)
    # s gives the top percentage sparsity
    # s = 0.2 -> only keep the top 20% of non zero values
    if s >= 1 || s < 0
        return c
    end
    s = round(Int, min(length(c),length(c) * s))
    indexperm = partialsortperm(abs.(c),1:s, rev=true) #max-s indices
    c[1:length(c) .∉ Ref(indexperm)] .= 0.0 #turn all indices not in indexperm to 0.0
    return c
end


#error calc
rel_error = function(y_truth, y_pred)
    norm((y_truth - y_pred) ./ y_truth)
end


######################################################
# putting everything together
function fit_srfe(X,Y,λ,N,func ;σ2 = 1, q=0, quantization=0, K=10, pruning=1.0)
    #generate weights ω (and ζ)
    #normal weights
    m,d = size(X)
    #ζ = rand(Uniform(0.0,2π),N)
    println("generate weights")
    if q >= size(X)[2] || q <= 0
        ω = rand(Normal(0.0,σ2),(N,d))
    else
        #sparse weights
        ω = zeros(Float64,(N,d))
        for i in 1:N
            ω[i, sample(1:d,q, replace=false)] = rand(Normal(0.0,σ2),q)
        end
    end
    
    println("compute features")
    #A = compute_featuremap(X,ω)
    A = compute_featuremap(X,ω,func)
    #quantization
    if quantization == 1
        println("quantizing")
        A = quantize_msq(A,K)
    end
    if quantization == 2
        println("quantizing")
        A = ΣΔ_quantization(A,K)
    end
    println("lasso")
    #c = basispursuit(A,Y,η)
    lasso = LassoRegression(λ; fit_intercept=false)
    c = MLJLinearModels.fit(lasso,A,Y)
    #prune!(c,pruning)
    return c, ω
end




######################################################
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



