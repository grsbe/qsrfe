using LinearAlgebra
using Random, Distributions
using Convex, SCS
using RDatasets
using MLJ
using DataFrames

Random.seed!(42) 


#another dataset
bX, bY = @load_iris
bX = Matrix(DataFrame(bX))
bY = collect(bY)
bX

bX = (bX .- mean(bX)) ./ std(bX)
bY = (bY .- mean(bY)) ./ std(bY)
(Xtrain, Xtest), (ytrain, ytest) = partition((bX, bY), 0.9, rng=123, multi=true)

####################
#weight init
σ2 = 1
N = 500 #number of weights
#normal weights
ω = rand(Normal(0.0,σ2),(N,d))

#sparse weights
q = 2
ωs = zeros(Float64,(N,d))
for i in 1:N
    ωs[i, sample(1:d,q, replace=false)] = rand(Normal(0.0,σ2),q);
end
ωs
ζ = rand(Uniform(0.0,2π),N)

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
function fit_srfe(X,Y,η,N ;σ2 = 1, q=0, quantization=0, K=10, pruning=1.0)
    #generate weights ω (and ζ)
    #normal weights
    #if stable
    m,d = size(X)
    ζ = rand(Uniform(0.0,2π),N)
    #end
    if q >= size(X)[2] || q <= 0
        ω = rand(Normal(0.0,σ2),(N,d))
    else
        #sparse weights
        ω = zeros(Float64,(N,d))
        for i in 1:N
            ω[i, sample(1:d,q, replace=false)] = rand(Normal(0.0,σ2),q)
        end
    end
    
    #A = compute_featuremap(X,ω)
    A = compute_stable_featuremap(X,ω,ζ)
    #quantization
    if quantization == 1
        A = quantize_msq(A,K)
    end
    if quantization == 2
        A = ΣΔ_quantization(A,K)
    end
    c = basispursuit(A,Y,η)
    #prune!(c,pruning)
    return c, ω, ζ
end




######################################################
# main computations
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

##########################################################
#constants
η = 0.01
N= 400
c, ω, ζ = fit_srfe(Xtrain,ytrain,η,N;σ2=1,q=0, quantization=0,K=1)

y_pred = compute_stable_featuremap(Xtest,ω, ζ) * c
rel_error(ytest,y_pred)

prune!(c,0.1)

using Plots
plot(c)
plot(abs.(y_pred - ytest))

plot(ytest)
plot!(y_pred)
c
y_pred
ytest