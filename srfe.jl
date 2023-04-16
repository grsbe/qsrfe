module srfe

export fit_srfe, prune!, ReLU, rff, rel_error, compute_featuremap, MSQ, βQ, ΣΔQ

using LinearAlgebra
using Random, Distributions
using MLJ
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


function compute_featuremap(x,ω,func)
    m, d1 = size(x)
    N, d2 = size(ω)
    if d1 != d2 
        error("mismatching dimensions")
    end
    A = zeros(Float64, m,N)
    for i in 1:m
        for j in 1:N
            A[i,j] = func(x[i,:] ⋅ ω[j,:])
        end
    end
    return A
end

function compute_featuremap(x,ω,func,ζ)
    m, d1 = size(x)
    N, d2 = size(ω)
    if d1 != d2 
        error("mismatching dimensions")
    end
    A = zeros(Float64, m,N)
    for i in 1:m
        for j in 1:N
            A[i,j] = func(x[i,:] ⋅ ω[j,:] + ζ[j])
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
function rounding_quantizer(a,Δ,limit)
    if abs(a) <= limit
        return floor(a / Δ) * Δ + Δ / 2
    else
        return sign(a) * limit
    end
end

function stepsize(K)
    return K = 1 / (K - 1/2)
end

function MSQ(A,K,limit=1)
    m, N = size(A)
    Δ = stepsize(K)
    q = zeros(Float64,(m,N))
    for i in 1:m
        for j in 1:N
            q[i,j] = rounding_quantizer(A[i,j],Δ,limit)
        end
    end
    return q
end

function ΣΔQ(A,K,r=1,limit=1)
    m, N = size(A)
    Δ = stepsize(K)
    q = zeros(Float64,(m,N))
    if r == 1
        u = zeros(Float64,N+1) #u[i+1] = u_i
        for i in 1:m
            for j in 1:N
                q[i,j] = rounding_quantizer(A[i,j] + u[j],Δ,limit)
                u[j+1] = u[j] + A[i,j] - q[i,j]
            end
        end
    elseif r == 2
        u = zeros(Float64,N+1) #u[i+1] = u_i
        for i in 1:m
            q[i,1] = rounding_quantizer(A[i,1],Δ,limit)
            u[2] = A[i,1] - q[i,1]
            for j in 2:N
                q[i,j] = rounding_quantizer(A[i,j] + 2u[j] - u[j-1],Δ,limit)
                u[j+1] = 2u[j] - u[j-1] + A[i,j] - q[i,j]
            end
        end
    else
        error("r only implemented for r = 1,2")
    end

    return q
end

using ToeplitzMatrices
using FFTW

function βQ(A, β, λ, K,limit=1)
    m, N = size(A)
    Δ = stepsize(K)
    p = round(Int64,N/λ)
    A = (A .* (2 * K - β)) ./(2 * K - 1)
    H_β = Toeplitz([[1,-β];zeros(Float64,λ-2)],[[1];zeros(Float64,λ-1)])
    H =  diagm(ones(N)) - kron(diagm(ones(p)),H_β)
    u = zeros(N+1) #u[i+1] = u_i
    q = zeros((m, N))
    for i in 1:m
        q[i,1] = rounding_quantizer(A[i,1],Δ,limit)
        u[2] = A[i,1] - q[i,1]
        for j in 2:N
            q[i,j] = rounding_quantizer(A[i,j]+H[j,j-1]*u[j],Δ,limit)
            u[j+1] = H[j,j-1] * u[j] + A[i,j]  - q[i,j]
        end
    end
    return q
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

#replace by Lasso


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
function rel_error(y_truth, y_pred)
    norm((y_truth - y_pred) ./ y_truth)
end


######################################################
# putting everything together
function fit_srfe(X,Y,lasso_lambda,N,func ;σ2 = 1, q=0, quantization=0, K=10,r=1,β=1.1,λ=2, limit=1,pruning=1.0)
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
    #A = compute_featuremap(X,ω, func,ζ)
    A = compute_featuremap(X,ω,func)
    #quantization
    if quantization == 1
        println("quantizing")
        A = MSQ(A,K,limit)
    end
    if quantization == 2
        println("quantizing")
        A = ΣΔQ(A,K,r,limit)
    end
    if quantization == 3
        println("quantizing")
        A = βQ(A,β,λ,K,limit)
    end
    println("lasso")
    #c = basispursuit(A,Y,η)
    lasso = LassoRegression(lasso_lambda; fit_intercept=false)
    c = MLJLinearModels.fit(lasso,A,Y)
    #prune!(c,pruning)
    return c, ω #,ζ
end

end #end module

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


using DataFrames
#boston dataset
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
N= 2000
func = rff
c, ω = fit_srfe(Xtrain,ytrain,λ,N,func;σ2=1,q=0, quantization=0,K=1,r=1)

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



