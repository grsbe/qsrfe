using LinearAlgebra
using Random, Distributions
using Convex, SCS
using RDatasets



Random.seed!(123) 


#datasets
d = 10 # dimension of feature vectors
m = 50 #number of features in dataset

γ = 1 #x spread

X = rand(Normal(0.0,γ),(m,d))
f = function(x) 
    s = 0
    for i in 1:(length(x)-1)
        s += exp(-x[i]^2) / x[i+1]^2 + 1
    end
    return 1/length(x) .* s
end
Y = [f(X[i,:]) for i in 1:m]


Xtest = rand(Normal(0.0,γ),(m,d))
Ytest = [f(Xtest[i,:]) for i in 1:m]
#another dataset
bX, bY = @load_boston
bX = Matrix(bX)
bY = collect(bY)


####################
#weight init
σ2 = 1
N = 500 #number of weights
ω = rand(Normal(0.0,σ2),(N,d))

# compute the kernel 
#Todo: time measurements
compute_kernel = function(ω,x)
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

#basis pursuit
η = 0.01
function basispursuit(A,y,η)
    m, N = size(A)
    c = Variable(N)
    p = minimize(norm(c,1), norm(A * c - y) <= η)
    solve!(p, SCS.Optimizer; silent_solver = false)
    print(p.status)
    return evaluate(c)
end

function prune(c, s)
    pass
end


#error calc
rel_error = function(y_truth, y_pred)
    norm((y_truth - y_pred) ./ y_truth)
end

######################################################

A = compute_kernel(ω,X)
c = basispursuit(A,Y,η)

using Plots
plot(c)

y_pred = compute_kernel(ω,Xtest) * c
rel_error(Ytest,y_pred)

plot(y_pred - Ytest)
