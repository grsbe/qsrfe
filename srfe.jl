module srfe

export fit_srfe, prune!, ReLU, rff, rel_error, compute_featuremap, MSQ, βQ, ΣΔQ

using LinearAlgebra
using Random, Distributions
using MLJ
using MLJLinearModels


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

function complex(x)
    cis(x)
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

function ΣΔQ(A,K,r=1,limit=1,λ =0)
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

    if λ != 0
        if N % λ != 0
            error("choose number of weights to be divisible by λ")
        end
        p = N / λ
        if r == 1
            V = kron(diagm(ones(p)),ones(λ))
            q = V * transpose(q) * sqrt(2/p) / norm(ones(λ))
            q = transpose(q)
        elseif r==2 
            if λ % 2 == 0
                error("hat_λ not an integer, choose uneven λ")
            end
            hat_λ = convert(Int64,(λ + 1) / 2)
            v= [1:hat_λ;(hat_λ-1):-1:1]
            V = kron(diagm(ones(p)),v)
            q = V * transpose(q) * sqrt(2/p) / norm(v)
            q = transpose(q)
        end
        
    end

    return q
end

using ToeplitzMatrices
using FFTW

function βQ(A, β, λ, K,limit=1,condensation=true)
    m, N = size(A)
    Δ = stepsize(K)
    if N % λ != 0
        error("choose number of weights to be divisible by λ")
    end
    p = N / λ
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

    if condensation
        v= [β^(-i) for i in 1:λ]
        V = kron(diagm(ones(p)),v)
        q = V * transpose(q) * sqrt(2/p) / norm(v)
        q = transpose(q)
    end
    
    return q
end

#basis pursuit


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
    mean(abs.((y_truth - y_pred) ./ y_truth))
end

#generate weights
function gen_weights(N,d,q=0,σ2=1)
    #generate weights ω (and ζ)
    #normal weights
    
    ζ = rand(Uniform(0.0,2π),N)
    #println("generate weights")
    if q >= d || q <= 0
        ω = rand(Normal(0.0,σ2),(N,d))
    else
        #sparse weights
        ω = zeros(Float64,(N,d))
        for i in 1:N
            ω[i, sample(1:d,q, replace=false)] = rand(Normal(0.0,σ2),q)
        end
    end
    return ω, ζ
end

######################################################
# putting everything together
function fit_srfe(X,Y,lasso_lambda,N,func ;σ2 = 1, q=0, quantization=0, K=10,r=1,β=1.1,λ=1, limit=1,pruning=1.0, max_iter=1000)
    #weights
    m,d = size(X)
    ω, ζ = gen_weights(N,d,q,σ2)
    #println("compute features")
    A = compute_featuremap(X,ω, func,ζ)

    #quantization
    if quantization == 1
        #println("quantizing")
        A = MSQ(A,K,limit)
    end
    if quantization == 2
        #println("quantizing")
        A = ΣΔQ(A,K,r,limit,λ)
    end
    if quantization == 3
        #println("quantizing")
        A = βQ(A,β,λ,K,limit)
    end
    #println("lasso")
    #c = basispursuit(A,Y,η)
    lasso = LassoRegression(lasso_lambda; fit_intercept=false)
    solver = FISTA(max_iter=max_iter)
    c = MLJLinearModels.fit(lasso,A,Y;solver)
    #prune!(c,pruning)
    return c, ω ,ζ
end

end #end module



