
@with_kw struct srfeRegressor
    N::UInt128 = 1000
    λ::Real = 0.003
    q::UInt64 = 0
    quantizer::Quantizer = nothing
    func = rff
end

Normal()

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

# compute the feature matrix
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

######################################################
# putting everything together
function fit_srfe(X,Y,lasso_lambda,N,func ;σ2 = 1, q=0, quantizer=nothing, pruning=1.0, max_iter=1000)
    #weights
    m,d = size(X)
    ω, ζ = gen_weights(N,d,q,σ2)
    #println("compute features")
    A = compute_featuremap(X,ω, func,ζ)

    if !isnothing(quantizer)
        A = quantize(quantizer,A)
    end
    
    lasso = LassoRegression(lasso_lambda; fit_intercept=false)
    solver = FISTA(max_iter=max_iter)
    c = MLJLinearModels.fit(lasso,A,Y;solver)
    #prune!(c,pruning)
    return c, ω ,ζ
end