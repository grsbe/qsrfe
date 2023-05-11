
@with_kw struct srfeRegressor
    N::Int64 = 1000
    λ::Float64 = 0.003
    q::Int64 = 0
    func = rff
    σ2::Float64 = 1
    pruning::Float64=1.0
    intercept::Bool = false
end


function fit(model::srfeRegressor,X,y;quantizer::Quantizer=nothing,max_iter=20000)
    #weights
    m,d = size(X)
    ω, ζ = gen_weights(model.N,d,model.q,model.σ2)
    #println("compute features")
    A = compute_featuremap(X,ω, model.func,ζ)

    if !isnothing(quantizer)
        A = quantize(quantizer,A)
        if quantizer.condense
            A = condense(quantizer,A)
        end
    end
    
    lasso = LassoRegression(lasso_lambda; fit_intercept=model.intercept) #if intercept is true, the last element of c is the intercept
    solver = FISTA(max_iter=max_iter)
    c = MLJLinearModels.fit(lasso,A,Y;solver)
    prune!(c,model.pruning)
    return c, ω ,ζ
end

function predict(model::srfeRegressor,X,c,ω,ζ;quantizer::Quantizer=nothing)
    A = compute_featuremap(X,ω, model.func,ζ)
    if model.intercept
        A = hcat(A,ones(size(A,1)))
    end
    if !isnothing(quantizer)
        A = quantize(quantizer,A)
        if quantizer.condense
            A = condense(quantizer,A)
        end
    end
    return A * c
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

function prune(c, s=0.2)
    # s gives the top percentage sparsity
    # s = 0.2 -> only keep the top 20% of non zero values
    if s >= 1 || s < 0
        return copy(c)
    end
    c_ = copy(c)
    s = round(Int, min(length(c),length(c) * s))
    indexperm = partialsortperm(abs.(c_),1:s, rev=true) #max-s indices
    c_[1:length(c) .∉ Ref(indexperm)] .= 0.0 #turn all indices not in indexperm to 0.0
    return c_
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
        if quantizer.condense
            A = condense(quantizer,A)
        end
    end
    
    lasso = LassoRegression(lasso_lambda; fit_intercept=false)
    solver = FISTA(max_iter=max_iter)
    c = MLJLinearModels.fit(lasso,A,Y;solver)
    prune!(c,pruning)
    return c, ω ,ζ
end