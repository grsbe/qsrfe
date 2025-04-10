# all the solvers for the L2, BP, BPDN and LASSO problem, either by calling a function or self-implementation
using QRupdate, LinearAlgebra
using MLJLinearModels
using SparseArrays

function L2(A,y,λ=0.0)
    # fit a linear model
    if λ <= 0
        return A \ y
    else
        return (A' * A + λ * I) \ (A' * y)
    end
end

using LinearAlgebra
function gain(A,y,xa,Sa,So; posterior_covariance = false)
    G = Sa * A' / (A * Sa * A' + So)
    return G
end

function posterior(A,Sa,So)
    return inv(Symmetric(A' * inv(So) * A + inv(Sa)))
end

function BayesianInversion(A,y,xa,Sa,So)
    diff = (Sa * A' * (So + A * Sa * A')) \ (y - A * xa)
    return xa + diff
end


let cache = Dict{Float64, Matrix{Float64}}()
    global function cached_L2(A,y,λ)
        if λ ∉ keys(cache)
            cache[λ] = (A' * A + I * λ) \ (foot' * y)
        end
        return cache[λ]
    end
end


function LASSO(A::AbstractMatrix, y::AbstractVector, λ::Float64, intercept::Bool=false, max_iter::Integer=200000)
    lasso = LassoRegression(λ; fit_intercept=intercept) #if intercept is true, the last element of c is the intercept
    solver = FISTA(max_iter=max_iter)
    c = MLJLinearModels.fit(lasso,A,y;solver)
    return c
end

function ElasticNet(A::AbstractMatrix, y::AbstractVector, λ::Float64, α::Float64, intercept::Bool=false, max_iter::Integer=200000)
    elastic = ElasticNetRegression(λ,α; fit_intercept=intercept) #if intercept is true, the last element of c is the intercept
    solver = FISTA(max_iter=max_iter)
    c = MLJLinearModels.fit(elastic,A,y;solver)
    return c
end


function OMP(A::AbstractMatrix, y::AbstractVector, s::Int = size(A,2),tol::Union{Nothing,Float64}=10^(-10); normalize::Bool = false)
    x = fill(0.0,size(A,2))
    S = (Int)[]
    R = Array{Float64, 2}(undef, 0, 0)
    j = 0
    B = copy(A)
    if normalize
        normalize!.(eachcol(B))
    end
    while  length(S) < s
        if (!isnothing(tol) && norm(y - A * x) <= tol)
            break
        end
        j = argmax( B' * (y - B * x) )
        R = qraddcol(A[:,S], R, A[:,j])
        push!(S,j)
        x[S], _ = csne(R, A[:,S], y)
    end
    return x
end

"""
RandOMP(A, b; tol=1e-10)

Implementation of randomized orthogonal matching pursuit (RandOMP) algorithm,
as in [1].

[1] Elad, Michael, and Irad Yavneh. “A Plurality of Sparse Representations Is Better Than the Sparsest One Alone.” 
    IEEE Transactions on Information Theory 55, no. 10 (October 2009): 4701–14. 
    https://doi.org/10.1109/TIT.2009.2027565.

"""
function RandOMP(A, b, σ, σx; max_iter=nothing, tol=1e-10)
    if max_iter === nothing
        max_iter = size(A, 1)
    end
    c = σx^2 / (σx^2 + σ^2)
    coef = ((c^2)/(2*σ^2))
    norm_vec = norm.(eachcol(A))
    n = size(A, 2)
    x = zeros(n)
    r = copy(b)
    pr_vec = similar(x)
    tmp = similar(x)
    S = []
    notS = collect(1:n)
    for i in 1:max_iter
        tmp .= (A' * r).^2 ./ norm_vec
        L = maximum(tmp)
        pr_vec = exp.(coef * (tmp .- L))

        j = sample(notS, Weights(pr_vec[notS]))
        push!(S, j)
        notS = notS[notS .!= j]
        x[S] .= A[:, S] \ b
        r .= b - A * x
        if norm(r) < tol
            break
        end
    end
    return x
end

function MMSE_RandOMP(A, b, σ, σx, N; max_iter=nothing, tol=1e-10)
    x0 = zeros(size(A, 2))
    for i=1:N
        x0 .+= (1/N) * RandOMP(A, b, σ, σx; max_iter=max_iter, tol=tol)
    end
    return x0
end


function coherence_band(index::Int64, C, η)
    """
    compute the η coherence coherence band for the index
    output: bitvector of the coherence band 
    1 = in band, 0 not in band
    """
    return C[index,:] .> η
end

function coherence_band(index::Vector{Int64}, C, η)
    """
    compute the η coherence coherence band for the index
    returns bitmap which represent whether the index is in a coherence band of the support
    1 = in a band of the given index list
    0 = not in a band
    """
    row = falses(size(C,2))
    for i in index
        row = row .| coherence_band(i, C, η) #bitwise or to combine all bitmap hits, this returns a bitmap :( this is hacky .| since bitvectors are actually arrays
    end
    return row
end


function LO(A,y,S_0,C, η; tol=10^-(14)) #local optimization
    # get list of indices in S, that determine the support of x, e.g. the active columns in A
    S = copy(S_0)
    fullS = collect(1:size(A,2))
    for s in S_0
        S = union(S,fullS[coherence_band(s,C,η)])
        x = A[:,S] \ y #how is this gonna stay sparse, like what the hell i dont get it
        S = S[abs.(x) .> tol]
    end
    return S
end

function wrong_LO(A::AbstractMatrix, y::AbstractVector, x::AbstractVector, S::Vector{Int64})
    """ This is what I thought, the algo did but it didnt.., keeping this in case i actually still use it
    expects S to be a bitmap with ones at the area of interest"""
    
    y_new = y - A[:,ind] * x[ind]
    x_ = copy(x)
    x_[S] = A[:,S] \ y_new  # more efficient version possible?, this gets very very slow if S is big
    return x_
end


function BLOOMP(A::AbstractMatrix, y::AbstractVector,η::Float64,s::Int = size(A,2);C::Union{AbstractMatrix,Nothing}=nothing, tol=10^(-14))
    x = fill(0.0,size(A,2))
    x_old = fill(0.0,size(A,2))
    S = (Int)[] #current support
    #R = Array{Float64, 2}(undef, 0, 0)
    ranklist = fill(0.0,size(A,2))
    if isnothing(C)
        _ , C = coherence(A)
    end
    fullS = collect(1:size(C,2))
    iter = 0
    while true #length(S) < s
        iter = iter + 1
        if iter >= s #length(x)
            println("stopped after $(iter -1) iterations")
            break
        end
        
        x_old = x[:] #save copy of old x

        #find coherence band
        band_excluded = fullS[coherence_band(fullS[coherence_band(S, C, η)],C,η)] #double band exclusion
        
        #find next point source with band exclusion
        ranklist = A' * (y - A * x)
        ranklist[band_excluded] .= 0.0
        j = argmax( ranklist )
        S = union(S,j)
        #local optimization
        S = LO(A,y,S,C,η;tol=tol)

        #solve
        x[S] = A[:,S] \ y

        #use this instead if we need more speed: (not tested if works or if it is actually faster)
        #R = qraddcol(A[:,S_old], R, A[:,exclusion(S,S_old)])
        #x[S], _ = csne(R, A[:,S], y)

        #stop if sol is found or nothing changes anymore
        if !isnothing(tol) && (norm(y - A * x) <= tol || norm(x-x_old) <= tol)
            break
        end
    end
    return x
end


# work in progress
function BOMP(A::AbstractMatrix, y::AbstractVector,η::Float64,s::Int = size(A,2), max::Int=size(A,2);C::Union{AbstractMatrix,Nothing}=nothing, tol=10^(-14))
    x = fill(0.0,size(A,2))
    S = (Int)[]
    R = Array{Float64, 2}(undef, 0, 0)
    j = 0
    ranklist = fill(0.0,size(A,2))
    if isnothing(C)
        _ , C = coherence(A)
    end
    fullS = collect(1:size(C,2))
    iter = 0
    while  length(S) < s && iter < max
        iter = iter + 1
        if (!isnothing(tol) && norm(y - A * x) <= tol)
            break
        end
        band_excluded = fullS[coherence_band(fullS[coherence_band(S, C, η)],C,η)] #double band exclusion
        #find next point source with band exclusion
        ranklist = A' * (y - A * x)
        ranklist[band_excluded] .= 0.0
        j = argmax( ranklist )

        R = qraddcol(A[:,S], R, A[:,j])
        push!(S,j)
        x[S], _ = csne(R, A[:,S], y)
    end
    return x
end


function IRLS(A::AbstractMatrix,y::AbstractVector,ϵ::Float64,γ::Float64 = 1.0; max_iter=1000, epss = 10^(-20))
    w = fill(1.0,size(A,2))
    x = fill(0.0,size(A,2))
    v = fill(0.0,size(A,1))
    iter = 0
    while iter < max_iter && ϵ > epss #norm(A * x - y) > epss enabling the last stopping condition might break the algo
        # this only works if A has full row rank
        if rank(A) >= maximum(size(A))
            v = (A * diagm(w.^(-1)) * A') \ y #this is expensive-ish
            x = diagm(w.^(-1)) * A' * v
        else # if not full rank, this is way slower
            if iter < 2
                println("slower solving algo")
            end
            #x = diagm(w.^(-0.5)) * pinv(A * diagm(w.^(-0.5))) * y
            x = (A * diagm(w.^(-0.5))) \ y
            x = diagm(w.^(-0.5)) * x
        end
        ϵ = min(ϵ,γ * maximum(x))
        w .= 1 ./ sqrt.( x.^2 .+ ϵ )
        iter = iter + 1
    end
    return x
end


# model = Model(Gurobi.Optimizer)
# # @variable(x, x>= 0)  #if we wanted to enforce positivity
# @constraint(model, c, A * x == y) # no noise
# @constraint(model, c2, (A * x - y).^2 .<= ϵ) #noise
using JuMP, Gurobi
function BPDN(A::AbstractMatrix, y::AbstractVector, noise::Float64 = 0.0; positive::Bool = false, verbose=false)
    n = size(A)[2]
    model = Model(Gurobi.Optimizer)
    if !verbose
        set_attribute(model, "LogToConsole", 0)
    end
    @variable(model, x[1:n])
    @variable(model, t)
    @constraint(model, [t; x] in MOI.NormOneCone(1 + length(x))) # this is apparently the way to go to program L1 minimization
    if noise > 0.0
        @constraint(model, [noise; A * x - y] in MOI.SecondOrderCone(1 + length(y))) # formulation as second cone makes solver way faster?
    else
        @constraint(model, c, A * x == y) # no noise
    end
    if positive
        @constraint(model, x .>= 0)
    end

    @objective(model, Min, t)
    optimize!(model)
    if is_solved_and_feasible(model)
        xout = [value(x[i]) for i in 1:n] #unpack results
        return xout  #, model
    else
        throw("No feasable solution or solution found")
        return nothing
    end
end

function BPDNL1(A::AbstractMatrix, y::AbstractVector, noise::Float64 = 0.0, positive::Bool = false)
    n = size(A)[2]
    model = Model(Gurobi.Optimizer)
    @variable(model, x[1:n])
    @variable(model, t)
    @constraint(model, [t; x] in MOI.NormOneCone(1 + length(x))) # this is apparently the way to go to program L1 minimization
    if noise > 0.0
        @constraint(model, c2, [noise; A * x - y] in MOI.NormOneCone(1 + length(y)))
    else
        @constraint(model, c, A * x == y) # no noise
    end
    if positive
        @constraint(model, x .>= 0)
    end
    @objective(model, Min, t)
    optimize!(model)
    if is_solved_and_feasible(model)
        xout = [value(x[i]) for i in 1:n] #unpack results
        return xout  #, model
    else
        throw("No feasable solution or solution found")
        return nothing
    end
end



function ELNET_Gurobi(A::AbstractMatrix, y::AbstractVector, noise::Float64 = 0.0, ratio::Float64 = 0.5; verbose=false)
    # solves the elastic net problem directly
    # min |x|_1 + λ |x|_2^2   so that   |Ax - y|_2^2 <= ϵ
    # ratio is the ratio between the L1 and L2 norm
    # 0.0 is pure L1, 1.0 is pure L2
    if (0. > ratio) || (ratio > 1.)
        throw("ratio must be between 0 and 1")
    end
    n = size(A)[2]
    model = Model(Gurobi.Optimizer)
    if !verbose
        set_attribute(model, "LogToConsole", 0)
    end
    @variable(model, x[1:n])
    @variable(model, t)
    @constraint(model, [t; x .* (1-ratio)] in MOI.NormOneCone(1 + length(x))) # this is apparently the way to go to program L1 minimization
    @constraint(model, [t; x .* ratio] in MOI.SecondOrderCone(1 + length(x)))
    if noise > 0.0
        @constraint(model, [noise; A * x - y] in MOI.SecondOrderCone(1 + length(y))) # formulation as second cone makes solver way faster?
    else
        @constraint(model, c, A * x == y) # no noise
    end
    @objective(model, Min, t)
    optimize!(model)
    if is_solved_and_feasible(model)
        xout = [value(x[i]) for i in 1:n] #unpack results
        return xout  #, model
    else
        throw("No feasable solution or solution found")
        return nothing
    end
end
