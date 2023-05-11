#quantization schemes

abstract type Quantizer end

@with_kw struct MSQ <: Quantizer
    K::Int64 = 3
    limit::Float64 = 1
    condense::Bool = false
end

@with_kw struct ΣΔQ <: Quantizer
    K::Int64 = 3
    limit::Float64 = 1
    r::Int8 = 1
    λ::Int64 = 3
    condense::Bool = false
end

@with_kw struct βQ <: Quantizer
    K::Int64 = 3
    limit::Float64 = 1
    β::Float64 = 1.2
    λ::Int64 = 2
    condense::Bool = false
end

#quantize function signatures
function quantize(q::Quantizer,A::AbstractMatrix{<:Real})
    return _quantize(q,A)
end

function _quantize(q::MSQ,A)
    return _MSQ(A,q.K,q.limit)
end

function _quantize(q::ΣΔQ,A)
    return _ΣΔQ(A,q.K,q.r,q.limit)
end

function _quantize(q::βQ,A)
    return _βQ(A,q.β,q.λ,q.K,q.limit)
end

#condense functions
function condense(q::Quantizer,A::AbstractMatrix{<:Real})
    return _condense(q,A)
end

function _condense(q::ΣΔQ,A)
    return ΣΔcondense(A,q.r,q.λ)
end

function _condense(q::βQ,A)
    return βcondense(A,q.β,q.λ)
end


#functions that do the quantizing
function rounding_quantizer_even(a,Δ,limit)
    if abs(a) <= limit
        return floor(a / Δ) * Δ + Δ / 2
    else
        return sign(a) * limit
    end
end

function rounding_quantizer_odd(a,Δ,limit)
    if abs(a) <= limit
        return round(a / Δ) * Δ
    else
        return sign(a) * limit
    end
end

function stepsize_even(K)
    if K <=1
        error("Must choose K >= 2")
    end
    K = K/2
    return K = 1 / (K - 1/2)
end

function stepsize_odd(K)
    if K <=1
        error("Must choose K >= 2")
    end
    return 2 / (K-1)
end

function _MSQ(A,K,limit=1)
    m, N = size(A)
    if isodd(K)
        Δ = stepsize_odd(K)
        rounding_quantizer = rounding_quantizer_odd
    else
        Δ = stepsize_even(K)
        rounding_quantizer = rounding_quantizer_even
    end
    
    q = zeros(Float64,(m,N))
    for i in 1:m
        for j in 1:N
            q[i,j] = rounding_quantizer(A[i,j],Δ,limit)
        end
    end
    return q
end

function _ΣΔQ(A,K,r=1,limit=1)
    m, N = size(A)
    if isodd(K)
        Δ = stepsize_odd(K)
        rounding_quantizer = rounding_quantizer_odd
    else
        Δ = stepsize_even(K)
        rounding_quantizer = rounding_quantizer_even
    end
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

function ΣΔcondense(q,r,λ)
    m, N = size(q)
    if λ != 0
        if N % λ != 0
            error("choose number of weights to be divisible by λ")
        end
        p = N / λ
        p = convert(Int128, p)
        if r == 1
            V = kron(diagm(ones(p)),ones(λ))'
            q = V * transpose(q) .* sqrt(2/p) ./ norm(ones(λ))
            q = transpose(q)
        elseif r==2 
            if λ % 2 == 0
                error("hat_λ not an integer, choose uneven λ")
            end
            hat_λ = convert(Int128,(λ + 1) / 2)
            v= [1:hat_λ;(hat_λ-1):-1:1]
            V = kron(diagm(ones(p)),v)'
            q = V * transpose(q) .* sqrt(2/p) ./ norm(v)
            q = transpose(q)
        end
        
    end
    return q
end

using ToeplitzMatrices
using FFTW

function _βQ(A, β,λ, K,limit=1,)
    m, N = size(A)
    if isodd(K)
        Δ = stepsize_odd(K)
        rounding_quantizer = rounding_quantizer_odd
    else
        Δ = stepsize_even(K)
        rounding_quantizer = rounding_quantizer_even
    end
    if λ == 0
        λ = 1
    end
    if N % λ != 0
        error("choose number of weights to be divisible by λ")
    end
    
    p = N / λ
    p = convert(Int64,p)
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

function βcondense(q,β,λ)
    m, N = size(q)
    p = N / λ
    p = convert(Int64,p)
    v= [β^(-i) for i in 1:λ]
    V = kron(diagm(ones(p)),v)'
    q = V * transpose(q) .* sqrt(2/p) ./ norm(v)
    return transpose(q)
end




