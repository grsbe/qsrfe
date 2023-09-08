using Distributions, Random, LinearAlgebra

# just the matrix of two random fourier features 
function z(xk,wi,wj,ti,tj)
    return [cos(xk ⋅ wi + ti) cos(xk ⋅ wj + tj)]
end

# scalar product of two fourier feature vectors
function z2(xk,wi,wj,ti,tj)
    return cos(xk ⋅ wi + ti) ⋅ cos(xk ⋅ wj + tj)
end

#computes scalar product of two fourier feature vectors
function trial(m = 1000, d = 100)
    #init arrays
    arr = Array{Float64}(undef, 0, 2)
    wi = rand(Normal(),d)
    wj = rand(Normal(),d)
    ti = rand(Normal())
    tj = rand(Normal())

    s = 0.0
    for i in 1:m
        xk = rand(Normal(),d)
        arr = vcat(arr,z(xk,wi,wj,ti,tj)) #builds the array
        s += z2(xk,wi,wj,ti,tj)
    end

    return [(s / (norm(arr[:,1]) * norm(arr[:,2]))) (1 / (norm(arr[:,1]) * norm(arr[:,2])))]
end


dat = Array{Float64}(undef, 0, 2)
for i in 1:2000
    dat = vcat(dat,trial())
    if i % 1000 == 0
        print("it: ", i, " ")
    end
end

# checking
2 / 1000
mean(dat[:,2])
mean(dat[:,1])


using Plots
histogram(dat[:,2], bins=1000)

new = dat[:,2] .- 500 .+ 1000*(2/π)


trials = 20000
d = 50
arr = Array{Float64}(undef, 0, 2)
s = 0.0
wi = rand(Normal(),d)
wj = rand(Normal(),d)
ti = rand(Normal())
tj = rand(Normal())

for i in 1:trials
    xk = rand(Normal(),d)
    arr = vcat(arr,z(xk,wi,wj,ti,tj))
    s += z2(xk,wi,wj,ti,tj)
end

res = s / (norm(arr[:,1]) * norm(arr[:,2]))

norm(arr[:,1]) * norm(arr[:,2])
normlist = [norm(arr[1:a,1]) * norm(arr[1:a,2]) for a in 1:trials]

using Plots
plot(1:trials,normlist,label="norm product")
m = 0.5
plot!(1:trials,f(1:trials,m),label="m=$(m)")

f(x,m) = x*m