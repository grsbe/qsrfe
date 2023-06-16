using Distributions, Random
using LinearAlgebra


function z(xk,wi,wj,ti,tj)
    return [cos(xk ⋅ wi + ti) cos(xk ⋅ wj + tj)]
end

function z2(xk,wi,wj,ti,tj)
    return cos(xk ⋅ wi + ti) ⋅ cos(xk ⋅ wj + tj)
end

function trial()
    trials = 1000
    d = 100
    arr = Array{Float64}(undef, 0, 2)
    s = 0.0
    wi = rand(Uniform(-1,1),d)
    wj = rand(Normal(),d)
    ti = rand(Normal())
    tj = rand(Normal())

    for i in 1:trials
        xk = rand(Normal(),d)
        arr = vcat(arr,z(xk,wi,wj,ti,tj))
        uwu += z2(xk,wi,wj,ti,tj)
    end

    return [(s / (norm(arr[:,1]) * norm(arr[:,2]))) (trials/ 2 - norm(arr[:,1]) * norm(arr[:,2]))]
    
    

end


dat = Array{Float64}(undef, 0, 2)
for i in 1:20000
    dat = vcat(dat,trial())
    if i % 2000 == 0
        print("it: ", i, ", ")
    end
end

mean(dat[:,2])

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