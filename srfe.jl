module srfe

export fit_srfe, prune!, ReLU, rff, rel_error, compute_featuremap, MSQ, βQ, ΣΔQ, quantize, condense

using LinearAlgebra
using Random, Distributions
using MLJ
using MLJLinearModels
using Parameters

include("quantization.jl")
include("model.jl")
include("utils.jl")

end #end module

