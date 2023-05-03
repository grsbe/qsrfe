module qsrfe

export fit_srfe, prune!, ReLU, rff, rel_error, gen_weights, compute_featuremap, MSQ, βQ, ΣΔQ, quantize, condense, Quantizer, srfeRegressor, fit, predict, mse

using LinearAlgebra
using Random, Distributions
using MLJ
using MLJLinearModels
using Parameters

include("quantization.jl")
include("model.jl")
include("utils.jl")

end # module
