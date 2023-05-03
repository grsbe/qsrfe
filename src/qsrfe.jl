module qsrfe

export fit_srfe, prune!, ReLU, rff, gen_weights, compute_featuremap, MSQ, βQ, ΣΔQ, quantize, condense, Quantizer, srfeRegressor, fit, predict

using LinearAlgebra
using Random, Distributions
using MLJ
using MLJLinearModels
using Parameters

include("quantization.jl")
include("model.jl")

include("utils.jl")
export load_dataset, rel_error, mse

end # module
