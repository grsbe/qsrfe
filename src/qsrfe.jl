module qsrfe



using LinearAlgebra
using Random, Distributions
using MLJ
using MLJLinearModels
using Parameters
using DataFrames
using ToeplitzMatrices
using FFTW

include("quantization.jl")
export MSQ, βQ, ΣΔQ, quantize, condense, Quantizer
include("model.jl")
export fit_srfe, prune!, prune, ReLU, rff, gen_weights, compute_featuremap, srfeRegressor, fit, predict
include("utils.jl")
export load_dataset, rel_error, mse, test_metrics, trainandevaluate, abs_error

end # module
