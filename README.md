# qsrfe

implementation of Sparse Random Feature Expansions and modern quantization algorithms

## Usage

### importing the package

> pkg> add <https://github.com/grsbe/qsrfe.jl>  
> julia> using qsrfe  

### fitting a srfe model

> model = srfeRegressor(N=N,λ=λ, σ2=1.0, intercept=true)  
> c, ω, ζ = qsrfe.fit(model,xtrain,ytrain;max_iter=2000000,verbose=true)  
> ytrainpred = qsrfe.predict(model,xtrain,c, ω, ζ)  

### fitting a quantized srfe model

> #K = number of points in [-1,1]  
> quant1 = MSQ(K=2)  
> quant2 = ΣΔQ(K=2,r=1,λ=32,condense=true)  
> quant3 = βQ(K=2,β=1.5,λ=32,condense=true)  
> c, ω, ζ = qsrfe.fit(model,xtrain,ytrain,quant3;max_iter=2000000,verbose=true)  
> ytrainpred = qsrfe.predict(model,xtrain,c, ω, ζ,quant)  
