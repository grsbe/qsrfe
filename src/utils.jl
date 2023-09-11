#error calc
function rel_error(y_truth, y_pred)
    mean(abs.((y_truth - y_pred) ./ y_truth))
end

function abs_error(y_truth, y_pred)
    mean(abs.(y_truth - y_pred))
end

function mse(y_truth, y_pred)
    mean((y_truth - y_pred).^2)
end

function rel_mse(y_truth, y_pred)
    norm(y_truth - y_pred) / norm(y_truth)
end

#dataset loader


function load_dataset(X,Y;normalize=true,partitioning=0.8,rng=123)
    X = Matrix(DataFrame(X))
    Y = collect(Y)

    if normalize
        foreach(normalize!, eachcol(X))
    end
    
    return partition((X, Y), partitioning, rng=rng, multi=true)
end

function test_metrics(ytest,ypred,ytrain,ypredtrain)
    println("MSE: ",mse(ytest,ypred)," train MSE: ",mse(ypredtrain,ytrain))
    println("rel: ",rel_error(ytest,ypred)," train rel: ",rel_error(ytrain,ypredtrain))
    return
end



function trainandevaluate(model::srfeRegressor,quant::Quantizer, (xtrain, xtest), (ytrain, ytest);trials=1)
    #(xtrain, xtest), (ytrain, ytest) = load_dataset(X,Y;normalize=normalize,partitioning=partitioning)
    testerror = Array{Float64}(undef,trials)
    trainerror = Array{Float64}(undef,trials)
    
    abstesterror = Array{Float64}(undef,trials)
    abstrainerror = Array{Float64}(undef,trials)

    msetesterror = Array{Float64}(undef,trials)
    msetrainerror = Array{Float64}(undef,trials)

    relmsetesterror = Array{Float64}(undef,trials)
    relmsetrainerror = Array{Float64}(undef,trials)

    for i in 1:trials
        c, ω, ζ = qsrfe.fit(model,xtrain,ytrain,quant)
        ytrainpred = qsrfe.predict(model,xtrain,c, ω, ζ,quant)
        ytestpred = qsrfe.predict(model,xtest,c, ω, ζ,quant)
        testerror[i],trainerror[i] = rel_error(ytest,ytestpred), rel_error(ytrain,ytrainpred)
        abstesterror[i], abstrainerror[i] = abs_error(ytest,ytestpred), abs_error(ytrain,ytrainpred)
        msetesterror[i], msetrainerror[i] = mse(ytest,ytestpred), mse(ytrain,ytrainpred)
        relmsetesterror[i], relmsetrainerror[i] = rel_mse(ytest,ytestpred), rel_mse(ytrain,ytrainpred)

    end

    return mean(testerror), mean(trainerror), mean(abstesterror), mean(abstrainerror), mean(msetesterror), mean(msetrainerror), mean(relmsetesterror), mean(relmsetrainerror)
end

function trainandevaluate(model::srfeRegressor,(xtrain, xtest), (ytrain, ytest);trials=1)
    #(xtrain, xtest), (ytrain, ytest) = load_dataset(X,Y;normalize=normalize,partitioning=partitioning)
    testerror = Array{Float64}(undef,trials)
    trainerror = Array{Float64}(undef,trials)
    
    abstesterror = Array{Float64}(undef,trials)
    abstrainerror = Array{Float64}(undef,trials)

    msetesterror = Array{Float64}(undef,trials)
    msetrainerror = Array{Float64}(undef,trials)

    relmsetesterror = Array{Float64}(undef,trials)
    relmsetrainerror = Array{Float64}(undef,trials)

    for i in 1:trials
        c, ω, ζ = qsrfe.fit(model,xtrain,ytrain)
        ytrainpred = qsrfe.predict(model,xtrain,c, ω, ζ)
        ytestpred = qsrfe.predict(model,xtest,c, ω, ζ)
        testerror[i],trainerror[i] = rel_error(ytest,ytestpred), rel_error(ytrain,ytrainpred)
        abstesterror[i], abstrainerror[i] = abs_error(ytest,ytestpred), abs_error(ytrain,ytrainpred)
        msetesterror[i], msetrainerror[i] = mse(ytest,ytestpred), mse(ytrain,ytrainpred)
        relmsetesterror[i], relmsetrainerror[i] = rel_mse(ytest,ytestpred), rel_mse(ytrain,ytrainpred)

    end

    return mean(testerror), mean(trainerror), mean(abstesterror), mean(abstrainerror), mean(msetesterror), mean(msetrainerror), mean(relmsetesterror), mean(relmsetrainerror)
end