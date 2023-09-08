#error calc
function rel_error(y_truth, y_pred)
    mean(abs.((y_truth - y_pred) ./ y_truth))
end

function mse(y_truth, y_pred)
    mean((y_truth - y_pred).^2)
end

#dataset loader


function load_dataset(X,Y;normalize=true,partitioning=0.8)
    X = Matrix(DataFrame(X))
    Y = collect(Y)

    if normalize
        foreach(normalize!, eachcol(X))
    end
    
    return partition((X, Y), partitioning, rng=123, multi=true)
end

function test_metrics(ytest,ypred,ytrain,ypredtrain)
    println("MSE: ",mse(ytest,ypred)," train MSE: ",mse(ypredtrain,ytrain))
    println("rel: ",rel_error(ytest,ypred)," train rel: ",rel_error(ytrain,ypredtrain))
    return
end

using ProgressBars

function trainandevaluate(model::srfeRegressor,quant::Quantizer, X,Y;trials=1,normalize=true,partitioning=0.8)
    (xtrain, xtest), (ytrain, ytest) = load_dataset(X,Y;normalize=normalize,partitioning=partitioning)
    besttesterror = 100000000.0
    besttrainerror = 100000000.0
    a,b = 0.0, 0.0
    for i in ProgressBar(1:trials)
        c, ω, ζ = qsrfe.fit(model,xtrain,ytrain,quant)
        ytrainpred = qsrfe.predict(model,xtrain,c, ω, ζ)
        ytestpred = qsrfe.predict(model,xtest,c, ω, ζ)
        a,b = rel_error(ytest,ytestpred), rel_error(ytrain,ytrainpred)
        if a < besttesterror
            besttesterror, besttrainerror = a,b
        end
        
    end

    return besttesterror, besttrainerror
end

function trainandevaluate(model::srfeRegressor,(xtrain, xtest), (ytrain, ytest);trials=1,normalize=true,partitioning=0.8)
    (xtrain, xtest), (ytrain, ytest) = load_dataset(X,Y;normalize=normalize,partitioning=partitioning)
    testerror = Array{Float64}(undef,trials)
    trainerror = Array{Float64}(undef,trials)

    for i in ProgressBar(1:trials)
        c, ω, ζ = qsrfe.fit(model,xtrain,ytrain)
        ytrainpred = qsrfe.predict(model,xtrain,c, ω, ζ)
        ytestpred = qsrfe.predict(model,xtest,c, ω, ζ)
        testerror[i],trainerror[i] = rel_error(ytest,ytestpred), rel_error(ytrain,ytrainpred)
    end

    return mean(besttesterror), mean(besttrainerror)
end