#error calc

function rel_L2_error(y_truth, y_pred)
    norm(y_truth - y_pred) / norm(y_truth)
end

function rmse(y_truth, y_pred)
    sqrt(mean((y_truth - y_pred).^2))
end

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


using DataFrames
function create_df(initial_vector; empty=true)
    # creates an empty dataframe with dimensions of the initial vector as one row
    # this is way more complicated than one empty column in Dataframes
    df = DataFrame(B = initial_vector)
    df[!, :id] = 1:size(df, 1)
    tmp = stack(df)
    df = select!(unstack(tmp, :id, :value), Not(:variable))
    if empty
        deleteat!(df,1)
    end
    return df
end


function trainandevaluate(model,quant::Quantizer, (xtrain, xtest), (ytrain, ytest);trials=1)
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
        c = fit(model,xtrain,ytrain,quant)
        ytrainpred = predict(model,xtrain,quant)
        ytestpred = predict(model,xtest,quant)
        testerror[i],trainerror[i] = rel_error(ytest,ytestpred), rel_error(ytrain,ytrainpred)
        abstesterror[i], abstrainerror[i] = abs_error(ytest,ytestpred), abs_error(ytrain,ytrainpred)
        msetesterror[i], msetrainerror[i] = mse(ytest,ytestpred), mse(ytrain,ytrainpred)
        relmsetesterror[i], relmsetrainerror[i] = rel_mse(ytest,ytestpred), rel_mse(ytrain,ytrainpred)
        println("here:", mse(ytest,ytestpred))
    end

    return mean(testerror), mean(trainerror), mean(abstesterror), mean(abstrainerror), mean(msetesterror), mean(msetrainerror), mean(relmsetesterror), mean(relmsetrainerror)
end

function trainandevaluate(model,(xtrain, xtest), (ytrain, ytest);trials=1)
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
        c = fit(model,xtrain,ytrain)
        ytrainpred = predict(model,xtrain)
        ytestpred = predict(model,xtest)
        testerror[i],trainerror[i] = rel_error(ytest,ytestpred), rel_error(ytrain,ytrainpred)
        abstesterror[i], abstrainerror[i] = abs_error(ytest,ytestpred), abs_error(ytrain,ytrainpred)
        
        msetesterror[i], msetrainerror[i] = mse(ytest,ytestpred), mse(ytrain,ytrainpred)
        relmsetesterror[i], relmsetrainerror[i] = rel_mse(ytest,ytestpred), rel_mse(ytrain,ytrainpred)

    end

    return mean(testerror), mean(trainerror), mean(abstesterror), mean(abstrainerror), mean(msetesterror), mean(msetrainerror), mean(relmsetesterror), mean(relmsetrainerror)
end