#error calc
function rel_error(y_truth, y_pred)
    mean(abs.((y_truth - y_pred) ./ y_truth))
end

function mse(y_truth, y_pred)
    mean((y_truth - y_pred).^2)
end

#dataset loader
using MLJ, DataFrames

function load_dataset(X,Y;normalize=true,partitioning=0.8)
    X = Matrix(DataFrame(X))
    Y = collect(Y)

    if normalize
        X = (X .- mean(X)) ./ std(X)
    end
    
    return partition((X, Y), partitioning, rng=123, multi=true)
end

function test_metrics(ytest,ypred,ytrain,ypredtrain)
    println("MSE: ",mse(ytest,ypred)," train MSE: ",mse(ypredtrain,ytrain))
    println("rel: ",rel_error(ytest,ypred)," train rel: ",rel_error(ytrain,ypredtrain))
    return
end