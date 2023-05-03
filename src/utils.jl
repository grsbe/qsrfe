#error calc
function rel_error(y_truth, y_pred)
    mean(abs.((y_truth - y_pred) ./ y_truth))
end

function mse(y_truth, y_pred)
    mean((y_truth - y_pred).^2)
end