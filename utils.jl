#error calc
function rel_error(y_truth, y_pred)
    mean(abs.((y_truth - y_pred) ./ y_truth))
end