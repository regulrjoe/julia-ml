







# Logistic regression hypothesis function
# 1 / (1 + e^-(0'x))
# Input:
#   X -> nxm Matrix of input data set
#   T -> n Vector of parameters to evaluate
function h(X, T)
    X = Helpers.check_ones_col(X, T)
    Y = Array{Float64}(size(X,1))
    for i = 1:size(X,1)
        Y[i] = Sigmoid.g(T' * X[i,:])
    end
    return Y
end

# Sigmoid function for logistic regression
function g(z)
    1 / (1 + e^-z)
end
