module LogisticRegression

include("feature_scaling.jl")
include("gradient_descent.jl")
include("helpers.jl")

export train, run

function train(X::Array{Float64,2}, Y::Array{Float64,1})
end

# Cost function
# 1m * ()−y'log⁡(h) − (1−y)' * log⁡(1−h))
# Input:
#   X -> Matrix of input data set
#   Y -> Vector of output of training set
#   T -> Vector of parameters to evaluate
#   h -> Hypothesis function to evaluate
function J(X::Array{Float64,2}, Y::Array{Float64,1}, T::Array{Float64,1})
    m = size(X,1)
    1/m * (-Y' * log.(h(X,T)) - (1 - Y)' * log.(1 - h(X,T)))
end

# Logistic regression hypothesis function
# 1 / (1 + e^-(0'x))
# Input:
#   X -> nxm Matrix of input data set
#   T -> n Vector of parameters to evaluate

# Applied to vector
function h(x::Array{Float64,1}, T::Array{Float64,1})
    g(T' * x)
end
# Applied to matrix
function h(X::Array{Float64,2}, T::Array{Float64,1})
    Y = Array{Float64}(size(X,1))
    for i = 1:size(X,1)
        Y[i] = g(T' * X[i,:])
    end
    return Y
end

# Sigmoid function for logistic regression
function g(z)
    1 / (1 + e^-z)
end

end
