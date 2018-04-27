module LogisticRegression

include("feature_scaling.jl")
include("gradient_descent.jl")
include("helpers.jl")

export train, predict

# Gradient Descent Configuration
mutable struct GDConfig
    alpha::Float64
    epsilon::Float64
    max_its::Int64
end
# Output Logisitc Regression Parameters
mutable struct Parameters
    thetas::Array{Float64,1}
    normalization::Bool
    means::Array{Float64,1} # means for normalization
    sdevs::Array{Float64,1} # standard deviations for normalization
end

function train(X::Array{Float64,2}, Y::Array{Float64,1})
end

# Run theta parameters on new entry without normalization
# Input
#   X   ->  Input data
#   T   ->  Theta parameters
# Output
#   Hypothesis of X with T
function predict(X::Array{Float64}, T::Array{Float64, 1})
    X = Helpers.check_ones_col(X)
    h(X, T)
end
# Run theta parameters on new entry with normalization
# Input
#   X   ->  Input data
#   T   ->  Theta parameters
#   NP  ->  Normalization parameters
# Output
#   Hypothesis of X with T
function predict(X::Array{Float64}, T::Array{Float64, 1}, NP::FeatureScaling.Params)
    X = Helpers.check_ones_col(X, T)
    Xcopy = copy(X)
    FeatureScaling.fnormalize!(Xcopy, NP)
    h(Xcopy, T)
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
    Y = g(X * T)
end

# Sigmoid function for logistic regression
function g(z)
    1 ./ (1 + e.^-z)
end

end
