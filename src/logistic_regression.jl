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

function train(X::Array{Float64,2}, Y::Array{Float64,1};
        alpha::Float64 = 0.01, epsilon::Float64 = 0.0001,
        regularization::Float64 = 0.0, max_its::Int64 = 5000,
        plot_cost = true)
    X = Helpers.check_ones_col(X)
    Xcopy = copy(X)
    norm_params = FeatureScaling.fnormalize!(Xcopy)
    thetas = GradientDescent.gradient_descent(Xcopy, Y, h, J!,
        config = GradientDescent.GDConfig(alpha, epsilon, regularization, max_its),
        plot_cost = plot_cost)
    (thetas, norm_params)
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
#   G -> Gradients vector to edit
#   reg -> Regularization parameter
function J!(X::Array{Float64,2}, Y::Array{Float64,1}, T::Array{Float64,1}; G = [], reg::Number = 0)
    cost = 1/size(X, 1) * (-Y' * log.(h(X, T)) - (1 - Y)' * log.(1 - h(X, T)))
    if !isempty(G)
        for i = 1:length(G)
            G[i] = 1/size(X,1) * sum((h(X, T) - Y) .* X[:,i])
        end
    end
    return cost
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
