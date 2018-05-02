module LinearRegression

include("feature_scaling.jl")
include("gradient_descent.jl")
include("helpers.jl")

export train, predict

# Linear Regression Algorithm with Normal Equation or Gradient Descent
# Input:
#   X -> Matrix of input data set
#   Y -> Vector of output of training set
#   alpha -> Step size parameter for gradient descent
#   epsilon -> Minimum error parameter for gradient descent
#   regularization -> Regularization parameter for gradient descent
#   max_its -> Maximum iterations paramter for gradient descent
#   plot_cost -> Plot overtime change in cost bool.
# Output:
#   Optimum paramaters theta
#   Normalization parameters
function train(X::Array{Float64,2}, Y::Array{Float64,1};
        alpha::Float64 = 0.01, epsilon::Float64 = 0.0001,
        lambda::Float64 = 0.0, max_its::Int64 = 5000,
        plot_cost = false)
    X = Helpers.check_ones_col(X)
    if size(X, 2) < 10^4
        thetas = lambda != 0 ?
            normal_equation(X, Y) :
            normal_equation(X, Y, lambda)
        return Tuple([thetas])
    else
        Xcopy = copy(X)
        norm_params = FeatureScaling.fnormalize!(Xcopy)
        gd_output = GradientDescent.gradient_descent(Xcopy, Y, h, J,
            config = GradientDescent.GDConfig(alpha, epsilon, lambda, max_its),
            plot_cost = plot_cost)
        println("Cost: ", gd_output[2][1], "\nIterations: ", gd_output[2][2])
        return (gd_output[1], norm_params)
    end
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
function predict(X::Array{Float64}, T::Array{Float64, 1}, NP::Dict{String, Array{Float64, 1}})
    X = Helpers.check_ones_col(X, T)
    Xcopy = copy(X)
    FeatureScaling.fnormalize!(Xcopy, NP)
    h(Xcopy, T)
end

# Normal Equation (use when n features < 10^6)
# Input:
#   X -> Matrix of input data set
#   Y -> Vector of output of training set
# Output:
#   Optimized parameters theta
function normal_equation(X::Array{Float64,2}, Y::Array{Float64,1})
    inv(X' * X) * X' * Y
end

# Normal Equation with regularization
function normal_equation(X::Array{Float64,2}, Y::Array{Float64,1}, lambda::Float64)
    n = size(X,2)
    L = eye(n); L[1,1] = 0
    inv(X' * X + lambda * L) * X' * Y
end
# Cost function
# ∑((h - Y)^2) / 2m
# Input:
#   X -> Matrix of input data set
#   Y -> Vector of output of training set
#   T -> Vector of parameters to evaluate
#   reg -> Regularization parameters
function J(X::Array{Float64,2}, Y::Array{Float64,1}, T::Array{Float64,1}; lambda::Number = 0)
    (sum((h(X, T) - Y) .^ 2) + (lambda * sum(T[2:end].^2))) / (2 * size(X,1))
end

# Linear regression hypothesis function
# (θ[0] + θ[1]X[1] + θ[2]X[2] + ... + θ[n]X[n])
# Input:
#   X -> nxm Matrix of input data set
#   T -> n Vector of parameters to evaluate

# Applied to vector
function h(x::Array{Float64,1}, T::Array{Float64,1})
    T' * x
end
# Applied to matrix
function h(X::Array{Float64,2}, T::Array{Float64,1})
    X * T
end

end
