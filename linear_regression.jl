module LinearRegression

include("feature_scaling.jl")
include("gradient_descent.jl")
include("helpers.jl")

export train

# Gradient Descent Configuration
mutable struct GDConfig
    alpha::Float64
    epsilon::Float64
    max_its::Int64
end
# Output Linear Regression Parameters
mutable struct Parameters
    thetas::Array{Float64,1}
    means::Array{Float64,1}
    sdevs::Array{Float64,1}
end

gdconf = GDConfig(0.01, 0.0001, 5000)
params = Parameters([],[],[])

# Linear Regression Algorithm with Gradient Descent
# Input:
#   X -> Matrix of input data set
#   Y -> Vector of output of training set
#   alpha -> Gradient descent's step size
#   epsilon -> Gradient descent's minimum difference between iterations
#   max_iterations -> Gradient descent's maximum steps before ending iteration
# Output:
#   optimum paramaters theta
function train(X::Array{Float64,2}, Y::Array{Float64,1})
    X = Helpers.check_ones_col(X)
    if size(X, 2) < 10^4
        global params.thetas = normal_equation(X, Y)
    else
        Xcopy = copy(X)
        norm_params = FeatureScaling.fnormalize!(Xcopy)
        global params.means = norm_params[1]
        global params.sdevs = norm_params[2]
        thetas = zeros(size(Xcopy,2))
        global params.thetas = GradientDescent.gradient_descent!(Xcopy, Y, h, J,
            a = gdconf.alpha,
            u = gdconf.epsilon,
            max_iterations = gdconf.max_its)[1]
    end
end

function run(X::Array{Float64})
    X = Helpers.check_ones_col(X)
    if (!isempty(params.means) && !isempty(params.sdevs))
        FeatureScaling.fnormalize!(X, params.means, params.sdevs)
    end
    return h(X, params.thetas)
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

# Cost function
# Input:
#   X -> Matrix of input data set
#   Y -> Vector of output of training set
#   T -> Vector of parameters to evaluate
#   h -> Hypothesis function to evaluate
function J(X::Array{Float64,2}, Y::Array{Float64,1}, T::Array{Float64,1})
    m = size(X,1)
    sqrd_error = sum((h(X, T) - Y) .^ 2)
    sqrd_error / 2m
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
