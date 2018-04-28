module GradientDescent

using Plots; gr()

# Gradient Descent Configuration
mutable struct GDConfig{F<:Float64, I<:Int64}
    alpha::F # Step size paramater
    epsilon::F # Minimum error parameter
    reg::F # Regularization parameter
    max_its::I # Maximum number of iterations
end

# Batch Gradient Descent function with Vectorized implementation
# θ := θ − α/m X'(h(Xθ) − Y)
# Input:
#   X -> Matrix of input data set
#   Y -> Vector of output of training set
#   h -> Hypothesis function
#   J -> Cost function
#   a -> Step size alpha
#   u -> Minimum difference between iterations, epsilon
#   max_iterations -> Maximum steps before ending iteration
# Output:
#   T -> Vector of optimized parameters theta
function gradient_descent(X::Array{Float64, 2}, Y::Array{Float64,1}, h::Function, Cost::Function;
                            config::GDConfig = GDConfig(0.01, 0.0001, 0.0, 5000), plot_cost = false)
    m = size(X, 1) # Number of input entries
    n = size(X, 2) # Number of input features
    T = zeros(n)
    temp = Array{Float64}(zeros(n)) # Temp vector of new parameters to evaluate
    jval = 0 # Current iteration's cost function and Last iteration's cost function
    jvals = Array{Float64}(0) # Overtime change in cost
    it = 1
    for it = 1:config.max_its
        if config.reg != 0
            T = T - (1 - config.alpha * (config.reg / m)) - (config.alpha / m) * X' * (h(X, T) - Y)
        else
            T = T - (config.alpha / m) * X' * (h(X, T) - Y)
        end
        jval = Cost(X, Y, T, reg = config.reg)
        append!(jvals, jval)
        if it > 1
            if abs(jvals[length(jvals) - 1] - jval) < config.epsilon
                break
            end
        end
    end
    if plot_cost
        plot_cost_ot(jvals)
    end
    return (T, [jval, it])
end

function plot_cost_ot(j_vals::Array{Float64, 1})
    display(plot(j_vals, linewidth = 3, title = "Cost Overtime", xlabel = "Iteration", ylabel = "Cost"))
end

function plot_cost(X::Array, Y::Array; min_theta::Number = -10, max_theta::Number = 10, h::Function = Hypothesis.linr)
    theta0_vals = collect(linspace(min_theta, max_theta, 100))
    theta1_vals = collect(linspace(min_theta, max_theta, 100))
    J_vals = Array{Float64}(length(theta0_vals), length(theta1_vals))
    for i = 1:length(theta0_vals)
        for j = 1:length(theta1_vals)
            J_vals[i,j] = Cost.J(X, Y, [theta0_vals[i], theta1_vals[j]], h)
        end
    end
    display(contour(theta0_vals, theta1_vals, J_vals))
end

end
