module GradientDescent

using Plots; gr()

# Overtime change in cost
J_Overtime = Array{Float64}(0)

# Batch Gradient Descent function for Linear Regression
# Input:
#   X -> Matrix of input data set
#   Y -> Vector of output of training set
#   T -> Vector of parameters theta
#   a -> Step size alpha
#   u -> Minimum difference between iterations, epsilon
#   max_iterations -> Maximum steps before ending iteration
# Output:
#   T -> Vector of optimized parameters theta
#   JOvertime -> Vector of Overtime changes of J(θ)
function gradient_descent(X::Array{Float64, 2}, Y::Array{Float64,1}, h::Function, Cost::Function; a::Float64 = 0.003, u::Float64 = 0.0001, max_iterations::Int64 = 50000, doplot = true)
    m = size(X, 1) # Number of input entries
    n = size(X, 2) # Number of input features
    T = zeros(n)
    temp = Array{Float64}(zeros(n)) # Temp vector of new parameters to evaluate
    J = 0 # Current iteration's cost function and Last iteration's cost function
    for it = 1:max_iterations
        for i = 1:n
            temp[i] = T[i] - a * 1/m * sum((h(X, T) - Y) .* X[:,i])
        end
        for i = 1:n
            T[i] = temp[i]
        end
        J = Cost(X, Y, T)
        J_Overtime = reshape([JOvertime; J], length(JOvertime) + 1, 1)
        println(it, ": ", J)
        if abs(J_Overtime[end] - J) < u
            break
        end
    end
    if doplot
        plot_cost_ot()
    end
    return T
end

function plot_cost_ot()
    display(plot(J_Overtime, linewidth = 3, title = "Cost Overtime", xlabel = "Iteration", ylabel = "Cost"))
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
