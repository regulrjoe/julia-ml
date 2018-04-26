push!(LOAD_PATH, pwd())

include("linear_regression.jl")
include("helpers.jl")

using CSV, DataFrames, StatPlots

# Data reading
data = CSV.read("machine-learning-ex1/ex1/ex1data2.txt", datarow = 1)
X = convert(Array{Float64}, data[:, 1:size(data,2)-1]) # First to second-to-last columns
Y = convert(Array{Float64}, data[:, size(data,2)]) # Last column

# Data manipulation
X = Helpers.check_matrix_float64(X)
X = Helpers.check_ones_col(X)

lr = LinearRegression

# Linear regression
@time output = lr.train(X, Y)

if size(output, 1) == 1
    Y2 = lr.run(X, output[1])
elseif size(output, 1) == 2
    Y2 = lr.run(X, output[1], output[2])
end
