push!(LOAD_PATH, pwd())

include("logistic_regression.jl")
include("helpers.jl")

using CSV, DataFrames, StatPlots
gr(size = (400, 400))

# Data reading
data = CSV.read("machine-learning-ex2/ex2/ex2data1.txt", datarow = 1)
X = convert(Array{Float64}, data[:, 1:end-1]) # First to second-to-last columns
Y = convert(Array{Float64}, data[:, end]) # Last column
Positives = data[data[:, :Column3] .== 1, :]
Negatives = data[data[:, :Column3] .== 0, :]

# Plot data
@df Positives scatter(:Column1, :Column2, colour = :blue)
@df Negatives scatter!(:Column1, :Column2, colour = :red)

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
