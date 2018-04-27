include("logistic_regression.jl")
include("helpers.jl")
include("feature_scaling.jl")
include("plot_boundary.jl")

using CSV, DataFrames, StatPlots
plotly()

# Data reading
data = CSV.read("machine-learning-ex2/ex2/ex2data1.txt", datarow = 1)
X = convert(Array{Float64}, data[:, 1:end-1]) # First to second-to-last columns
Y = convert(Array{Float64}, data[:, end]) # Last column
Positives = data[data[:, :Column3] .== 1, :]
Negatives = data[data[:, :Column3] .== 0, :]

# Data manipulation
X = Helpers.check_matrix_float64(X)
X = Helpers.check_ones_col(X)

lr = LogisticRegression

# Linear regression
@time output = lr.train(X, Y, alpha = 0.3, max_its = 2000, epsilon = 0.000001)

println("Thetas: ", output[1])
Y2 = lr.predict(
    [1.0; 45.0; 85.0],
    output[1],
    output[2]
)

# Plot data
XP = Helpers.check_ones_col(convert(Array{Float64}, Positives[:,1:end-1]))
XN = Helpers.check_ones_col(convert(Array{Float64}, Negatives[:,1:end-1]))
FeatureScaling.fnormalize!(X, FeatureScaling.Params(output[2].mu, output[2].sd))
FeatureScaling.fnormalize!(XP, FeatureScaling.Params(output[2].mu, output[2].sd))
FeatureScaling.fnormalize!(XN, FeatureScaling.Params(output[2].mu, output[2].sd))
scatter(XP[:,2], XP[:,3], colour = :blue)
scatter!(XN[:,2], XN[:,3], colour = :red)
PlotBoundary.plot_b!(X, output[1])
