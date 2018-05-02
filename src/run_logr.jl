include("feature_mapping.jl")
include("feature_scaling.jl")
include("helpers.jl")
include("logistic_regression.jl")
include("plot_boundary.jl")

using CSV, DataFrames, StatPlots
plotly()

# Data reading
data = CSV.read("machine-learning-ex2/ex2/ex2data2.txt", datarow = 1)
X = convert(Array{Float64}, data[:, 1:end-1]) # First to second-to-last columns
Y = convert(Array{Float64}, data[:, end]) # Last column
Positives = data[data[:, :Column3] .== 1, :]
Negatives = data[data[:, :Column3] .== 0, :]

# Data manipulation
X = Helpers.check_matrix_float64(X)
# X = FeatureMapping.map2features(X[:,1], X[:,2], 6)
X = Helpers.check_ones_col(X)

lr = LogisticRegression

# Linear regression
@time output = lr.train(X, Y, alpha = 0.3, max_its = 100000, epsilon = 0.00001, lambda = 1.0)

println("Thetas: ", output[1])

Y2 = lr.predict(
    [1.0; 45.0; 85.0],
    output[1],
    output[2]
)

# Plot data
XP = Helpers.check_ones_col(convert(Array{Float64}, Positives[:,1:end-1]))
XN = Helpers.check_ones_col(convert(Array{Float64}, Negatives[:,1:end-1]))
FeatureScaling.fnormalize!(X, output[2])
FeatureScaling.fnormalize!(XP, output[2])
FeatureScaling.fnormalize!(XN, output[2])
scatter(XP[:,2], XP[:,3], colour = :blue)
scatter!(XN[:,2], XN[:,3], colour = :red)
PlotBoundary.plot_b!(X, output[1])
