module FeatureScaling

export fnormalize!

# Feature scaling for matrix with mean normalization
# Input:
#   X -> Matrix of input data set
# Output:
#   mu -> Vector of mean of each feature
#   sd -> Vector of standard deviation of each feature
function fnormalize!(X::Array{Float64, 2})
    m = size(X, 1) # Number of input entries
    n = size(X, 2) # Number of features
    mu = zeros(n) # Vector of feature's average
    sd = zeros(n) # Vector of feature's standard deviation
    for i = 1:n
        mu[i] = mean(X[:,i])
        sd[i] = std(X[:,i])
        if sd[i] != 0.0
            X[:,i] = (X[:,i] - mu[i]) / sd[i]
        end
    end
    return mu, sd
end

# Feature scaling with normalization parameters known
# Input:
#   x -> Vector of input data
#   mu -> Vector of mean for each feature
#   sd -> Vector of standard deviation for each feature

# Vector normalization
function fnormalize!(x::Array{Float64,1}, mu::Array{Float64,1}, sd::Array{Float64,1})
    for i = 1:length(x)
        if sd[i] != 0.0
            x[i] = (x[i] - mu[i]) / sd[i]
        end
    end
end
# Matrix normalization
function fnormalize!(X::Array{Float64,2}, mu::Array{Float64,1}, sd::Array{Float64,1})
    for i = 1:size(X,2)
        if sd[i] != 0.0
            X[:,i] = (X[:,i] - mu[i]) / sd[i]
        end
    end
end

end
