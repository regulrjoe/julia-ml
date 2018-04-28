module FeatureScaling

export fnormalize!, NormalizationParams

# Normalization Parameters
mutable struct Params{A<:Array{Float64,1}}
    mu::A # means for normalization
    sd::A # standard deviations for normalization
end

# Feature scaling for matrix with mean normalization
# Input:
#   X -> Matrix of input data set
# Output:
#   params -> Normalization Parameters
function fnormalize!(X::Array{Float64})
    m = size(X, 1) # Number of input entries
    n = size(X, 2) # Number of features
    mu = zeros(n)
    sd = zeros(n)
    for i = 1:n
        mu[i] = mean(X[:,i])
        sd[i] = std(X[:,i])
        if sd[i] != 0.0
            X[:,i] = (X[:,i] - mu[i]) / sd[i]
        end
    end
    return Dict("mu" => mu, "sd" => sd)
end

# Feature scaling with normalization parameters known
# Input:
#   x -> Vector of input data
#   mu -> Vector of mean for each feature
#   sd -> Vector of standard deviation for each feature

# Vector normalization
function fnormalize!(x::Array{Float64,1}, params::Dict{String, Array{Float64,1}})
    for i = 1:length(x)
        if params["sd"][i] != 0.0
            x[i] = (x[i] - params["mu"][i]) / params["sd"][i]
        end
    end
    return nothing
end
# Matrix normalization
function fnormalize!(X::Array{Float64,2}, params::Dict{String, Array{Float64,1}})
    for i = 1:size(X,2)
        if params["sd"][i] != 0.0
            X[:,i] = (X[:,i] - params["mu"][i]) / params["sd"][i]
        end
    end
    return nothing
end

end
