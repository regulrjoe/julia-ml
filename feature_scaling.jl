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
function fnormalize!(X::Array{Float64, 2})
    m = size(X, 1) # Number of input entries
    n = size(X, 2) # Number of features
    params = Params(zeros(n), zeros(n))
    for i = 1:n
        params.mu[i] = mean(X[:,i])
        params.sd[i] = std(X[:,i])
        if params.sd[i] != 0.0
            X[:,i] = (X[:,i] - params.mu[i]) / params.sd[i]
        end
    end
    return params
end

# Feature scaling with normalization parameters known
# Input:
#   x -> Vector of input data
#   mu -> Vector of mean for each feature
#   sd -> Vector of standard deviation for each feature

# Vector normalization
function fnormalize!(x::Array{Float64,1}, params::Params)
    for i = 1:length(x)
        if params.sd[i] != 0.0
            x[i] = (x[i] - params.mu[i]) / params.sd[i]
        end
    end
end
# Matrix normalization
function fnormalize!(X::Array{Float64,2}, params::Params)
    for i = 1:size(X,2)
        if params.sd[i] != 0.0
            X[:,i] = (X[:,i] - params.mu[i]) / params.sd[i]
        end
    end
end

end
