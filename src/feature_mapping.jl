module FeatureMapping


# Map two features to quadratic features.
# Input:
#   F1  -> Feature 1
#   F2  -> Feature 2
#   degree ->   Degree of polynomial
function map2features(X1, X2, degree)
    assert(size(X1, 1) == size(X2, 1)) # X1 and X2 must be of same size
    out = ones(size(X1, 1), sum(1:degree+1))
    c = 1 # Column
    for i = 1:degree
        for j = 0:i
            out[:,c+=1] = (X1 .^ (i - j)) .* (X2 .^ j)
        end
    end
    return out
end

end
