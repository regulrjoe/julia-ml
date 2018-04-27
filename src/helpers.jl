module Helpers

export check_ones_col, check_matrix_float64

# Check if matrix's has ones column, if not, add ones column
function check_ones_col(X)
    if !has_ones_col(X)
        X = add_ones_col(X)
    end
    X
end

# Check X matrix width against θ vector length, if not equal, add ones column
function check_ones_col(X, T)
    if !have_equal_n_cols(X, T')
        X = add_ones_col(X)
    end
    X
end

function check_matrix_float64(X)
    if !is_matrix_float64(X)
        X = convert_matrix_float64(X)
    end
    X
end

# Check if input object is a Float64 Matrix
is_matrix_float64(X) = typeof(X) == Array{Float64, 2}

# Convert object to Float64 Matrix
convert_matrix_float64(X) = X = map(x->convert(Float64, x), X)

# Check if matrix's first column is all 1's
has_ones_col(X) = X[:,1] == ones(size(X,1))

# Add column of 1's at beginning of matrix
add_ones_col(X::Array) = reshape([ones(size(X,1)) X], size(X, 1), size(X, 2)+1)

# Check if both arrays have equal number of columns
have_equal_n_cols(X1, X2) = size(X1,2) == size(X2, 2)

end
