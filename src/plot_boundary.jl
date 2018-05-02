module PlotBoundary

include("feature_mapping.jl")
using StatPlots; plotly()

export plot_b!


function plot_b!(X::Array{Float64,2}, T::Array{Float64,1})
    if size(X, 2) <= 3
        plot_x = [minimum(X[:,2])-2, maximum(X[:,2])+2]
        plot_y = (-1 ./ T[3]) .* (T[2] .* plot_x + T[1])
        display(plot!(plot_x, plot_y))
    else
        fm = FeatureMapping
        u = linspace(-1, 1.5, 50)
        v = linspace(-1, 1.5, 50)

        z = zeros(length(u), length(v))
        # Evaluate z = theta * x over the grid
        for i = 1:length(u)
            for j = 1:length(v)
                z[i,j] = (fm.map2features(u[i], v[j], 6) * T)[1]
            end
        end
        z = z'

        # Plot z = 0
        # Specify range [0, 0]
        display(contour(u, v, z))
    end
end

end
