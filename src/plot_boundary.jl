module PlotBoundary

using StatPlots; plotly()

export plot_b!


function plot_b!(X::Array{Float64,2}, T::Array{Float64,1})
    if size(X, 2) <= 3
        plot_x = [minimum(X[:,2])-2, maximum(X[:,2])+2]
        plot_y = (-1 ./ T[3]) .* (T[2] .* plot_x + T[1])
        display(plot!(plot_x, plot_y))
    end
end

end
