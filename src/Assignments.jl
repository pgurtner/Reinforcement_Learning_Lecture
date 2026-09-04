module Assignments

module Utils
using CairoMakie

function make_GL_plot(title)
    fig = Figure()
    ax = Axis(fig[1, 1], title = title)
    return fig, ax
end
end

include("assignment1.jl")
include("assignment2.jl")

T8.final()
end