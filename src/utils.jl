


"""
    plotdatascatter!(data::Matrix{T}; axes=[1,2], show=true, kwarg...) where T<:AbstractFloat

scatterplot of the data along the given two axes.
"""
function plotdatascatter!(data::Matrix{T}; axes=[1,2], show=true, kwarg...) where T<:AbstractFloat
    x,y = eachrow(data[axes,:])
    scatter!(x, y; label="", markersize=1, markeralpha=0.7, markerstrokewidth=-1, color=:grey, kwarg...)
    show && gui()
end

"""
    plotMvNormal!(mvg::MvNormal; axes=[1,2], linecolor=1, kwarg...)

plot the 90%, 95%, and 99% confidence ellipse for a given Gaussian distribution.
"""
function plottemplate!(mvg::MvNormal; axes=[1,2], linecolor=1, kwarg...)
    loc, cov = mvg.μ[axes], mvg.Σ[axes,axes]
    # n_std_prob: 90%:2.146, 95%:2.448, 99%3.035, 99.5%: 3.255 (df=2)
    covellipse!(loc,cov; n_std=2.146, linecolor, linealpha=1.0, fillalpha=0, kwarg...)
    covellipse!(loc,cov; n_std=2.448, linecolor, linealpha=0.5, fillalpha=0, kwarg...)
    covellipse!(loc,cov; n_std=3.035, linecolor, linealpha=0.2, fillalpha=0, kwarg...)
    gui()
end


function plotTemplate!(t::Template; axes=[1,2], show=true, colors=nothing, kwarg...)
    colors = isnothing(colors) ? (1:length(t.labels)) : colors
    for i in 1:length(t.labels)
        loc, cov = t.mus[i][axes], t.sigmas[i][axes,axes]
        covellipse!(loc, cov; n_std=2.448, fillalpha=0, linealpha=1,
                    linesidth=1, linecolor=colors[i], label=t.labels[i], kwarg...)
        covellipse!(loc, cov; n_std=3.035, fillalpha=0, linealpha=0.5,
                    linesidth=0.7, linecolor=colors[i], kwarg..., label="")
    end
    show && gui()
end







