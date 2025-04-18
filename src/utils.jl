

"""
    plotdatascatter!(data::AbstractMatrix; axes=[1,2], show=true, groupdict=nothing, colors=nothing, kwarg...)

Scatterplot of the data along the given two axes.
"""
function plotdatascatter(data::AbstractMatrix; axes=[1,2], show=true, groupdict=nothing, colors=nothing, kwarg...)
    plot(); plotdatascatter!(data; axes, show, groupdict, colors, kwarg...)
end
function plotdatascatter!(data::AbstractMatrix; axes=[1,2], show=true, groupdict=nothing, colors=nothing, kwarg...)
    if isnothing(groupdict)
        x,y = eachrow(view(data,axes,:))
        p = scatter!(x, y; label="", markersize=1, markeralpha=0.7, markerstrokewidth=-1, color=:grey, kwarg...)
    else
        colors = isnothing(colors) ? (1:length(groupdict)) : colors
        labels = sort(collect(keys(groupdict)))
        colordict = (colors isa Dict) ? colors : Dict(l=>Int(c) for (l,c) in zip(labels,colors))
        for (l,c) in sort(colordict)
            x,y = eachrow(view(data,axes,groupdict[l]))
            p = scatter!(x, y; label="", markersize=1, markeralpha=0.7, markerstrokewidth=-1, color=c, kwarg...)
        end
    end
    show && gui()
    return plot!()
end

"""
    plotmvg!(mvg::MvNormal; axes=[1,2], linecolor=1, kwarg...)

Plot the 95% and 99% confidence ellipse for a given Gaussian distribution.
"""
function plotmvg(mvg::MvNormal; axes=[1,2], linecolor=1, kwarg...)
    plot(); plotmvg!(mvg; axes, linecolor, kwarg...)
end
function plotmvg!(mvg::MvNormal; axes=[1,2], linecolor=1, kwarg...)
    loc, cov = mvg.μ[axes], mvg.Σ[axes,axes]
    # n_std_prob: 90%:2.146, 95%:2.448, 99%3.035, 99.5%: 3.255 (df=2)
    #covellipse!(loc,cov; n_std=2.146, linecolor, linealpha=1.0, fillalpha=0, kwarg...)
    covellipse!(loc,cov; n_std=2.448, linecolor, linealpha=1.0, fillalpha=0, label="", kwarg...)
    covellipse!(loc,cov; n_std=3.035, linecolor, linealpha=0.5, fillalpha=0, label="", kwarg...)
    gui()
    return plot!()
end

"""
    plotTemplate!(t::Template; axes=[1,2], show=true, colors=nothing, kwarg...)

Plot all the MVG ellipse on plane `axes`
"""
function plotTemplate(t::Template; axes=[1,2], show=true, colors=nothing, kwarg...)
    plot(); plotTemplate!(t; axes, show, colors, kwarg...)
end
function plotTemplate!(t::Template; axes=[1,2], show=true, colors=nothing, kwarg...)
    colors = isnothing(colors) ? (1:length(t.mvgs)) : colors
    labels = sort(collect(keys(t.mvgs)))
    colordict = (colors isa Dict) ? colors : Dict(l=>Number(c) for (l,c) in zip(labels,colors))
    for (l,c) in sort(colordict)
        loc, cov = t.mvgs[l].μ[axes], t.mvgs[l].Σ[axes,axes]
        covellipse!(loc, cov; n_std=2.448, fillalpha=0, linealpha=1,
                    linewidth=1, linecolor=c, label=l, kwarg...)
        covellipse!(loc, cov; n_std=3.035, fillalpha=0, linealpha=0.5,
                    linewidth=0.7, linecolor=c, kwarg..., label="")
    end
    show && gui()
    return plot!()
end


"""
    plotLDA(t::Template; num_components=4, show=true, kwarg...)

Plot LDA vectors of the template
"""
function plotLDA(t::Template; num_components=4, show=true, kwarg...)
    num_components = min(num_components, ndims(t))
    tr   = plot(t.TraceMean;color=:blue)
    ldas = [plot(view(t.ProjMatrix,:,i);color=i) for i in 1:num_components]
    p    = plot(tr,ldas...; layout=(num_components+1,1), legend=false, size=(1600,900), kwarg...)
    show && gui()
    return p
end

##### load/write data #########

"""
    loaddata(filename::AbstractString, datapath::AbstractString="data")

Load from .h5 file.
"""
function loaddata(filename::AbstractString; datapath="data")
    if !isfile(filename)
        error("$filename doesn't exist!!")
    else
        if split(filename,".")[end] == "h5" && HDF5.ishdf5(filename)
            return h5open(filename) do f
                dset  = f[datapath]
                dsize = length(dset)*sizeof(HDF5.get_jl_type(dset))
                # if file size > 2^28 ≈ 256 MB, then use memmap
                return dsize > 2^28 && HDF5.ismmappable(dset) ?
                       HDF5.readmmap(dset) : read(dset)
            end
        else
            error("$filename is neither a .h5 file!!")
        end
    end
end

"""
    writedata(filename::AbstractString, data; datapath::AbstractString="data", overwrite::Bool=true)

Write `data` to the given `filename` and `datapath`.
The default `overwrite=true` will overwrite an existing file.
Use `overwrite=false` to create a new file if not existing and preserve existing content.
"""
function writedata(filename::AbstractString, data; datapath::AbstractString="data", overwrite::Bool=true)
    filename *= split(filename,".")[end] == "h5" ? "" : ".h5"
    h5open(filename, overwrite ? "w" : "cw") do f
        write(f,datapath,data)
    end
end





