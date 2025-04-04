




wavecycle = [0., 1., .5, .25, .1, .05, .025, .001]

function vals2wave(vals::AbstractVector; dc_bias=.0, noise=.0, powermodel=count_ones)
    wave = dc_bias .+ randn(length(vals)*4+4).*noise
    for (i,v) in enumerate(vals)
        wave[i*4-3:i*4+4] .+= powermodel(v)*(wavecycle+randn(8)*noise)
    end
    return wave
end

function gendataset(N=25600, numbytes=4; value_range=0:255, dc_bias=.0, noise=0.01)
    IVs    = rand(value_range,(numbytes,N))
    traces = stack(vals2wave.(eachcol(IVs); dc_bias, noise))
    return IVs, traces
end





