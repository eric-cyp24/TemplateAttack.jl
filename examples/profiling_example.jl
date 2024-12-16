using ArgParse
using TemplateAttack
using TemplateAttack:loaddata


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "labels"
            help = "intermediate values (.npy file)"
            required = true
        "traces"
            help = "traces (.npy file)"
            required = true
        "--outdir"
            default = "Templates/"
        "--numofcomponents"
            arg_type = Int
            default = 0
        "--nicv_th"
            help = ""
            arg_type = Float64
            default = 0.1
            range_tester = x->0.0<x<1.0
        "--POIe_left"
            help = "number of samples for the POI extension to the left (prior)."
            arg_type = Int
            default = 0
        "--POIe_right"
            help = "number of samples for the POI extension to the right (afterward)."
            arg_type = Int
            default = 0
    end
    return parse_args(s)
end



function main()
    # parse arguments
    args = parse_commandline()

    # load data
    println("loading data...      ")
    Traces = loaddata(args["traces"])
    IVs    = loaddata(args["labels"])
    IVs    = ndims(IVs)==1 ? reshape(IVs,:,1) : IVs

    OUTDIR = joinpath(dirname(args["traces"]), args["outdir"])
    numofcomponents = args["numofcomponents"]
    nicv_th = args["nicv_th"]
    POIe_left  = args["POIe_left"]
    POIe_right = args["POIe_right"]

    # setup outputs
    ispath(OUTDIR) || mkpath(OUTDIR)
    templatefilename = "Templates_"*splitext(basename(args["labels"]))[1]*"_nicv$(string(nicv_th)[2:end])_POIe$(POIe_left)-$(POIe_right).h5"
    outfile = joinpath(OUTDIR, templatefilename)
    println("writing template file: $outfile")
    
    # run profiling
    println("profiling...        ")
    Templates = runprofiling(IVs, Traces; nicv_th, POIe_left, POIe_right, numofcomponents, outfile)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
