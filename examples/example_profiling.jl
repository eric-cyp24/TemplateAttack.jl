using TemplateAttack
using TemplateAttack:loaddata


function Kyber768_profiling(INDIR, OUTDIR, Traces, X, Y, S, Buf; 
                            POIe_left=80, POIe_right=20, nicv_th=0.001, buf_nicv_th=0.004)
    numofcomponents, priors = 3, :uniform
    fn = "Templates_X_proc_nicv$(string(nicv_th)[2:end])_POIe$(POIe_left)-$(POIe_right)_lanczos2.h5"
    outfile = joinpath(OUTDIR, fn)
    println("profiling for X: $outfile")
    Templates_X = runprofiling( X, Traces; nicv_th, POIe_left, POIe_right, 
                                           priors, numofcomponents, outfile);

    numofcomponents, priors = 3, :uniform
    fn = "Templates_Y_proc_nicv$(string(nicv_th)[2:end])_POIe$(POIe_left)-$(POIe_right)_lanczos2.h5"
    outfile = joinpath(OUTDIR, fn)
    println("profiling for Y: $outfile")
    Templates_Y = runprofiling( Y, Traces; nicv_th, POIe_left, POIe_right, 
                                           priors, numofcomponents, outfile);

    numofcomponents, priors = 4, :binomial
    fn = "Templates_S_proc_nicv$(string(nicv_th)[2:end])_POIe$(POIe_left)-$(POIe_right)_lanczos2.h5"
    outfile = joinpath(OUTDIR, fn)
    println("profiling for S: $outfile")
    Templates_S = runprofiling( S, Traces; nicv_th, POIe_left, POIe_right, 
                                           priors, numofcomponents, outfile);

    numofcomponents, priors = 16, :uniform
    fn = "Templates_Buf_proc_nicv$(string(buf_nicv_th)[2:end])_POIe$(POIe_left)-$(POIe_right)_lanczos2.h5"
    outfile = joinpath(OUTDIR, fn)
    println("profiling for Buf: $outfile")
    Templates_Buf = runprofiling( Buf, Traces; nicv_th=buf_nicv_th, POIe_left, POIe_right, 
                                               priors, numofcomponents, outfile);
end

INDIR  = joinpath(@__DIR__, "../data/Traces/", "SOCKET_HPF/DK1/test_20240724/lanczos2_25/")
OUTDIR = joinpath(INDIR,"Outputs")
Traces = loaddata( joinpath(INDIR, "traces_lanczos2_25_proc.npy") )
X      = loaddata( joinpath(INDIR, "X_proc.npy")                  )
Y      = loaddata( joinpath(INDIR, "Y_proc.npy")                  )
S      = loaddata( joinpath(INDIR, "S_proc.npy")                  )
Buf    = loaddata( joinpath(INDIR, "Buf_proc.npy")                )
Kyber768_profiling(INDIR, OUTDIR, Traces, X, Y, S, Buf; POIe_left, POIe_right, nicv_th, buf_nicv_th)




