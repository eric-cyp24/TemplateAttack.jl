using Dates:Time,Second
using Mmap, HDF5
using StatsBase:sample
using TemplateAttack
using TemplateAttack:loaddata

### Parameters ########
include("Parameters.jl")
ntraces   = 192000
trlen     = 10920
ntest     = 500
TracesDIR = scratchTracesDIR
###

Dir = DirHPFnew
POIe_left, POIe_right = 40, 80
nicv_th, buf_nicv_th  = 0.001, 0.004
#######################


srcDir(dev) = joinpath(TracesDIR, Dir[dev],"lanczos2_25/")
pooledDir(devices)=joinpath("SOCKET_HPF/Pooled/Pooled_HPF/", join(sort(String.(devices)),"_")*"/")

function Kyber768_profiling(INDIR, OUTDIR, Traces, X, Y, S, Buf; nvalid=nothing, 
                            POIe_left=80, POIe_right=20, nicv_th=0.001, buf_nicv_th=0.004)
    numofcomponents, priors = 3, :uniform
    fn = "Templates_X_proc_nicv$(string(nicv_th)[2:end])_POIe$(POIe_left)-$(POIe_right)_lanczos2.h5"
    outfile = joinpath(OUTDIR, fn)
    println("profiling for X: $outfile")
    Templates_X = runprofiling( X, Traces; nicv_th, POIe_left, POIe_right, 
                                           priors, numofcomponents, outfile, nvalid);

    numofcomponents, priors = 3, :uniform
    fn = "Templates_Y_proc_nicv$(string(nicv_th)[2:end])_POIe$(POIe_left)-$(POIe_right)_lanczos2.h5"
    outfile = joinpath(OUTDIR, fn)
    println("profiling for Y: $outfile")
    Templates_Y = runprofiling( Y, Traces; nicv_th, POIe_left, POIe_right, 
                                           priors, numofcomponents, outfile, nvalid);

    numofcomponents, priors = 4, :binomial
    fn = "Templates_S_proc_nicv$(string(nicv_th)[2:end])_POIe$(POIe_left)-$(POIe_right)_lanczos2.h5"
    outfile = joinpath(OUTDIR, fn)
    println("profiling for S: $outfile")
    Templates_S = runprofiling( S, Traces; nicv_th, POIe_left, POIe_right, 
                                           priors, numofcomponents, outfile, nvalid);

    numofcomponents, priors = 16, :uniform
    fn = "Templates_Buf_proc_nicv$(string(buf_nicv_th)[2:end])_POIe$(POIe_left)-$(POIe_right)_lanczos2.h5"
    outfile = joinpath(OUTDIR, fn)
    println("profiling for Buf: $outfile")
    Templates_Buf = runprofiling( Buf, Traces; nicv_th=buf_nicv_th, POIe_left, POIe_right, 
                                               priors, numofcomponents, outfile, nvalid);
end

function pooledTraces(MixDIR, devices, ntest=500)
    isdir(MixDIR) || mkpath(joinpath(MixDIR,"Templates/")) # create Dir if not exist
    nprofile = Int((ntraces/length(devices))÷1000)*1000-ntest
    numtr    = length(devices)*(ntest+nprofile)
    TrFILE   = h5open(joinpath(MixDIR,"traces_lanczos2_25_proc.h5"),"w")
    BufFILE  = h5open(joinpath(MixDIR,"Buf_proc.h5"),"w")
    XFILE    = h5open(joinpath(MixDIR,"X_proc.h5"),"w")
    YFILE    = h5open(joinpath(MixDIR,"Y_proc.h5"),"w")
    SFILE    = h5open(joinpath(MixDIR,"S_proc.h5"),"w")
    Traces   = create_dataset( TrFILE, "data", datatype(Float32), dataspace(trlen,numtr))
    Buf      = create_dataset(BufFILE, "data", datatype(  UInt8), dataspace(    8,numtr))
    X        = create_dataset(  XFILE, "data", datatype(  Int16), dataspace(   16,numtr))
    Y        = create_dataset(  YFILE, "data", datatype(  Int16), dataspace(   16,numtr))
    S        = create_dataset(  SFILE, "data", datatype(  Int16), dataspace(   16,numtr))
    for (n,dev) in enumerate(devices)
        print("$n/$(length(devices)) sampling from $dev: \r")
        selected  = sample(1:ntraces, nprofile+ntest; replace=false)
        sort!(view(selected,1:nprofile));sort!(view(selected,nprofile+1:nprofile+ntest))
        print("$n/$(length(devices)) sampling from $dev: loading Traces...   \r")
        devTraces = loaddata(joinpath(srcDir(dev), "traces_lanczos2_25_proc.npy"))
        Trtmp = devTraces[:,selected]
        print("$n/$(length(devices)) sampling from $dev: writing Traces...   \r")
        Traces[:,(n-1)*nprofile+1 :      n*nprofile] = Trtmp[:,1:nprofile]
        Traces[:,   end-n*ntest+1 : end-(n-1)*ntest] = Trtmp[:,nprofile+1:end]
        print("$n/$(length(devices)) sampling from $dev: ")
        print("Buf  ")
        devBuf    = loaddata(joinpath(srcDir(dev), "Buf_proc.npy"))
        Buf[:,(n-1)*nprofile+1 :      n*nprofile] = view(devBuf,:,selected[1:nprofile])
        Buf[:,   end-n*ntest+1 : end-(n-1)*ntest] = view(devBuf,:,selected[nprofile+1:end])
        print("X  ")
        devX      = loaddata(joinpath(srcDir(dev), "X_proc.npy"))
        X[:,(n-1)*nprofile+1 :      n*nprofile] = view(devX,:,selected[1:nprofile])
        X[:,   end-n*ntest+1 : end-(n-1)*ntest] = view(devX,:,selected[nprofile+1:end])
        print("Y  ")
        devY      = loaddata(joinpath(srcDir(dev), "Y_proc.npy"))
        Y[:,(n-1)*nprofile+1 :      n*nprofile] = view(devY,:,selected[1:nprofile])
        Y[:,   end-n*ntest+1 : end-(n-1)*ntest] = view(devY,:,selected[nprofile+1:end])
        print("S  ")
        devS      = loaddata(joinpath(srcDir(dev), "S_proc.npy"))
        S[:,(n-1)*nprofile+1 :      n*nprofile] = view(devS,:,selected[1:nprofile])
        S[:,   end-n*ntest+1 : end-(n-1)*ntest] = view(devS,:,selected[nprofile+1:end])
        print("\r                                            \r")
    end
    close(TrFILE)
    close(BufFILE)
    close(XFILE)
    close(YFILE)
    close(SFILE)
end


function main()
    # profiling for mixed traces
    for idx in devpoolsidx
        # setting filepaths, creating pooled training traces
        devices = devicespools[idx]
        MixDIR  = joinpath(TracesDIR, pooledDir(devices), "lanczos2_25/")
        nvalid  = length(devices)*ntest
        pooledTraces(MixDIR, devices, ntest)
        OUTDIR = joinpath(MixDIR, "Templates_POIe$(POIe_left)-$(POIe_right)/")
        isdir(OUTDIR) || mkpath(OUTDIR)

        # loading data
        Traces = loaddata( joinpath(MixDIR, "traces_lanczos2_25_proc.h5"); datapath="data")
        Buf    = loaddata( joinpath(MixDIR, "Buf_proc.h5"); datapath="data")
        X      = loaddata( joinpath(MixDIR,   "X_proc.h5"); datapath="data")
        Y      = loaddata( joinpath(MixDIR,   "Y_proc.h5"); datapath="data")
        S      = loaddata( joinpath(MixDIR,   "S_proc.h5"); datapath="data")
        
        # profiling on pooled traces
        println("*** Device: $idx *************************")
        secs = @elapsed Kyber768_profiling(MixDIR, OUTDIR, Traces, X, Y, S, Buf; nvalid,
                                           POIe_left, POIe_right, nicv_th, buf_nicv_th)
        ts = Time(0) + Second(floor(secs))
        println("time: $ts -> profiling $devices")
        println("**********************************************************")

        # remove files
        rm(joinpath(MixDIR, "traces_lanczos2_25_proc.h5"))
        rm(joinpath(MixDIR, "Buf_proc.h5"))
        rm(joinpath(MixDIR,   "X_proc.h5"))
        rm(joinpath(MixDIR,   "Y_proc.h5"))
        rm(joinpath(MixDIR,   "S_proc.h5"))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
