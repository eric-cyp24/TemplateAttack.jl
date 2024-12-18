module TemplateAttack

using LinearAlgebra, SparseArrays, Mmap, Printf, Statistics
using Plots, StatsPlots, HDF5, Distributions
using Npy
using LeakageAssessment: groupbyval, sizecheck, computenicv_mthread, NICV, plotNICV

### setting TMPFILE location ###
TMPDIR  = ispath("/local/scratch/cyp24/") ? "/local/scratch/cyp24/" : joinpath(@__DIR__, "../data/tmp/")
TMPFILE = joinpath(TMPDIR, "TemplateAttack.jl.tmp")
OUTDIR  = joinpath(@__DIR__, "../data/Output/")
ispath(TMPDIR) || mkpath(TMPDIR)
ispath(OUTDIR) || mkpath(OUTDIR)
###

include("template.jl")
include("profiling.jl")

using HypothesisTests: pvalue, UnequalCovHotellingT2Test, At_Binv_A
using EMAlgorithm: GaussianMixtureModel, emalgorithm_fixedweight_mprocess!
include("adjustment.jl")

include("evaluation.jl")
include("utils.jl")

export Template, loadtemplate, writetemplate

export LDA, buildTemplate, runprofiling, validate

export adjust!

export match, likelihoods, loglikelihoods, key_guessing, 
       success_rate, guessing_entropy

export plotTemplate,  plotdatascatter,  plotmvg,
       plotTemplate!, plotdatascatter!, plotmvg!

end # module TemplateAttack
