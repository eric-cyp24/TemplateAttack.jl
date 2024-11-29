module TemplateAttack

using StatsPlots, HDF5, Npy, Statistics, Distributions
using LinearAlgebra, SparseArrays, Mmap, Printf
using LeakageAssessment: groupbyval, sizecheck, computenicv_mthread, NICV, plotNICV

include("template.jl")
include("profiling.jl")
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
