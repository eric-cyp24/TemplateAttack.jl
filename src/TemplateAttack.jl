module TemplateAttack

using StatsPlots, HDF5, Npy, Statistics
using Distributions: MvNormal
using LinearAlgebra, SparseArrays, Mmap
using LeakageAssessment: groupbyval, NICV

include("template.jl")
#include("profiling.jl")
include("utils.jl")

export Template, addtemplate, addpooledcovMatrix, loadtemplate, writetemplate, match, best_match,
       plotdatascatter!, plottemplate!, plotTemplate!


end # module TemplateAttack
