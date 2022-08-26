module FoliationsManifoldsAutoencoders

export PolyModel,
    PolySetLinearPart!,
    PolyGetLinearPart,
    PolySubs!

# mapmethods.jl
export SSMCalc,
    ISFCalc

# vfmethods.jl
export SSMVFCalc,
    ISFVFCalc

# densepolymanifold.jl
export DensePolyManifold,
    fromFunction,
    setLinearPart!,
    getLinearPart

# mappolynomialinvariance.jl
# vfpolynomialinvariance.jl
export iFoliationMAP,
    iManifoldMAP,
    iFoliationVF,
    iManifoldVF

export dataPrune
export LinearFit
export ISFPadeManifold,
    PadeQ, PadeQpoint,
    PadeP, PadePpoint,
    PadeU, PadeUpoint,
    GaussSouthwellLinearSetup,
    GaussSouthwellOptim
    
export ISFImmersionManifold,
    ImmersionB,
    ImmersionW0,
    ImmersionWp,
    ImmersionBpoint,
    ImmersionW0point,
    ImmersionWppoint,
    ISFImmersionLoss,
    ISFImmersionRiemannianGradient,
    ISFImmersionReconstruct,
    ImmersionReconstruct,
    ISFImmersionSolve!

export AENCManifold,
    AENC_S,
    AENC_Spoint,
    AENC_Wl,
    AENC_Wlpoint,
    AENC_Wnl,
    AENC_Wnlpoint,
    AENCFitLoss,
    AENCFitRiemannianGradient,
    AENCROMLoss,
    AENCROMRiemannianGradient,
    AENCLoss,
    AENCRiemannianGradient,
    AENCIndentify

# from many places..
export Eval
    
export toFullDensePolynomial
export ManifoldAmplitudeSquare
export MAPManifoldFrequencyDamping,
    ODEManifoldFrequencyDamping
    
export ISFNormalForm,
    dataPrune,
    PCAEmbed,
    frequencyEmbed

# the overall function, with too many parameters hardcoded
export FoliationIdentify

# export from dependencies
export ManifoldsBase
export Manifolds
export Manopt
# export Manopt.ApproxHessianFiniteDifference

using SpecialFunctions
using LinearAlgebra
using ManifoldsBase
using Manifolds
using Manopt
using Printf
using Combinatorics
# using ForwardDiff

# this is from external source
using TRS
using BSON: @load, @save
using Random

# from mapmethods.jl
using DynamicPolynomials
using MultivariatePolynomials
using TaylorSeries

# from densepolynomials.jl
# using DynamicPolynomials
# using MultivariatePolynomials
using SparseArrays

# isfutils for filtering data, might not need
using DSP
using DelimitedFiles

# linear part of the submersion
# this consists of vectors of unit length, therefore they are on the unit sphere

import Base.randn
import Base.zeros
import Base.zero
import Base.transpose
import Base.*

import Base.getindex

# a helper type for the function
struct alwaysone
end

getindex(a::alwaysone, k) = 1

import ManifoldsBase.inner
import ManifoldsBase.retract!
import ManifoldsBase.retract
import ManifoldsBase.project!
import ManifoldsBase.manifold_dimension
import ManifoldsBase.zero_vector!
import ManifoldsBase.vector_transport_to!
import ManifoldsBase.vector_transport_to

# include("polymethods.jl")
# include("mapmethods.jl")
# include("vfmethods.jl")
include("matrixmanifoldutils.jl")
include("extrafunctions.jl")
include("linearmanifold.jl")
include("restrictedstiefel.jl")
include("tensormanifold.jl")
include("densepolymanifold.jl") # worked with densepolymanifold.jl
include("mappolynomialinvariance.jl")
include("vfpolynomialinvariance.jl")
include("polynomialmanifold.jl")
include("isfpadehybridmanifold.jl") # worked with isfpadehybridmanifold-04.jl
include("isfimmersionmanifold.jl")
include("aencmanifold.jl")
include("instfreqdamp.jl")
include("foliationidentify.jl")

# need to remove this?
# include("isfutils.jl")

end # module
