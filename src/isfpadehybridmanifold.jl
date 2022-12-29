#----------------------------------------------------------------------------------------------------------------------------------------
#
# U is the submersion
# to make things symmetric the invariance equation is
# (P o U)(x) = (Q o U)(y)
# where DQ(0) = I
# therefore the ROM is S = Q^{-1} o P
#
# This is inspired by the Pade approximation. The advantage is the on both sides of the equation we have the same order of polynomials
# The resulting S = Q^{-1} o P is higher order than P and Q on their own
# the order of P and Q must be the same for this to work
#----------------------------------------------------------------------------------------------------------------------------------------

"Scaling function to use in optimisation problems" 
function AmplitudeScaling(dataIN, Xstar)
#     @show size(Xstar)
    return (0.1^2 .+ sum((dataIN .- reshape(Xstar,:,1)) .^ 2, dims = 1))*2^(-7)
#     return ones(1,size(dataIN,2))*2^(-7)
end

struct ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, 𝔽} <: AbstractManifold{𝔽}
    mlist
    M        :: ProductManifold 
    R        :: ProductRetraction 
    VT       :: ProductVectorTransport
    Xstar    :: AbstractArray
end

function inner(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, p, X, Y) where {mdim, ndim, Porder, Qorder, Uorder, field}
    return inner(M.M, p, X, Y)
end

function retract!(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, q, p, X, method::AbstractRetractionMethod) where {mdim, ndim, Porder, Qorder, Uorder, field}
#     println("ISF RETRACT")
    return retract!(M.M, q, p, X, method)
end

# function vector_transport_to!(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, Y, p, X, q, method::AbstractVectorTransportMethod) where {mdim, ndim, Porder, Qorder, Uorder, field}
# vector_transport_to!(::ISFPadeManifold, ::Any, ::Any, ::Any, ::Any, ::AbstractVectorTransportMethod)
function vector_transport_to!(M::ISFPadeManifold, Y, p, X, q, method::AbstractVectorTransportMethod)
#     println("ISF VECTOR TRANSPORT 1")
    return vector_transport_to!(M.M, Y, p, X, q, method)
end

# function manifold_dimension(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}) where {mdim, ndim, Porder, Qorder, Uorder, field}
function manifold_dimension(M::ISFPadeManifold)
    return manifold_dimension(M.M)
end

function PadeP(M::ISFPadeManifold)
    return M.mlist[1]
end

function PadePpoint(X)
    return X.parts[1]
end

function PadeQ(M::ISFPadeManifold)
    return M.mlist[2]
end

function PadeQpoint(X)
    return X.parts[2]
end

function PadeU(M::ISFPadeManifold)
    return M.mlist[3]
end

function PadeUpoint(X)
    return X.parts[3]
end

function zero_vector!(M::ISFPadeManifold, X, p)
    return zero_vector!(M.M, X, p)
end

@doc raw"""
    zero(M::ISFPadeManifold)

Creates a zero `ISFPadeManifold` data representation.
"""
function zero(M::ISFPadeManifold)
    out = ProductRepr(map(zero, M.mlist))
#     setLinearPart!(PadeQ(M), PadeQpoint(out), I)
    return out
end

function randn(M::ISFPadeManifold)
    out = ProductRepr(map(randn, M.mlist))
#     setLinearPart!(PadeQ(M), PadeQpoint(out), I)
    return out
end

function project!(M::ISFPadeManifold, Y, p, X)
    return ProductRepr(map(project!, M.mlist, Y.parts, p.parts, X.parts))
end

function MaximumNorm(X::AbstractArray, ii::Integer)
    return (sqrt(sum(X .^ 2)), ii)
end

function MaximumNorm(X::ProductRepr, ii=nothing)
     res = map(MaximumNorm, X.parts, collect(1:length(X.parts)))
     mx = 0
     id = 0
     for k=1:length(res)
        if res[k][1] >= mx
            mx = res[k][1]
            id = k
        end
    end
    if ii == nothing
        return (mx, res[id][2])
    else
        return (mx, (ii, res[id][2]...))
    end
end

function NextElement(X::ProductRepr, id)
    # id[1] == 1 : P
    # id[1] == 2 : Q
    # id[1] == 3 : U
    # id[1] == 3, 
    #   id[2] == 1 : U1
    #   id[2] == 2 : U2
    #       id[3] == 1 : U2, 1
    # ...
    if id[1] == 1
        return (2, 0, 0)
    elseif id[1] == 2
        return (3, 1, 0)
    elseif id[1] == 3
        if id[2] == 1
            # the constant part
            return (id[1], id[2]+1, 0)
        elseif id[2] == 2
            # the constant part
            return (id[1], id[2]+1, 1)
        elseif id[3] < length(X.parts[id[1]].parts[id[2]].parts)
            # nonlinear part
            return (id[1], id[2], id[3]+1)
        elseif id[2] < length(X.parts[id[1]].parts)
            return (id[1], id[2]+1, 1)
        else
            return (1,0,0)
        end
    else
        return (1,0,0)
    end
end

function NextMaximumNorm(M::PolynomialManifold{mdim, ndim, order, nl_start, field}, X::ProductRepr, id) where {mdim, ndim, order, nl_start, field}
    # id[1] == 1 : P
    # id[1] == 2 : Q
    # id[1] == 3 : U
    # id[1] == 3, 
    #   id[2] == 1 : U1
    #   id[2] == 2 : U2
    #       id[3] == 1 : U2, 1
    # ...
    if id[1] == 1
        # return to Q, but no need to calculate
        return (2, 0, 0)
    elseif id[1] == 2
        # return with the linear part of U, no need to calculate
        return (3, 1, 0)
    elseif id[1] == 3
        # return with nonlinear parts of U,
        if id[2] < nl_start
            return (id[1], id[2]+1, 1)
        elseif id[2] < length(X.parts[id[1]].parts)
            norms = map(a -> sqrt(sum(a .^ 2)), X.parts[id[1]].parts[id[2]+1].parts)
            println("norms = ", norms)
            # random proportional to the size
            cs = cumsum(norms)
            ix = findfirst(cs./cs[end] .> rand(eltype(cs)))
            return (id[1], id[2]+1, ix)
        else
            # if it is the last nonlinear part, the return P
            return (1,0,0)
        end
    else
        # all else fails, return P
        return (1,0,0)
    end
end

@doc raw"""
    ISFPadeManifold(mdim, ndim, Porder, Qorder, Uorder, B=nothing, field::AbstractNumbers=ℝ)

Returns an `ISFPadeManifold` object, which provides the matrix manifold structure for an invariant foliation. The invariance equation, where these appear is
```math
\boldsymbol{P}\left(\boldsymbol{U}\left(\boldsymbol{x}\right)\right) = \boldsymbol{Q}\left(\boldsymbol{U}\left( \boldsymbol{F}(\boldsymbol{x}_{k})\right)\right)
```
where ``\boldsymbol{U}`` is a polynomial with HT tensor coefficients, ``\boldsymbol{P}`` is a general dense polynomial and ``\boldsymbol{Q}`` is a near identity polynomial, that is ``D \boldsymbol{Q}(\boldsymbol{0}) = \boldsymbol{I}``.
The purpose of polynomial ``\boldsymbol{Q}`` is to have a Pade approximated nonlinear map, ``\boldsymbol{S} = \boldsymbol{Q}^{-1} \circ \boldsymbol{P}``. This can balance polynomials on both sides of the invariance equation. In practice, we did not find much use for it yet.

The parameters are
  * `mdim`: co-dimension of the foliation
  * `ndim`: the dimesnion of the underlying phase space
  * `Porder`: order of polynomial ``\boldsymbol{P}``
  * `Qorder`: order of polynomial ``\boldsymbol{Q}``
  * `Uorder`: order of polynomial ``\boldsymbol{U}``
  * `B`: the matrix ``\boldsymbol{W}_1``, such that ``\boldsymbol{U} (\boldsymbol{W}_1 \boldsymbol{z})`` is constraing to be linear.
  * `fields`: dummy, a standard parameter of `Manifolds.jl`
"""
function ISFPadeManifold(mdim, ndim, Porder, Qorder, Uorder, B=nothing, field::AbstractNumbers=ℝ; node_rank = 4, X = zeros(ndim))
    mlist = (DenseNonconstManifold(mdim, mdim, Porder), DenseNearIdentityManifold(mdim, mdim, Qorder), SubmersionManifold(mdim, ndim, Uorder, B, node_rank = node_rank))
    M = ProductManifold(mlist...)
    R = ProductRetraction(map(x->getfield(x,:R), mlist)...)
    VT = ProductVectorTransport(map(x->getfield(x,:VT), mlist)...)
    return ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}(mlist, M, R, VT, X)
end

function ISFPadeSetRestriction(M::ISFPadeManifold, B)
    function setB(M, B)
        if typeof(M) <: ProductManifold
            map(x->setB(x, B), M.manifolds)
        elseif typeof(M) <: RestrictedStiefel
            M.B .= B
        end
        nothing
    end
    setB(M.M, B)
    return nothing
end

struct XYcache
    X::PolynomialCache
    Y::PolynomialCache
end

function makeCache(M::ISFPadeManifold, X, dataIN, dataOUT)
    return XYcache(makeCache(PadeU(M), PadeUpoint(X), dataIN), makeCache(PadeU(M), PadeUpoint(X), dataOUT))
end

# this updates everything, even if correct
function updateCache!(DV::XYcache, M::ISFPadeManifold, X, dataIN, dataOUT)
#     println("\n UPDATE")
    updateCache!(DV.X, PadeU(M), PadeUpoint(X), dataIN)
    updateCache!(DV.Y, PadeU(M), PadeUpoint(X), dataOUT)
    return nothing
end

function updateCachePartial!(DV::XYcache, M::ISFPadeManifold, X, dataIN, dataOUT, ord, ii)
#     println("\n UPDATE PARTIAL")
    updateCachePartial!(DV.X, PadeU(M), PadeUpoint(X), dataIN, ord = ord, ii = ii)
    updateCachePartial!(DV.Y, PadeU(M), PadeUpoint(X), dataOUT, ord = ord, ii = ii)
    return nothing
end

function ISFPadeLoss(M::ISFPadeManifold, X, dataIN, dataOUT; DV=makeCache(M, X, dataIN, dataOUT), Xnew=nothing, ord = 1, ii = 1)
    # copy over the new value if not the same object
    if Xnew != nothing
        if ord < 3
            PadeUpoint(X).parts[ord] .= Xnew
        elseif PadeUpoint(X).parts[ord].parts[ii] !== Xnew
            PadeUpoint(X).parts[ord].parts[ii] .= Xnew
        end
        updateCachePartial!(DV, M, X, dataIN, dataOUT, ord, ii)
        print("_")
    end

    datalen = size(dataIN,2)

    scale = AmplitudeScaling(dataIN, M.Xstar)

    Uoy = Eval(PadeU(M), PadeUpoint(X), dataOUT; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), dataIN; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), Uoy)
    PoUox = Eval(PadeP(M), PadePpoint(X), Uox)
    L0 = QoUoy .- PoUox
    print(".")
    return sum( (L0 .^ 2) ./ scale )/2/datalen
end

function ISFPadeLossInfinity(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, X, dataIN, dataOUT; DV=makeCache(M, X, dataIN, dataOUT)) where {mdim, ndim, Porder, Qorder, Uorder, field}
    datalen = size(dataIN,2)

    scale = (0.1^2 .+ sum((dataIN .- reshape(M.Xstar,:,1)) .^ 2, dims = 1))

    Uoy = Eval(PadeU(M), PadeUpoint(X), dataOUT; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), dataIN; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), Uoy)
    PoUox = Eval(PadeP(M), PadePpoint(X), Uox)
    L0 = QoUoy .- PoUox
    
    return maximum(sqrt.( sum(L0 .^ 2,dims=1) ./ sum(Uox .^ 2,dims=1) )), sum(sqrt.( sum(L0 .^ 2,dims=1) ./ sum(Uox .^ 2,dims=1) ))/datalen, sum( (L0 .^ 2) ./ scale )/datalen
#      sum( (L0 .^ 2) ./ scale )/datalen
end

function ISFPadeLossHistogram(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, X, dataIN, dataOUT) where {mdim, ndim, Porder, Qorder, Uorder, field}
    datalen = size(dataIN,2)

    scale = (0.1^2 .+ sum((dataIN .- reshape(M.Xstar,:,1)) .^ 2, dims = 1))

    Uoy = Eval(PadeU(M), PadeUpoint(X), dataOUT)
    Uox = Eval(PadeU(M), PadeUpoint(X), dataIN)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), Uoy)
    PoUox = Eval(PadeP(M), PadePpoint(X), Uox)
    L0 = QoUoy .- PoUox
    
    return vec(sum(Uox .^ 2,dims=1)), vec(sqrt.( sum(L0 .^ 2,dims=1) ./ sum(Uox .^ 2,dims=1) ))
end

function ISFPadeGradientP(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, X, dataIN, dataOUT; DV=makeCache(M, X, dataIN, dataOUT)) where {mdim, ndim, Porder, Qorder, Uorder, field}
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN, M.Xstar)

    Uoy = Eval(PadeU(M), PadeUpoint(X), dataOUT; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), dataIN; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), Uoy)
    PoUox = Eval(PadeP(M), PadePpoint(X), Uox)
    L0 = (QoUoy .- PoUox) ./ scale

    return L0_DF(PadeP(M), PadePpoint(X), Uox, L0 = -1.0*L0)/datalen
end

function ISFPadeGradientQ(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, X, dataIN, dataOUT; DV=makeCache(M, X, dataIN, dataOUT)) where {mdim, ndim, Porder, Qorder, Uorder, field}
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN, M.Xstar)

    Uoy = Eval(PadeU(M), PadeUpoint(X), dataOUT; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), dataIN; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), Uoy)
    PoUox = Eval(PadeP(M), PadePpoint(X), Uox)
    L0 = (QoUoy .- PoUox) ./ scale

    return L0_DF(PadeQ(M), PadeQpoint(X), Uoy, L0 = L0)/datalen
end

function gradUmonomial(M::Union{ConstantManifold, LinearManifold}, X, L0_JPoUox, L0_JQoUoy, dataIN, dataOUT, _p1, _p2)
    datalen = size(dataIN,2)
    return (L0_DF(M, X, dataOUT, L0 = L0_JQoUoy) .- L0_DF(M, X, dataIN, L0 = L0_JPoUox))/datalen
end

# Calculate the gradient with respect to a node in U
# ord -> monomial order
# ii  -> index of the node 
function ISFPadeGradientU(M::ISFPadeManifold, X, dataIN, dataOUT, ord, ii; DV=makeCache(M, X, dataIN, dataOUT), Xnew=nothing)
    nl_start = typeof(PadeU(M)).parameters[4]
    # copy over the new value if not the same object
    if Xnew != nothing
        if ord < nl_start
            PadeUpoint(X).parts[ord] .= Xnew
        elseif PadeUpoint(X).parts[ord].parts[ii] !== Xnew
            PadeUpoint(X).parts[ord].parts[ii] .= Xnew
        end
        updateCachePartial!(DV, M, X, dataIN, dataOUT, ord, ii)
        print(",")
    end

    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN, M.Xstar)

    Uoy = Eval(PadeU(M), PadeUpoint(X), dataOUT; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), dataIN; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), Uoy)
    PoUox = Eval(PadeP(M), PadePpoint(X), Uox)
    L0 = (QoUoy .- PoUox) ./ scale
    
    # the Jacobians call tensorVecs for each leaf once, so they are expensive
    JQoUoy = Jacobian(PadeQ(M), PadeQpoint(X), Uoy)
    JPoUox = Jacobian(PadeP(M), PadePpoint(X), Uox)
    L0_JQoUoy = dropdims(sum(JQoUoy .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    L0_JPoUox = dropdims(sum(JPoUox .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    grad = (L0_DF_parts(PadeU(M), PadeUpoint(X), dataOUT, L0 = L0_JQoUoy, ord = ord, ii = ii, DV=DV.Y) .- 
            L0_DF_parts(PadeU(M), PadeUpoint(X), dataIN, L0 = L0_JPoUox, ord = ord, ii = ii, DV=DV.X))/datalen
#         @show size(grad)
    return grad
end

# function gradUmonomial(M::TensorManifold, X, L0_JPoUox, L0_JQoUoy, dataIN, dataOUT, DVUox, DVUoy)
#     datalen = size(dataIN,2)
# #     DVUoy = tensorVecs(M, X, [dataOUT])
# #     tensorBVecs!(DVUoy, M, X)
# #     DVUox = tensorVecs(M, X, [dataIN])
# #     tensorBVecs!(DVUox, M, X)
#     
#     # this is missing the penalty term
#     function tmp(ii)
#         G = (L0_DF(M, X, DVUoy, dataOUT, L0_JQoUoy, ii) .- L0_DF(M, X, DVUox, dataIN, L0_JPoUox, ii))/datalen 
#         return G
#     end
#     return ProductRepr(map(tmp, collect(1:length(X.parts)))...)
# end

function ISFPadeGradientAll(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, X, dataIN, dataOUT; DV=makeCache(M, X, dataIN, dataOUT)) where {mdim, ndim, Porder, Qorder, Uorder, field}
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN, M.Xstar)
    
    Uoy = Eval(PadeU(M), PadeUpoint(X), dataOUT; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), dataIN; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), Uoy)
    PoUox = Eval(PadeP(M), PadePpoint(X), Uox)
    L0 = (QoUoy .- PoUox) ./ scale

    GP = L0_DF(PadeP(M), PadePpoint(X), Uox, L0 = -1.0*L0)/datalen
    GQ = L0_DF(PadeQ(M), PadeQpoint(X), Uoy, L0 = L0)/datalen

    # the Jacobians call tensorVecs for each leaf once, so they are expensive
    JQoUoy = Jacobian(PadeQ(M), PadeQpoint(X), Uoy)
    JPoUox = Jacobian(PadeP(M), PadePpoint(X), Uox)
    L0_JQoUoy = dropdims(sum(JQoUoy .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    L0_JPoUox = dropdims(sum(JPoUox .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    
    GU = (L0_DF(PadeU(M), PadeUpoint(X), dataOUT, L0 = L0_JQoUoy./datalen, DV = DV.Y) .- L0_DF(PadeU(M), PadeUpoint(X), dataIN, L0 = L0_JPoUox./datalen, DV = DV.X))
    print("*")
    return ProductRepr(GP, GQ, GU)
end

function ISFPadeRiemannianGradient(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, X, dataIN, dataOUT; DV=makeCache(M, X, dataIN, dataOUT)) where {mdim, ndim, Porder, Qorder, Uorder, field}
    gr = ISFPadeGradientAll(M, X, dataIN, dataOUT; DV=DV)
    return project(M, X, gr)
end

function ISFPadeGradientHessianP(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, X, dataIN, dataOUT; DV=makeCache(M, X, dataIN, dataOUT)) where {mdim, ndim, Porder, Qorder, Uorder, field}
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN, M.Xstar)

    Uoy = Eval(PadeU(M), PadeUpoint(X), dataOUT; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), dataIN; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), Uoy)
    PoUox = Eval(PadeP(M), PadePpoint(X), Uox)
    L0 = (QoUoy .- PoUox) ./ scale
    
#     println("P GRAD TIME")
    grad = L0_DF(PadeP(M), PadePpoint(X), Uox, L0 = -1.0*L0)/datalen
#     println("P HESS TIME")
    hess = DFoxT_DFox(PadeP(M), PadePpoint(X), Uox; scale=scale)/datalen
    return grad, hess
end

function ISFPadeGradientHessianQ(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, X, dataIN, dataOUT; DV=makeCache(M, X, dataIN, dataOUT)) where {mdim, ndim, Porder, Qorder, Uorder, field}
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN, M.Xstar)

    Uoy = Eval(PadeU(M), PadeUpoint(X), dataOUT; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), dataIN; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), Uoy)
    PoUox = Eval(PadeP(M), PadePpoint(X), Uox)
    L0 = (QoUoy .- PoUox) ./ scale
    
    grad = L0_DF(PadeQ(M), PadeQpoint(X), Uoy, L0 = L0)/datalen
    hess = DFoxT_DFox(PadeQ(M), PadeQpoint(X), Uoy; scale=scale)/datalen
    return grad, hess
end

function ISFPadeHessianVectorU(M::ISFPadeManifold, X, Xp, dataIN, dataOUT, ord, ii; DV=makeCache(M, X, dataIN, dataOUT), Xnew=nothing)
    # copy over the new value if not the same object
    nl_start = typeof(PadeU(M)).parameters[4]
    if Xnew != nothing
        if ord < nl_start
            PadeUpoint(X).parts[ord] .= Xnew
        elseif PadeUpoint(X).parts[ord].parts[ii] !== Xnew
            PadeUpoint(X).parts[ord].parts[ii] .= Xnew
        end
        updateCachePartial!(DV, M, X, dataIN, dataOUT, ord, ii)
        print("~")
    end
    
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN, M.Xstar)

    Uoy = Eval(PadeU(M), PadeUpoint(X), dataOUT; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), dataIN; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), Uoy)
    PoUox = Eval(PadeP(M), PadePpoint(X), Uox)
    L0 = (QoUoy .- PoUox) ./ scale
    
    Dt_Uox_z0 = DF_dt_parts(PadeU(M), PadeUpoint(X), dataIN, dt = Xp, ord = ord, ii = ii, DV = DV.X)
    Dt_Uoy_z0 = DF_dt_parts(PadeU(M), PadeUpoint(X), dataOUT, dt = Xp, ord = ord, ii = ii, DV = DV.Y)
    Dx_QoUoy = JF_dx(PadeQ(M), PadeQpoint(X), Uoy, Dt_Uoy_z0)
    Dx_PoUox = JF_dx(PadeP(M), PadePpoint(X), Uox, Dt_Uox_z0)
    
    L0_grad = (Dx_QoUoy - Dx_PoUox) ./ scale
    
    JQoUoy = Jacobian(PadeQ(M), PadeQpoint(X), Uoy)
    JPoUox = Jacobian(PadeP(M), PadePpoint(X), Uox)
    L0_JQoUoy = dropdims(sum(JQoUoy .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    L0_JPoUox = dropdims(sum(JPoUox .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    L0_grad_JQoUoy = dropdims(sum(JQoUoy .* reshape(L0_grad, size(L0_grad,1), 1, size(L0_grad,2)), dims=1), dims=1)
    L0_grad_JPoUox = dropdims(sum(JPoUox .* reshape(L0_grad, size(L0_grad,1), 1, size(L0_grad,2)), dims=1), dims=1)
    
    pQ = L0_HF_dx(PadeQ(M), PadeQpoint(X), Uoy, L0, Dt_Uoy_z0)
    pP = L0_HF_dx(PadeP(M), PadePpoint(X), Uox, L0, Dt_Uox_z0)

    hess_A = ( L0_DF_parts(PadeU(M), PadeUpoint(X), dataOUT, L0 = L0_grad_JQoUoy, ord = ord, ii = ii, DV = DV.Y)
            .- L0_DF_parts(PadeU(M), PadeUpoint(X), dataIN, L0 = L0_grad_JPoUox, ord = ord, ii = ii, DV = DV.X) )/datalen
    hess_B = ( L0_DF_parts(PadeU(M), PadeUpoint(X), dataOUT, L0 = pQ, ord = ord, ii = ii, DV = DV.Y)
            .- L0_DF_parts(PadeU(M), PadeUpoint(X), dataIN, L0 = pP, ord = ord, ii = ii, DV = DV.X) )/datalen
    grad = (   L0_DF_parts(PadeU(M), PadeUpoint(X), dataOUT, L0 = L0_JQoUoy, ord = ord, ii = ii, DV = DV.Y)
            .- L0_DF_parts(PadeU(M), PadeUpoint(X), dataIN, L0 = L0_JPoUox, ord = ord, ii = ii, DV = DV.X) )/datalen
    if ord < nl_start
        return HessProjection(PadeU(M).mlist[ord], PadeUpoint(X).parts[ord], grad, hess_A + hess_B, Xp)
    else
        return HessProjection(PadeU(M).M.manifolds[ord].manifolds[ii], PadeUpoint(X).parts[ord].parts[ii], grad, hess_A + hess_B, Xp)
    end
    nothing
end

function ISFPadeGradientHessianU(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, X, dataIN, dataOUT, ord, ii; DV=makeCache(M, X, dataIN, dataOUT)) where {mdim, ndim, Porder, Qorder, Uorder, field}
#       DUox^T x JPoUox^T x JPoUox x DUox   -> 1
#     + DUoy^T x JQoUoy^T x JQoUoy x DUoy   -> 2
#     - DUox^T x JPoUox^T x JQoUoy x DUoy   -> 3
#     - DUoy^T x JQoUoy^T x JPoUox x DUox   -> 3^T
#     - DUox^T x L0J2PoUox x DUox
#     + DUoy^T x L0J2QoUoy x DUoy
    nl_start = typeof(PadeU(M)).parameters[4]
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN, M.Xstar)

    Uoy = Eval(PadeU(M), PadeUpoint(X), dataOUT; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), dataIN; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), Uoy)
    PoUox = Eval(PadeP(M), PadePpoint(X), Uox)
    L0 = (QoUoy .- PoUox) ./ scale

    JQoUoy = Jacobian(PadeQ(M), PadeQpoint(X), Uoy)
    JPoUox = Jacobian(PadeP(M), PadePpoint(X), Uox)
    L0_JQoUoy = dropdims(sum(JQoUoy .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    L0_JPoUox = dropdims(sum(JPoUox .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    # grad_U = L0^T (JQoUoy x DUoy - JPoUox x DUox)
    grad = (L0_DF_parts(PadeU(M), PadeUpoint(X), dataOUT, L0 = L0_JQoUoy, ord = ord, ii = ii; DV = DV.Y) .-
            L0_DF_parts(PadeU(M), PadeUpoint(X), dataIN, L0 = L0_JPoUox, ord = ord, ii = ii; DV = DV.X))/datalen
    
    # the hessians or P and Q
    J2QoUoy = Hessian(PadeQ(M), PadeQpoint(X), Uoy)
    J2PoUox = Hessian(PadeP(M), PadePpoint(X), Uox)
    L0J2QoUoy = dropdims(sum(J2QoUoy .* reshape(L0, size(L0,1), 1, 1, size(L0,2)), dims=1), dims=1)
    L0J2PoUox = dropdims(sum(J2PoUox .* reshape(L0, size(L0,1), 1, 1, size(L0,2)), dims=1), dims=1)

    if ord < nl_start
        hess = DFT_JFT_JF_DF(PadeU(M).mlist[ord], DV.X.tree.parts[ord], DV.Y.tree.parts[ord], JPoUox, JQoUoy, L0J2PoUox, L0J2QoUoy, dataIN, dataOUT, ii; scale=scale)/datalen
    else
        hess = DFT_JFT_JF_DF(PadeU(M).mlist[ord], DV.X.tree.parts[ord], DV.Y.tree.parts[ord], JPoUox, JQoUoy, L0J2PoUox, L0J2QoUoy, PadeU(M).P * dataIN, PadeU(M).P * dataOUT, ii; scale=scale)/datalen
    end
    
#     if ord >= nl_start
#         @time H11, H22, H12 = HessianCombination(PadeQ(M), PadeQpoint(X), PadeP(M), PadePpoint(X), Uoy, Uox)
#         @show size(H11), size(scale)
#         @time hessB = L0_DF1_DF2_parts(PadeU(M).mlist[ord], PadeUpoint(X).parts[ord], H11./reshape(scale,1,1,:), PadeU(M).P * dataOUT, PadeU(M).P * dataOUT; ii = ii, DVX = DV.Y.tree.parts[ord], DVY = DV.Y.tree.parts[ord])
#         @time hessB .+= L0_DF1_DF2_parts(PadeU(M).mlist[ord], PadeUpoint(X).parts[ord], H22./reshape(scale,1,1,:), PadeU(M).P * dataIN, PadeU(M).P * dataIN; ii = ii, DVX = DV.X.tree.parts[ord], DVY = DV.X.tree.parts[ord])
#         @time hessB .+= L0_DF1_DF2_parts(PadeU(M).mlist[ord], PadeUpoint(X).parts[ord], H12./reshape(scale,1,1,:), PadeU(M).P * dataOUT, PadeU(M).P * dataIN; ii = ii, DVX = DV.Y.tree.parts[ord], DVY = DV.X.tree.parts[ord])
#         
#         @show maximum(abs.(hess .- hessB/datalen)), maximum(abs.(hess))
# #         return grad, hessB
#     end
    
    return grad, hess
end

function testPadeLoss()
    din = 5
    dout = 2
    M1 = ISFPadeManifold(dout, din, 4, 4, 4)
    zero(M1)
    x1 = randn(M1)
    dataIN = 0.2*randn(din, 100)
    dataOUT = 0.2*randn(din, 100)
    
    ISFPadeLoss(M1, x1, dataIN, dataOUT)
    @time ISFPadeLoss(M1, x1, dataIN, dataOUT)

    gradP = ISFPadeGradientP(M1, x1, dataIN, dataOUT)
    @time gradP = ISFPadeGradientP(M1, x1, dataIN, dataOUT)
            
    # checking P derivatives
    flag = false
    xp = deepcopy(x1)
    eps = 1e-6
    let ord = 1, ii = nothing
        MP = PadeP(M1)
        XP = PadePpoint(x1)
        XPp = PadePpoint(xp)
        gradP = ISFPadeGradientP(M1, x1, dataIN, dataOUT)
        gradPp = deepcopy(gradP)
        for k1=1:size(XP,1), k2=1:size(XP,2)
            XPp[k1,k2] += eps
            gradPp[k1,k2] = (ISFPadeLoss(M1, xp, dataIN, dataOUT) - ISFPadeLoss(M1, x1, dataIN, dataOUT)) / eps
            relErr = (gradPp[k1,k2] - gradP[k1,k2]) / gradP[k1,k2]
            if abs(relErr) > 1e-4
                flag = true
                println("GP o=", ord, " el=", k1, ",", k2, "/", size(XP,1), ",", size(XP,2), " E = ", relErr, " G=", gradP[k1,k2], " A=", gradPp[k1,k2])
            end
            XPp[k1,k2] = XP[k1,k2]
        end
    end
        
    # checking Q derivatives
    flag = false
    xp = deepcopy(x1)
    eps = 1e-6
    let ord = 1, ii = nothing
        MP = PadeQ(M1)
        XP = PadeQpoint(x1)
        XPp = PadeQpoint(xp)
        gradP = ISFPadeGradientQ(M1, x1, dataIN, dataOUT)
        gradPp = deepcopy(gradP)
        for k1=1:size(XP,1), k2=1:size(XP,2)
            XPp[k1,k2] += eps
            gradPp[k1,k2] = (ISFPadeLoss(M1, xp, dataIN, dataOUT) - ISFPadeLoss(M1, x1, dataIN, dataOUT)) / eps
            relErr = (gradPp[k1,k2] - gradP[k1,k2]) / gradP[k1,k2]
            if abs(relErr) > 1e-4
                flag = true
                println("GQ o=", ord, " el=", k1, ",", k2, "/", size(XP,1), ",", size(XP,2), " E = ", relErr, " G=", gradP[k1,k2], " A=", gradPp[k1,k2])
            end
            XPp[k1,k2] = XP[k1,k2]
        end
    end
    
    # checking U derivatives
    flag = false
    xp = deepcopy(x1)
    eps = 1e-6
    for ord = 1:2
        ii = nothing
        MP = PadeU(M1).mlist[ord]
        XP = PadeUpoint(x1).parts[ord]
        XPp = PadeUpoint(xp).parts[ord]
        gradP = ISFPadeGradientU(M1, x1, dataIN, dataOUT, ord, ii)
        gradPp = deepcopy(gradP)
        for k1=1:size(XP,1), k2=1:size(XP,2)
            XPp[k1,k2] += eps
            gradPp[k1,k2] = (ISFPadeLoss(M1, xp, dataIN, dataOUT) - ISFPadeLoss(M1, x1, dataIN, dataOUT)) / eps
            relErr = (gradPp[k1,k2] - gradP[k1,k2]) / gradP[k1,k2]
            if abs(relErr) > 1e-4
                flag = true
                println("GU o=", ord, " el=", k1, ",", k2, "/", size(XP,1), ",", size(XP,2), " E = ", relErr, " G=", gradP[k1,k2], " A=", gradPp[k1,k2])
            end
            XPp[k1,k2] = XP[k1,k2]
        end
    end
    for ord=3:length(PadeUpoint(x1).parts)
        MP = PadeU(M1).mlist[ord]
        XP = PadeUpoint(x1).parts[ord]
        XPp = PadeUpoint(xp).parts[ord]
        for ii=1:nr_nodes(MP)
            gradP = ISFPadeGradientU(M1, x1, dataIN, dataOUT, ord, ii)
            gradPp = deepcopy(gradP)
            for k1=1:size(XP.parts[ii],1), k2=1:size(XP.parts[ii],2)
                XPp.parts[ii][k1,k2] += eps
    #             @show gradPp[k1,k2]
    #             @show gradP[k1,k2]
                gradPp[k1,k2] = (ISFPadeLoss(M1, xp, dataIN, dataOUT) - ISFPadeLoss(M1, x1, dataIN, dataOUT)) / eps
                relErr = (gradPp[k1,k2] - gradP[k1,k2]) / gradP[k1,k2]
                if abs(relErr) > 1e-4
                    flag = true
                    println("GU o=", ord, " node=", ii, "/", nr_nodes(MP), " el=", k1, ",", k2, "/", size(XP.parts[ii],1), ",", size(XP.parts[ii],2), " E = ", relErr, " G=", gradP[k1,k2], " A=", gradPp[k1,k2])
                end
                XPp.parts[ii][k1,k2] = XP.parts[ii][k1,k2]
            end
        end
    end
    
    # checking P HESSIAN
    flag = false
    xp = deepcopy(x1)
    eps = 1e-6
    let ord = 1, ii = nothing
        MP = PadeP(M1)
        XP = PadePpoint(x1)
        XPp = PadePpoint(xp)
        gradP, hessP = ISFPadeGradientHessianP(M1, x1, dataIN, dataOUT)
        hessPp = deepcopy(hessP)
#         @show size(XP.parts[ii])
        for k1=1:size(XP,1), k2=1:size(XP,2)
            XPp[k1,k2] += eps
            hessPp[:,:,k1,k2] .= (ISFPadeGradientP(M1, xp, dataIN, dataOUT) .- ISFPadeGradientP(M1, x1, dataIN, dataOUT))/eps
            relErr = maximum(abs.((hessPp[:,:,k1,k2] - hessP[:,:,k1,k2]) ./ hessP[:,:,k1,k2]))
            if abs(relErr) > 1e-4
                flag = true
                println("HP o=", ord, " el=", k1, ",", k2, "/", size(XP,1), ",", size(XP,2), " E = ", relErr, " HP=", maximum(abs.(hessP[:,:,k1,k2])), " A=", maximum(abs.(hessPp[:,:,k1,k2])))
#                 println("diff")
#                 display(hessPp[:,:,k1,k2] - hessP[:,:,k1,k2])
#                 println("analytic")
#                 display(hessP[:,:,k1,k2])
#                 println("approximate")
#                 display(hessPp[:,:,k1,k2])
            end
            XPp[k1,k2] = XP[k1,k2]
        end
    end
    
    # checking Q HESSIAN
    flag = false
    xp = deepcopy(x1)
    eps = 1e-8
    let ord = 1, ii = nothing
        MP = PadeQ(M1)
        XP = PadeQpoint(x1)
        XPp = PadeQpoint(xp)
        gradP, hessP = ISFPadeGradientHessianP(M1, x1, dataIN, dataOUT)
        hessPp = deepcopy(hessP)
#         @show size(XP.parts[ii])
        for k1=1:size(XP,1), k2=1:size(XP,2)
            XPp[k1,k2] += eps
            hessPp[:,:,k1,k2] .= (ISFPadeGradientP(M1, xp, dataIN, dataOUT) .- ISFPadeGradientP(M1, x1, dataIN, dataOUT))/eps
            relErr = maximum(abs.((hessPp[:,:,k1,k2] - hessP[:,:,k1,k2]) ./ hessP[:,:,k1,k2]))
            if abs(relErr) > 1e-4
                flag = true
                println("HP o=", ord, " el=", k1, ",", k2, "/", size(XP,1), ",", size(XP,2), " E = ", relErr, " HP=", maximum(abs.(hessP[:,:,k1,k2])), " A=", maximum(abs.(hessPp[:,:,k1,k2])))
#                 println("diff")
#                 display(hessPp[:,:,k1,k2] - hessP[:,:,k1,k2])
#                 println("analytic")
#                 display(hessP[:,:,k1,k2])
#                 println("approximate")
#                 display(hessPp[:,:,k1,k2])
            end
            XPp[k1,k2] = XP[k1,k2]
        end
    end
    
    # checking U HESSIAN
    flag = false
    xp = deepcopy(x1)
    eps = 1e-7
    for ord = 1:2
        ii = nothing
        MP = PadeU(M1).mlist[ord]
        XP = PadeUpoint(x1).parts[ord]
        XPp = PadeUpoint(xp).parts[ord]
        gradP, hessP = ISFPadeGradientHessianU(M1, x1, dataIN, dataOUT, ord, ii)
        hessPp = deepcopy(hessP)
#         @show size(XP.parts[ii])
        for k1=1:size(XP,1), k2=1:size(XP,2)
            XPp[k1,k2] += eps
            hessPp[:,:,k1,k2] .= (ISFPadeGradientU(M1, xp, dataIN, dataOUT, ord, ii) .- ISFPadeGradientU(M1, x1, dataIN, dataOUT, ord, ii))/eps
            relErr = maximum(abs.((hessPp[:,:,k1,k2] - hessP[:,:,k1,k2]) ./ hessP[:,:,k1,k2]))
#             if abs(relErr) > 1e-4
                flag = true
                println("HU o=", ord, " el=", k1, ",", k2, "/", size(XP,1), ",", size(XP,2), " E = ", relErr, " HP=", maximum(abs.(hessP[:,:,k1,k2])), " A=", maximum(abs.(hessPp[:,:,k1,k2])))
#                 println("diff")
#                 display(hessPp[:,:,k1,k2] - hessP[:,:,k1,k2])
#                 println("analytic")
#                 display(hessP[:,:,k1,k2])
#                 println("approximate")
#                 display(hessPp[:,:,k1,k2])
#             end
            XPp[k1,k2] = XP[k1,k2]
        end

        # Hessian times vector testing
        Cache = makeCache(M1, x1, dataIN, dataOUT)
        Xdelta = project(PadeU(M1).M.manifolds[ord], PadeUpoint(x1).parts[ord], randn(size(XP)))
#             @time ISFPadeGradientU(M1, x1, dataIN, dataOUT, ord, ii; DV=Cache, Xnew = randn(size(XP)))
#             @time hessDelta = ISFPadeHessianVectorU(M1, x1, Xdelta, dataIN, dataOUT, ord, ii; DV=Cache, 
#                                                     Xnew = project(PadeU(M1).M.manifolds[ord],randn(size(XP))))
        @time hessDelta = ISFPadeHessianVectorU(M1, x1, Xdelta, dataIN, dataOUT, ord, ii; DV=Cache)
        hessR = HessFullProjection(PadeU(M1).M.manifolds[ord], PadeUpoint(x1).parts[ord], gradP, hessP)
        hessDelta0 = reshape(reshape(hessR, size(hessP,1)*size(hessP,2), :) * vec(Xdelta), size(hessDelta))
        @show nrm = norm(hessDelta - hessDelta0)
        
        nhess = ApproxHessianFiniteDifference(
            PadeU(M1).M.manifolds[ord],
            PadeUpoint(x1).parts[ord],
            (M, x) -> project(PadeU(M1).M.manifolds[ord], PadeUpoint(x1).parts[ord], ISFPadeGradientU(M1, x1, dataIN, dataOUT, ord, ii; DV=Cache, Xnew=x));
            steplength=2^(-22),
            retraction_method=PadeU(M1).R.retractions[ord],
            vector_transport_method=PadeU(M1).mlist[ord].VT)
        @time hessDeltaFD = nhess(PadeU(M1).M.manifolds[ord], PadeUpoint(x1).parts[ord], Xdelta)
        @show nrm = norm(hessDelta - hessDeltaFD)
        if nrm > 1
            @show hessDelta
            @show hessDeltaFD
            @show hessDeltaFD ./ hessDelta
#                 return nothing
        end
    end
#     return
    for ord=3:length(PadeUpoint(x1).parts)
        MP = PadeU(M1).mlist[ord]
        XP = PadeUpoint(x1).parts[ord]
        XPp = PadeUpoint(xp).parts[ord]
        for ii=1:nr_nodes(MP)
            gradP, hessP = ISFPadeGradientHessianU(M1, x1, dataIN, dataOUT, ord, ii)
            hessPp = deepcopy(hessP)
    #         @show size(XP.parts[ii])
            for k1=1:size(XP.parts[ii],1), k2=1:size(XP.parts[ii],2)
                XPp.parts[ii][k1,k2] += eps
                hessPp[:,:,k1,k2] .= (ISFPadeGradientU(M1, xp, dataIN, dataOUT, ord, ii) .- ISFPadeGradientU(M1, x1, dataIN, dataOUT, ord, ii))/eps
                relErr = maximum(abs.((hessPp[:,:,k1,k2] - hessP[:,:,k1,k2]) ./ hessP[:,:,k1,k2]))
                if abs(relErr) > 1e-4
                    flag = true
                    println("HU o=",  ord, " node=", ii, "/", nr_nodes(MP), " el=", k1, ",", k2, "/", size(XP.parts[ii],1), ",", size(XP.parts[ii],2), " E = ", relErr, " HP=", maximum(abs.(hessP[:,:,k1,k2])), " A=", maximum(abs.(hessPp[:,:,k1,k2])))
#                     println("diff")
#                     display(hessPp[:,:,k1,k2] - hessP[:,:,k1,k2])
#                     println("analytic")
#                     display(hessP[:,:,k1,k2])
#                     println("approximate")
#                     display(hessPp[:,:,k1,k2])
                end
                XPp.parts[ii][k1,k2] = XP.parts[ii][k1,k2]
            end
            # Hessian times vector testing
            Cache = makeCache(M1, x1, dataIN, dataOUT)
            Xdelta = project(PadeU(M1).M.manifolds[ord].manifolds[ii], PadeUpoint(x1).parts[ord].parts[ii], randn(size(XP.parts[ii])))
#             @time ISFPadeGradientU(M1, x1, dataIN, dataOUT, ord, ii; DV=Cache, Xnew = randn(size(XP.parts[ii])))
#             @time hessDelta = ISFPadeHessianVectorU(M1, x1, Xdelta, dataIN, dataOUT, ord, ii; DV=Cache, 
#                                                     Xnew = project(PadeU(M1).M.manifolds[ord].manifolds[ii],randn(size(XP.parts[ii]))))
            @time hessDelta = ISFPadeHessianVectorU(M1, x1, Xdelta, dataIN, dataOUT, ord, ii; DV=Cache)
            hessR = HessFullProjection(PadeU(M1).M.manifolds[ord].manifolds[ii], PadeUpoint(x1).parts[ord].parts[ii], gradP, hessPp)
            hessDelta0 = reshape(reshape(hessR, size(hessP,1)*size(hessP,2), :) * vec(Xdelta), size(hessDelta))
            @show nrm = norm(hessDelta - hessDelta0)
            
            nhess = ApproxHessianFiniteDifference(
                PadeU(M1).M.manifolds[ord].manifolds[ii],
                PadeUpoint(x1).parts[ord].parts[ii],
                (M, x) -> project(PadeU(M1).M.manifolds[ord].manifolds[ii], PadeUpoint(x1).parts[ord].parts[ii], ISFPadeGradientU(M1, x1, dataIN, dataOUT, ord, ii; DV=Cache, Xnew=x));
                steplength=2^(-22),
                retraction_method=PadeU(M1).R.retractions[ord].retractions[ii],
                vector_transport_method=PadeU(M1).mlist[ord].VT.methods[ii])
            @time hessDeltaFD = nhess(PadeU(M1).M.manifolds[ord].manifolds[ii], PadeUpoint(x1).parts[ord].parts[ii], Xdelta)
            @show nrm = norm(hessDelta- hessDeltaFD)
            if nrm > 1
                @show hessDelta
                @show hessDeltaFD
                @show hessDeltaFD ./ hessDelta
#                 return nothing
            end
        end
    end
    nothing
end


# MF, XF
# Loss(MF, XF)
# Gradient(MF, XF)
# GradientHessian(MF, XF)
# Msub, Xsub [retraction -> Msub.R, Ms -> Msub.M (Msub is a tensorManifold or Linear Manifold)]
# radius, radius_max
# returns radius
# --------------------
# MF = M1
# XF = x1
# Loss = (MF, XF) -> ISFPadeLoss(MF, XF, dataIN, dataOUT)
# Gradient = (MF, XF) -> ISFPadeGradientU(MF, XF, dataIN, dataOUT, ord, ii)
# GradientHessian = (MF, XF) -> ISFPadeGradientHessianU(MF, XF, dataIN, dataOUT, ord, ii)
# Msub = PadeU(M1).M.mlist[ord].mlist[ii]
# Xsub = PadeUpoint(x1).parts[ord].parts[ii]
# ----------------
# needs to use cache, but only needs updating for U components
# problem: How to check if update is needed? Use a callback!
function trust_region!(MF, XF, Loss, Gradient, GradientHessian, Cache, UpdateCachePartial, Msub, Xsub, retraction, radius, radius_max; manifold = true, itmax = 5, trmax = 4)
#     print(":")
    if manifold
        GF, HF = GradientHessian(MF, XF)
        G = project(Msub, Xsub, GF)
        # projecting HESSIAN
        H = HessFullProjection(Msub, Xsub, G, HF)
    else
        G, H = GradientHessian(MF, XF)
    end
    ng = norm(G)
    loss = 0
    stop = false
    for iit=1:itmax
        print("|")
        loss = Loss(MF, XF)
        # update hessian at every 10 iterations
        if mod(iit,10) == 0 && (iit < itmax)
            if manifold
                GF, HF = GradientHessian(MF, XF)
                G .= project(Msub, Xsub, GF)
                # projecting HESSIAN
                H .= HessFullProjection(Msub, Xsub, GF, HF)
            else
                GF, HF = GradientHessian(MF, XF)
                G .= GF
                H .= HF
            end
            newhess = false
        end
        tr = 1
        tmp = copy(Xsub)
#         print("b")
        # this loop is only for establishing the trust region radius
        while tr <= trmax 
            H_mat = Symmetric(reshape(H, size(H,1)*size(H,2), :))
            G_mat = vec(G)
            delta, info = trs_small(H_mat, G_mat, radius)
            qmins = [dot(G_mat, delta[:,k]) + dot(delta[:,k], H_mat*delta[:,k])/2 for k=1:size(delta,2)]
            qv, qi = findmin(qmins)
            # doing the retraction
            if manifold
                Dp = project(Msub, Xsub, reshape(delta[:,qi], size(G)))
                # put it back after projection
                Xsub .= retract(Msub, Xsub, Dp, retraction)
                UpdateCachePartial(Cache)
            else
                # without projection
                Xsub .+= reshape(delta[:,qi], size(G))
                UpdateCachePartial(Cache)
            end
            nloss = Loss(MF, XF)
            rho = (loss - nloss) / ( - dot(delta[:,qi], G_mat) - dot(delta[:,qi], H_mat*delta[:,qi])/2)
            if nloss >= loss
                # force a decreasing radius
                rho = 0
            end
            if rho < 0.25
                # if radius can be lowered, try again
                if radius > 1e-6 
                    # try another Iteration
                    radius /= 4
                    Xsub .= tmp
                    UpdateCachePartial(Cache)
                    print("-")
                else
                    # did not converge
                    if nloss < loss
                        # but loss has decreased
                        loss = nloss
                        print("L")
                        break
                    else
                        # loss did not decrease
                        Xsub .= tmp
                        UpdateCachePartial(Cache)
#                         loss = nloss
                        stop = true
                        print("d")
                        print(" rho=", rho, "num=", (loss - nloss), " den=", ( - dot(delta[:,qi], G_mat) - dot(delta[:,qi], H_mat*delta[:,qi])/2))
                        break
                    end
                end
            elseif rho > 0.75
                # if radius can be increased, try again
                if (2*radius < radius_max) && (tr < trmax)
                    # try another Iteration
                    radius *= 2
                    Xsub .= tmp
                    UpdateCachePartial(Cache)
                    print("+")
                else
                    if nloss < loss
                        # but loss has decreased
                        loss = nloss
                        print("H")
                        break
                    else    
                        # loss did not decrease
                        Xsub .= tmp
                        UpdateCachePartial(Cache)
    #                     loss = nloss
                        stop = true
                        print("D")
                        print(" rho=", rho, "num=", (loss - nloss), " den=", ( - dot(delta[:,qi], G_mat) - dot(delta[:,qi], H_mat*delta[:,qi])/2))
                        break
                    end
                end
            else
                # the straightforward case 0.25 < rho < 0.75
                if nloss < loss
                    # but loss has decreased
                    loss = nloss
                    print("=")
                    break
                else    
                    # loss did not decrease
                    Xsub .= tmp
                    UpdateCachePartial(Cache)
#                     loss = nloss
                    stop = true
                    print("M")
                    break
                end
            end
            tr += 1
        end
        if manifold
            GF = Gradient(MF, XF)
            G .= project(Msub, Xsub, GF)
        else
            G .= Gradient(MF, XF)
        end
#         print("e")
        if (norm(G) < 1e-9) # || (norm(G)/ng < 2^(-5))
            break
        end
        if stop
            print("->STOP")
            break
        end
    end
    print("\n    L=", @sprintf("%.6e", loss), " G=", @sprintf("%.6e", ng), " after -> G=", @sprintf("%.6e", norm(G)), " r=", @sprintf("%.6e", norm(G)/ng), " Ginf=", @sprintf("%.6e", maximum(abs.(G))))
    return radius
end

# Tstep: time step between samples
# nearest: select the pair of eigenvalues closest to 'nearest'
# it excludes the said frequency and returns the complement instead
# return : X -> steady state
#          S1 -> linear part of the 2D ROM
#          U1 -> left invariant subspace (orthogonal)
#          W1 -> right invariant subspace (orthogonal), such that U1 * W1 = I
#          S2 -> the complementary 
#          U2 -> left invariant subspace (orthogonal)
#          W2 -> right invariant subspace (orthogonal), such that U1 * W1 = I
# returns: S1 : the linear part of the ROM
#          U1tr: the linear part of U transposed (left invariant subspace)
#          W1: right invariant subspace
#          Wperp: the complement of the left invariant subspace
#          Sperp: the dynamics on the left invariant subspace
function LinearFit(dataIN, dataOUT, Tstep, nearest)
    avgIN = sum(dataIN, dims=2) / size(dataIN,2)
    avgOUT = sum(dataOUT, dims=2) / size(dataOUT,2)
    XXt = (dataIN .- avgIN) * transpose(dataIN)
    YXt = (dataOUT .- avgOUT) * transpose(dataIN)
    # the first approximation
    A = YXt * inv(XXt)
    Xhat = -A * avgIN - avgOUT
    Xstar = (I - A) \ Xhat
#     println("Steady state 0")
#     display(Xstar)

    # again with scaling until converges. 
    # It is recursive because scaling is calculated based on previous estimate of steady state
    for k=1:3
        scale = sqrt.(AmplitudeScaling(dataIN, Xstar))
        avgIN = sum(dataIN .* scale, dims=2) / size(dataIN,2)
        avgOUT = sum(dataOUT .* scale, dims=2) / size(dataOUT,2)
        XXt = ((dataIN .- avgIN).*scale) * transpose(dataIN)
        YXt = ((dataOUT .- avgOUT).*scale) * transpose(dataIN)
        # scaled approximation
        A .= YXt * inv(XXt)
        Xhat .= -A * avgIN - avgOUT
        Xstar .= (I - A) \ Xhat
#         println("Steady state ", k)
#         display(Xstar)
    end
    
    F0 = schur(A)
    args = abs.(angle.(F0.values))
    println("------ START FITTING LINEAR MODEL ------")
    println("All frequencies [Hz]")
    println(unique(sort(args))/Tstep/(2*pi))
    println("All frequencies [1/rad]")
    println(unique(sort(args))/Tstep)
    # find the closest frequency to the specification
    mn, sel = findmin(abs.(args/Tstep/(2*pi) .- nearest))
    # find all the eigenvalues that has the same frequency
    ids = findall(x -> isapprox(args[sel], x), args)
    
    # right eigenvectors
    right_sel = zeros(Bool, size(args))
    right_sel[ids] .= true
    FR = ordschur(F0, right_sel)
    # right invariant subspace
    _W1 = FR.vectors[:,1:length(ids)]
    _U2 = transpose(FR.vectors[:,length(ids)+1:end])
    
    # left eigenvectors
    left_sel = .~ right_sel
    FL = ordschur(F0, left_sel)
    # left invariant subspace
    _W2 = FL.vectors[:,1:end-length(ids)]
    _U1 = transpose(FL.vectors[:,end-length(ids)+1:end])

    _S1 = (_U1*A)*transpose(_U1)
    _S2 = (_U2*A)*transpose(_U2)
    
    println("S1 frequencies [Hz] = ", sort(unique(abs.(angle.(eigvals(_S1))/(2*pi))))/Tstep)
    println("S1 frequencies [rad/s] = ", sort(unique(abs.(angle.(eigvals(_S1)))))/Tstep)
    println("Sperp frequencies [Hz] = ", sort(unique(abs.(angle.(eigvals(_S2))/(2*pi))))/Tstep)
    println("Sperp frequencies [rad/s] = ", sort(unique(abs.(angle.(eigvals(_S2)))))/Tstep)
    println("------ END FITTING LINEAR MODEL ------")
    
    return (X=Xstar, S1=_S1, U1=_U1, W1=_W1, S2=_S2, U2=_U2, W2=_W2)
end

@doc raw"""
    GaussSouthwellLinearSetup(Misf, Xisf, dataINorig, dataOUTorig, Tstep, nearest; perbox=2000, retbox=4, nbox=10, exclude = false)
    
Sets up the invariant foliation with linear estimates. The linear dynamics is estimated by the matrix ``\boldsymbol{A}``, the left invariant subspace is approximated by the orthogonal matrix ``\boldsymbol{U}_1``, the right invariant subspace is approximated by the orthogonal matrix ``\boldsymbol{W}_1``. The linearised dynamics is the matrix ``\boldsymbol{S}_1``, such that ``\boldsymbol{U}_1 \boldsymbol{A}=\boldsymbol{S}_1 \boldsymbol{U}_1`` and ``\boldsymbol{A} \boldsymbol{W}_1=\boldsymbol{W}_1 \boldsymbol{S}_1``.

The routine then sets ``D \boldsymbol{U} (0) = \boldsymbol{U}_1`` and ``D \boldsymbol{P} (0) = \boldsymbol{S}_1``. It also sets the constrint that ``\boldsymbol{U} (\boldsymbol{W}_1 \boldsymbol{z})`` is linear.
"""
function GaussSouthwellLinearSetup(Misf, Xisf, dataINorig, dataOUTorig, Tstep, nearest; perbox=2000, retbox=4, nbox=10, exclude = false)
    dataINlin, dataOUTlin, linscale = dataPrune(dataINorig, dataOUTorig; perbox=perbox, retbox=retbox, nbox=nbox)
    X, S1, U1, W1, S2, U2, W2 = LinearFit(dataINlin, dataOUTlin, Tstep, nearest)
    @show norm(U2 * W1)
    if ~exclude
        # standard 2dim subspace
        @show PadeUpoint(Xisf).parts[1] .= -U1 * X
        PadeUpoint(Xisf).parts[2] .= U1'
        setLinearPart!(PadeP(Misf), PadePpoint(Xisf), S1)
        # the restriction is by the remaining eigenvectors
        ISFPadeSetRestriction(Misf, W1)
        Misf.Xstar .= X
    else
        # standard n-2 dim subspace of the complementary dynamics
        PadeUpoint(Xisf).parts[2] .= U2'
        setLinearPart!(PadeP(Misf), PadePpoint(Xisf), Sperp)
        ISFPadeSetRestriction(Misf, W2)
        Misf.Xstar = X
    end
    return S2, U2, W2, X
end

mutable struct DebugEntryNorm <: DebugAction
    io::IO
    format::String
    field::Symbol
    function DebugEntryNorm(f::Symbol; prefix="$f:", format="$prefix %s", io::IO=stdout)
        return new(io, format, f)
    end
end

function (d::DebugEntryNorm)(::Problem, o::Options, i)
    (i >= 0) && Printf.format(d.io, Printf.Format(d.format), norm(getfield(o, d.field)))
    return nothing
end


@doc raw"""
    GaussSouthwellOptim(Misf, Xisf, dataIN, dataOUT, scale, Tstep, nearest; name = "", maxit=8000, gradstop = 1e-10)
    
Solves the optimisation problem
```math
\arg\min_{\boldsymbol{S},\boldsymbol{U}}\sum_{k=1}^{N}\left\Vert \boldsymbol{x}_{k}\right\Vert ^{-2}\left\Vert \boldsymbol{P}\left(\boldsymbol{U}\left(\boldsymbol{x}_{k}\right)\right)-\boldsymbol{Q}\left(\boldsymbol{U}\left(\boldsymbol{y}_{k}\right)\right)\right\Vert ^{2}.
```

The method is block coordinate descent, where we optimise for matrix coefficients of the representation in a cyclic manner and also by choosing the coefficient whose gradient is the gratest for optimisation.
"""
function GaussSouthwellOptim(Misf, Xisf, dataIN, dataOUT, scale, Tstep, nearest; name = "", maxit=8000, gradstop = 1e-10)
    nl_start = typeof(PadeU(Misf)).parameters[4]
    Cache = makeCache(Misf, Xisf, dataIN, dataOUT)
#     loss = ISFPadeLoss(Misf, Xisf, dataIN, dataOUT)
#     println("Initial L=", loss)
    UPD_GRAD = 2 + length(PadeUpoint(Xisf).parts) 
    radius = 0.5
    radius_max = 10.0
    updateCache!(Cache, Misf, Xisf, dataIN, dataOUT)
    grall = ISFPadeRiemannianGradient(Misf, Xisf, dataIN, dataOUT; DV=Cache)
    grall .= project(Misf, Xisf, grall)
    loss_table = []
    id = (1,0,0)
    for it=1:maxit
        if mod(it,UPD_GRAD) == 0
            updateCache!(Cache, Misf, Xisf, dataIN, dataOUT)
            grall .= ISFPadeRiemannianGradient(Misf, Xisf, dataIN, dataOUT; DV=Cache)
            grall .= project(Misf, Xisf, grall)
        end
        mx, id = MaximumNorm(grall)
        #         id = NextMaximumNorm(PadeU(M), grall, id)
#         if mx < gradstop
#             println("reached gradient below threshold. ", mx)
#             break
#         end
        if mod(it, UPD_GRAD) == UPD_GRAD-1
            id = (1,0,0)
        end
        if id[1] == 1
            # P
            grall.parts[1] .= 0
            print(it, ". P ")
            # this is linear so the radius can be anything
            #=radius ==# trust_region!(Misf, Xisf, 
                    (MF, XF) -> ISFPadeLoss(MF, XF, dataIN, dataOUT; DV=Cache),
                    (MF, XF) -> ISFPadeGradientP(MF, XF, dataIN, dataOUT; DV=Cache),
                    (MF, XF) -> ISFPadeGradientHessianP(MF, XF, dataIN, dataOUT; DV=Cache),
                    Cache, (c) -> (),
                    PadeP(Misf), PadePpoint(Xisf), PadeP(Misf).R,
                    radius_max, radius_max; itmax = 20, manifold = true)
            # the frequencies can only change here
            ev = eigvals(getLinearPart(PadeP(Misf), PadePpoint(Xisf)))
            print("\tDAMP ")
            map(print, [@sprintf("%.6e, ",-a) for a in log.(unique(abs.(ev)))./unique(abs.(angle.(ev)))])
            print("FRQ [Hz] ")
            map(print, [@sprintf("%.6e, ",a/Tstep/(2*pi)) for a in unique(abs.(angle.(ev)))])
            print("FRQ [r/s] ")
            map(print, [@sprintf("%.6e, ",a/Tstep) for a in unique(abs.(angle.(ev)))])
            print("  ", name)
        elseif id[1] == 2
            if length(PadeQpoint(Xisf)) > 0
                # Q
                grall.parts[2] .= 0
                print(it, ". Q ")
                #=radius ==# trust_region!(Misf, Xisf, 
                        (MF, XF) -> ISFPadeLoss(MF, XF, dataIN, dataOUT; DV=Cache),
                        (MF, XF) -> ISFPadeGradientQ(MF, XF, dataIN, dataOUT; DV=Cache),
                        (MF, XF) -> ISFPadeGradientHessianQ(MF, XF, dataIN, dataOUT; DV=Cache),
                        Cache, (c) -> (),
                        PadeQ(Misf), PadeQpoint(Xisf), PadeQ(Misf).R,
                        radius_max, radius_max; itmax = 20, manifold = true)
            end
        elseif id[1] == 3
            # U linear
            if id[2] < nl_start
                ord = id[2]
                ii = 1
                grall.parts[3].parts[ord] .= 0
                print(it, ". U ", ord-1, " ii ", ii)
                # The alternative style -- gradient calculation
#                 res = trust_regions(
#                     PadeU(Misf).M.manifolds[ord], 
#                     (M, x) -> ISFPadeLoss(Misf, Xisf, dataIN, dataOUT; DV=Cache, Xnew=x, ord=ord, ii=ii), 
#                     (M, x) -> project(PadeU(Misf).M.manifolds[ord], PadeUpoint(Xisf).parts[ord], 
#                                       ISFPadeGradientU(Misf, Xisf, dataIN, dataOUT, ord, ii; DV=Cache, Xnew=x)),
# #                     (M, x, Xp) -> ISFPadeHessianVectorU(Misf, Xisf, Xp, dataIN, dataOUT, ord, ii; DV=Cache, Xnew=x),
#                     ApproxHessianFiniteDifference(
#                         PadeU(Misf).M.manifolds[ord],
#                         PadeUpoint(Xisf).parts[ord],
#                         (M, x) -> project(PadeU(Misf).M.manifolds[ord], PadeUpoint(Xisf).parts[ord], 
#                                           ISFPadeGradientU(Misf, Xisf, dataIN, dataOUT, ord, ii; DV=Cache, Xnew=x));
#                         steplength=2^(-20),
#                         retraction_method=PadeU(Misf).R.retractions[ord],
#                         vector_transport_method=PadeU(Misf).mlist[ord].VT
#                         ),
#                     PadeUpoint(Xisf).parts[ord],
#                     retraction_method = PadeU(Misf).R.retractions[ord],
#                     stopping_criterion=StopWhenAny(
#                         StopWhenGradientNormLess(1e-8),
#                         StopAfterIteration(20)
#                         ),
#                     debug = [:Iteration, " | ", DebugCost(;format="L=%.4e "), DebugEntryNorm(:gradient; format="G=%.4e "), "\n"]
#                     )
#                 PadeUpoint(Xisf).parts[ord] .= res
                # The old style -- direct hessian calculation
                radius = trust_region!(Misf, Xisf, 
                        (MF, XF) -> ISFPadeLoss(MF, XF, dataIN, dataOUT; DV=Cache),
                        (MF, XF) -> ISFPadeGradientU(MF, XF, dataIN, dataOUT, ord, ii; DV=Cache),
                        (MF, XF) -> ISFPadeGradientHessianU(MF, XF, dataIN, dataOUT, ord, ii; DV=Cache),
                        Cache, (c) -> (),
                        PadeU(Misf).mlist[ord], PadeUpoint(Xisf).parts[ord], PadeU(Misf).R.retractions[ord],
                        radius, radius_max; itmax = 20, manifold = true)
                # printing the steady state
                if ord == 1
                    println("Steady state = ", PadeUpoint(Xisf).parts[ord])
                end
            else
                # this is the most complicated part
                ord = id[2]
                ii = id[3]
                grall.parts[3].parts[ord].parts[ii] .= 0
                print(it, ". U ", ord-1, " ii ", ii)
                # The alternative style -- gradient calculation
#                 res = trust_regions(
#                     PadeU(Misf).M.manifolds[ord].manifolds[ii], 
#                     (M, x) -> ISFPadeLoss(Misf, Xisf, dataIN, dataOUT; DV=Cache, Xnew=x, ord=ord, ii=ii), 
#                     (M, x) -> project(PadeU(Misf).M.manifolds[ord].manifolds[ii], PadeUpoint(Xisf).parts[ord].parts[ii], 
#                                       ISFPadeGradientU(Misf, Xisf, dataIN, dataOUT, ord, ii; DV=Cache, Xnew=x)),
#                     (M, x, Xp) -> ISFPadeHessianVectorU(Misf, Xisf, Xp, dataIN, dataOUT, ord, ii; DV=Cache, Xnew=x),
# #                     ApproxHessianFiniteDifference(
# #                         PadeU(Misf).M.manifolds[ord].manifolds[ii],
# #                         PadeUpoint(Xisf).parts[ord].parts[ii],
# #                         (M, x) -> project(PadeU(Misf).M.manifolds[ord].manifolds[ii], PadeUpoint(Xisf).parts[ord].parts[ii], 
# #                                           ISFPadeGradientU(Misf, Xisf, dataIN, dataOUT, ord, ii; DV=Cache, Xnew=x));
# #                         steplength=2^(-20),
# #                         retraction_method=PadeU(Misf).R.retractions[ord].retractions[ii],
# #                         vector_transport_method=PadeU(Misf).mlist[ord].VT.methods[ii]
# #                         ),
#                     PadeUpoint(Xisf).parts[ord].parts[ii],
#                     retraction_method = PadeU(Misf).R.retractions[ord].retractions[ii],
#                     stopping_criterion=StopWhenAny(
#                         StopWhenGradientNormLess(1e-8),
#                         StopAfterIteration(20)
#                         ),
#                     debug = [:Iteration, " | ", DebugCost(;format="L=%.4e "), DebugEntryNorm(:gradient; format="G=%.4e "), "\n"]
#                     )
#                 PadeUpoint(Xisf).parts[ord].parts[ii] .= res
                # The old style -- direct hessian calculation
                radius = trust_region!(Misf, Xisf, 
                        (MF, XF) -> ISFPadeLoss(MF, XF, dataIN, dataOUT; DV=Cache),
                        (MF, XF) -> ISFPadeGradientU(MF, XF, dataIN, dataOUT, ord, ii; DV=Cache),
                        (MF, XF) -> ISFPadeGradientHessianU(MF, XF, dataIN, dataOUT, ord, ii; DV=Cache),
                        Cache, (cc) -> updateCachePartial!(cc, Misf, Xisf, dataIN, dataOUT, ord, ii),
                        PadeU(Misf).M.manifolds[ord].manifolds[ii], PadeUpoint(Xisf).parts[ord].parts[ii], PadeU(Misf).R.retractions[ord].retractions[ii],
                        radius, radius_max; itmax = 20, manifold = true)
            end
        end
        infloss = ISFPadeLossInfinity(Misf, Xisf, dataIN, dataOUT; DV=Cache)
        println(" mx=", @sprintf("%.4e", mx), " E_max=", @sprintf("%.4e", infloss[1]), " E_avg=", @sprintf("%.4e", infloss[2]))
        push!(loss_table, (time=time(), E_max = infloss[1], E_avg = infloss[2], loss = infloss[3]))
        @save "ISFdata-$(name).bson" Misf Xisf Tstep scale it grall loss_table
    end
    return Misf, Xisf
end

#----------------------------------------------------------------------------------------------------------------------------------------
# END ISFPadeManifold
#----------------------------------------------------------------------------------------------------------------------------------------
