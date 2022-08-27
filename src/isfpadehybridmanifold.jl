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
function AmplitudeScaling(dataIN)
    return ampsq = sum(dataIN .^ 2, dims = 1)*2^(-7)
end

"defines the penalty term in optimisation"
const LAMBDA = 0

struct ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, 𝔽} <: AbstractManifold{𝔽}
    mlist
    M        :: ProductManifold 
    R        :: ProductRetraction 
    VT       :: ProductVectorTransport
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
            # the linear part
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

function NextMaximumNorm(X::ProductRepr, id)
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
        if id[2] < length(X.parts[id[1]].parts)
            val, ix = findmax(map(a -> sqrt(sum(a .^ 2)), X.parts[id[1]].parts[id[2]+1].parts))
            # the linear part
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
function ISFPadeManifold(mdim, ndim, Porder, Qorder, Uorder, B=nothing, field::AbstractNumbers=ℝ)
    mlist = (DenseNonconstManifold(mdim, mdim, Porder), DenseNearIdentityManifold(mdim, mdim, Qorder), SubmersionManifold(mdim, ndim, Uorder, B))
    M = ProductManifold(mlist...)
    R = ProductRetraction(map(x->getfield(x,:R), mlist)...)
    VT = ProductVectorTransport(map(x->getfield(x,:VT), mlist)...)
    return ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}(mlist, M, R, VT)
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
    X::ProductRepr
    Y::ProductRepr
end

function makeCache(M::ISFPadeManifold, X, dataIN, dataOUT)
    return XYcache(makeCache(PadeU(M), PadeUpoint(X), [dataIN]), makeCache(PadeU(M), PadeUpoint(X), [dataOUT]))
end

function updateCache!(DV::XYcache, M::ISFPadeManifold, X, dataIN, dataOUT, ord, ii)
#     tensorVecsInvalidate(DV.X.parts[ord], PadeU(M).mlist[ord], ii)
#     tensorVecsInvalidate(DV.Y.parts[ord], PadeU(M).mlist[ord], ii)
#     tensorBVecsInvalidate(DV.X.parts[ord], PadeU(M).mlist[ord], ii)
#     tensorBVecsInvalidate(DV.Y.parts[ord], PadeU(M).mlist[ord], ii)
#     tensorBVecsInvalidate(DV.X.parts[ord], PadeU(M).mlist[ord], 0) # surely, L0 has also changed!
#     tensorBVecsInvalidate(DV.Y.parts[ord], PadeU(M).mlist[ord], 0) # surely, L0 has also changed!
    updateCache!(DV.X, PadeU(M), PadeUpoint(X), [dataIN])
    updateCache!(DV.Y, PadeU(M), PadeUpoint(X), [dataOUT])
    return nothing
end

function updateCachePartial!(DV::XYcache, M::ISFPadeManifold, X, dataIN, dataOUT, ord, ii)
    tensorVecsInvalidate(DV.X.parts[ord], PadeU(M).mlist[ord], ii)
    tensorVecsInvalidate(DV.Y.parts[ord], PadeU(M).mlist[ord], ii)
    tensorBVecsInvalidate(DV.X.parts[ord], PadeU(M).mlist[ord], ii)
    tensorBVecsInvalidate(DV.Y.parts[ord], PadeU(M).mlist[ord], ii)
    tensorBVecsInvalidate(DV.X.parts[ord], PadeU(M).mlist[ord], 0) # surely, L0 has also changed!
    tensorBVecsInvalidate(DV.Y.parts[ord], PadeU(M).mlist[ord], 0) # surely, L0 has also changed!
    updateCachePartial!(DV.X, PadeU(M), PadeUpoint(X), [dataIN], ord, ii)
    updateCachePartial!(DV.Y, PadeU(M), PadeUpoint(X), [dataOUT], ord, ii)
    return nothing
end

function ISFPadeLoss(M::ISFPadeManifold, X, dataIN, dataOUT; DV=makeCache(M, X, dataIN, dataOUT))
    datalen = size(dataIN,2)
    reg = 0.0
    for k=2:length(PadeUpoint(X).parts)
#         reg += LAMBDA * sum(abs.(PadeUpoint(X).parts[k].parts[1]))
        # L2
        reg += LAMBDA * sum(abs.(PadeUpoint(X).parts[k].parts[1]) .^2)
    end

    scale = AmplitudeScaling(dataIN)

    Uoy = Eval(PadeU(M), PadeUpoint(X), [dataOUT]; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), [dataIN]; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), [Uoy])
    PoUox = Eval(PadeP(M), PadePpoint(X), [Uox])
    L0 = QoUoy .- PoUox
    print(".")
    return sum( (L0 .^ 2) ./ scale )/2/datalen + reg/2
end

function ISFPadeLossInfinity(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, X, dataIN, dataOUT) where {mdim, ndim, Porder, Qorder, Uorder, field}
    datalen = size(dataIN,2)
#     reg = 0.0
#     for k=2:length(PadeUpoint(X).parts)
# #         reg += LAMBDA * sum(abs.(PadeUpoint(X).parts[k].parts[1]))
#         # L2
#         reg += LAMBDA * sum(abs.(PadeUpoint(X).parts[k].parts[1]) .^2)
#     end

    scale = AmplitudeScaling(dataIN)

    Uoy = Eval(PadeU(M), PadeUpoint(X), [dataOUT])
    Uox = Eval(PadeU(M), PadeUpoint(X), [dataIN])
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), [Uoy])
    PoUox = Eval(PadeP(M), PadePpoint(X), [Uox])
    L0 = QoUoy .- PoUox
    
    return maximum(abs.( L0 ))
end

function ISFPadeGradientP(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, X, dataIN, dataOUT; DV=makeCache(M, X, dataIN, dataOUT)) where {mdim, ndim, Porder, Qorder, Uorder, field}
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN)

    Uoy = Eval(PadeU(M), PadeUpoint(X), [dataOUT]; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), [dataIN]; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), [Uoy])
    PoUox = Eval(PadeP(M), PadePpoint(X), [Uox])
    L0 = (QoUoy .- PoUox) ./ scale

    return L0_DF(PadeP(M), PadePpoint(X), nothing, Uox, -1.0*L0, nothing)/datalen
end

function ISFPadeGradientQ(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, X, dataIN, dataOUT; DV=makeCache(M, X, dataIN, dataOUT)) where {mdim, ndim, Porder, Qorder, Uorder, field}
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN)

    Uoy = Eval(PadeU(M), PadeUpoint(X), [dataOUT]; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), [dataIN]; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), [Uoy])
    PoUox = Eval(PadeP(M), PadePpoint(X), [Uox])
    L0 = (QoUoy .- PoUox) ./ scale

    return L0_DF(PadeQ(M), PadeQpoint(X), nothing, Uoy, L0, nothing)/datalen
end

function gradUmonomial(M::LinearManifold, X, L0_JPoUox, L0_JQoUoy, dataIN, dataOUT, _p1, _p2)
    datalen = size(dataIN,2)
    return (L0_DF(M, X, nothing, dataOUT, L0_JQoUoy, nothing) .- L0_DF(M, X, nothing, dataIN, L0_JPoUox, nothing))/datalen
end

# Calculate the gradient with respect to a node in U
# ord -> monomial order
# ii  -> index of the node 
function ISFPadeGradientU(M::ISFPadeManifold, X, dataIN, dataOUT, ord, ii; DV=makeCache(M, X, dataIN, dataOUT))
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN)

    Uoy = Eval(PadeU(M), PadeUpoint(X), [dataOUT]; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), [dataIN]; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), [Uoy])
    PoUox = Eval(PadeP(M), PadePpoint(X), [Uox])
    L0 = (QoUoy .- PoUox) ./ scale
    
    # the Jacobians call tensorVecs for each leaf once, so they are expensive
    JQoUoy = Jacobian(PadeQ(M), PadeQpoint(X), Uoy)
    JPoUox = Jacobian(PadeP(M), PadePpoint(X), Uox)
    L0_JQoUoy = dropdims(sum(JQoUoy .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    L0_JPoUox = dropdims(sum(JPoUox .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    if ord == 1
        return (L0_DF(PadeU(M).mlist[1], PadeUpoint(X).parts[1], nothing, dataOUT, L0_JQoUoy, ii) .- L0_DF(PadeU(M).mlist[1], PadeUpoint(X).parts[1], nothing, dataIN, L0_JPoUox, ii))/datalen
    else
#         DVUoy = tensorVecs(PadeU(M).mlist[ord], PadeUpoint(X).parts[ord], [dataOUT])
#         tensorBVecs!(DV.Y, PadeU(M).mlist[ord], PadeUpoint(X).parts[ord])
#         DVUox = tensorVecs(PadeU(M).mlist[ord], PadeUpoint(X).parts[ord], [dataIN])
#         tensorBVecs!(DV.X, PadeU(M).mlist[ord], PadeUpoint(X).parts[ord])
        grad = (L0_DF(PadeU(M).mlist[ord], PadeUpoint(X).parts[ord], DV.Y.parts[ord], dataOUT, L0_JQoUoy, ii) 
            .- L0_DF(PadeU(M).mlist[ord], PadeUpoint(X).parts[ord], DV.X.parts[ord], dataIN, L0_JPoUox, ii) )/datalen
        if ii == 1
#             grad .+= LAMBDA * sign.(PadeUpoint(X).parts[ord].parts[ii]) # L1 regularisation
            grad .+= LAMBDA * PadeUpoint(X).parts[ord].parts[ii] # L2 regularisation
        end
        return grad
    end
#     return gradUmonomial(PadeU(M).mlist[ord], PadeUpoint(X).parts[ord], L0_JPoUox, L0_JQoUoy, dataIN, dataOUT)
end

function gradUmonomial(M::TensorManifold, X, L0_JPoUox, L0_JQoUoy, dataIN, dataOUT, DVUox, DVUoy)
    datalen = size(dataIN,2)
#     DVUoy = tensorVecs(M, X, [dataOUT])
#     tensorBVecs!(DVUoy, M, X)
#     DVUox = tensorVecs(M, X, [dataIN])
#     tensorBVecs!(DVUox, M, X)
    
    # this is missing the penalty term
    function tmp(ii)
        G = (L0_DF(M, X, DVUoy, dataOUT, L0_JQoUoy, ii) .- L0_DF(M, X, DVUox, dataIN, L0_JPoUox, ii))/datalen 
        if ii == 1 
#             return G .+ LAMBDA * sign.(X.parts[ii]) # L1 penalty
            return G .+ LAMBDA * X.parts[ii] # L2 penalty
        else 
            return G
        end
    end
    return ProductRepr(map(tmp, collect(1:length(X.parts)))...)
end

function ISFPadeGradientAll(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, X, dataIN, dataOUT; DV=makeCache(M, X, dataIN, dataOUT)) where {mdim, ndim, Porder, Qorder, Uorder, field}
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN)
    
    Uoy = Eval(PadeU(M), PadeUpoint(X), [dataOUT]; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), [dataIN]; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), [Uoy])
    PoUox = Eval(PadeP(M), PadePpoint(X), [Uox])
    L0 = (QoUoy .- PoUox) ./ scale

    GP = L0_DF(PadeP(M), PadePpoint(X), nothing, Uox, -1.0*L0, nothing)/datalen
    GQ = L0_DF(PadeQ(M), PadeQpoint(X), nothing, Uoy, L0, nothing)/datalen

    # the Jacobians call tensorVecs for each leaf once, so they are expensive
    JQoUoy = Jacobian(PadeQ(M), PadeQpoint(X), Uoy)
    JPoUox = Jacobian(PadeP(M), PadePpoint(X), Uox)
    L0_JQoUoy = dropdims(sum(JQoUoy .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    L0_JPoUox = dropdims(sum(JPoUox .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    
    GU = ProductRepr( map( (M,X, dvx, dvy) -> gradUmonomial(M, X, L0_JPoUox, L0_JQoUoy, dataIN, dataOUT, dvx, dvy), PadeU(M).mlist, PadeUpoint(X).parts, DV.X.parts, DV.Y.parts) )
    print("*")
    return ProductRepr(GP, GQ, GU)
end

function ISFPadeRiemannianGradient(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, X, dataIN, dataOUT; DV=makeCache(M, X, dataIN, dataOUT)) where {mdim, ndim, Porder, Qorder, Uorder, field}
    gr = ISFPadeGradientAll(M, X, dataIN, dataOUT; DV=DV)
    return project(M, X, gr)
end

function ISFPadeGradientHessianP(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, X, dataIN, dataOUT; DV=makeCache(M, X, dataIN, dataOUT)) where {mdim, ndim, Porder, Qorder, Uorder, field}
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN)

    Uoy = Eval(PadeU(M), PadeUpoint(X), [dataOUT]; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), [dataIN]; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), [Uoy])
    PoUox = Eval(PadeP(M), PadePpoint(X), [Uox])
    L0 = (QoUoy .- PoUox) ./ scale
    
#     println("P GRAD TIME")
    grad = L0_DF(PadeP(M), PadePpoint(X), nothing, Uox, -1.0*L0, nothing)/datalen
#     println("P HESS TIME")
    hess = DFoxT_DFox(PadeP(M), PadePpoint(X), Uox, nothing; scale=scale)/datalen
    return grad, hess
end

function ISFPadeGradientHessianQ(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, X, dataIN, dataOUT; DV=makeCache(M, X, dataIN, dataOUT)) where {mdim, ndim, Porder, Qorder, Uorder, field}
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN)

    Uoy = Eval(PadeU(M), PadeUpoint(X), [dataOUT]; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), [dataIN]; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), [Uoy])
    PoUox = Eval(PadeP(M), PadePpoint(X), [Uox])
    L0 = (QoUoy .- PoUox) ./ scale
    
    grad = L0_DF(PadeQ(M), PadeQpoint(X), nothing, Uoy, L0, nothing)/datalen
    hess = DFoxT_DFox(PadeQ(M), PadeQpoint(X), Uoy, nothing; scale=scale)/datalen
    return grad, hess
end

function ISFPadeGradientHessianU(M::ISFPadeManifold{mdim, ndim, Porder, Qorder, Uorder, field}, X, dataIN, dataOUT, ord, ii; DV=makeCache(M, X, dataIN, dataOUT)) where {mdim, ndim, Porder, Qorder, Uorder, field}
#       DUox^T x JPoUox^T x JPoUox x DUox   -> 1
#     + DUoy^T x JQoUoy^T x JQoUoy x DUoy   -> 2
#     - DUox^T x JPoUox^T x JQoUoy x DUoy   -> 3
#     - DUoy^T x JQoUoy^T x JPoUox x DUox   -> 3^T
#     - DUox^T x L0J2PoUox x DUox
#     + DUoy^T x L0J2QoUoy x DUoy
    t = []
    push!(t, time())
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN)

    Uoy = Eval(PadeU(M), PadeUpoint(X), [dataOUT]; DV=DV.Y)
    Uox = Eval(PadeU(M), PadeUpoint(X), [dataIN]; DV=DV.X)
    QoUoy = Eval(PadeQ(M), PadeQpoint(X), [Uoy])
    PoUox = Eval(PadeP(M), PadePpoint(X), [Uox])
    L0 = (QoUoy .- PoUox) ./ scale

    push!(t, time())
    JQoUoy = Jacobian(PadeQ(M), PadeQpoint(X), Uoy)
    JPoUox = Jacobian(PadeP(M), PadePpoint(X), Uox)
    L0_JQoUoy = dropdims(sum(JQoUoy .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    L0_JPoUox = dropdims(sum(JPoUox .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    
    push!(t, time())
    # the hessians or P and Q
#     println("tensorHessian")
    J2QoUoy = Hessian(PadeQ(M), PadeQpoint(X), Uoy)
    J2PoUox = Hessian(PadeP(M), PadePpoint(X), Uox)
    L0J2QoUoy = dropdims(sum(J2QoUoy .* reshape(L0, size(L0,1), 1, 1, size(L0,2)), dims=1), dims=1)
    L0J2PoUox = dropdims(sum(J2PoUox .* reshape(L0, size(L0,1), 1, 1, size(L0,2)), dims=1), dims=1)
    
    if ord == 1
        push!(t, time())
        grad = (L0_DF(PadeU(M).mlist[1], PadeUpoint(X).parts[1], nothing, dataOUT, L0_JQoUoy, ii) .- L0_DF(PadeU(M).mlist[1], PadeUpoint(X).parts[1], nothing, dataIN, L0_JPoUox, ii))/datalen
        hess = DFT_JFT_JF_DF(PadeU(M).mlist[ord], nothing, nothing, JPoUox, JQoUoy, L0J2PoUox, L0J2QoUoy, dataIN, dataOUT, ii; scale=scale)/datalen
    else
#         push!(t, time())
#         DVUoy = tensorVecs(PadeU(M).mlist[ord], PadeUpoint(X).parts[ord], [dataOUT])
#         DVUox = tensorVecs(PadeU(M).mlist[ord], PadeUpoint(X).parts[ord], [dataIN])
#         push!(t, time())
#         tensorBVecs!(DVUoy, PadeU(M).mlist[ord], PadeUpoint(X).parts[ord])
#         tensorBVecs!(DVUox, PadeU(M).mlist[ord], PadeUpoint(X).parts[ord])
#         push!(t, time())
        DVUox = DV.X.parts[ord]
        DVUoy = DV.Y.parts[ord]
        # grad_U = L0^T (JQoUoy x DUoy - JPoUox x DUox)
        grad = (L0_DF(PadeU(M).mlist[ord], PadeUpoint(X).parts[ord], DVUoy, dataOUT, L0_JQoUoy, ii) 
            .- L0_DF(PadeU(M).mlist[ord], PadeUpoint(X).parts[ord], DVUox, dataIN, L0_JPoUox, ii))/datalen
        push!(t, time())
        hess = DFT_JFT_JF_DF(PadeU(M).mlist[ord], DVUox, DVUoy, JPoUox, JQoUoy, L0J2PoUox, L0J2QoUoy, dataIN, dataOUT, ii; scale=scale)/datalen
        if ii == 1
#             grad .+= LAMBDA * sign.(PadeUpoint(X).parts[ord].parts[ii]) # L1 regularisation
            grad .+= LAMBDA * PadeUpoint(X).parts[ord].parts[ii] # L2 regularisation
            hess .+= reshape(Diagonal(LAMBDA*ones(size(hess,1)*size(hess,2))),size(hess)) # L2 regularisation
        end
    end
    push!(t, time())
#     println("GHU times ", size(grad))
#     display(t[2:end] .- t[1:end-1])
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
    let ord = 1, ii = nothing
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
    for ord=2:length(PadeUpoint(x1).parts)
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
    eps = 1e-8
    let ord = 1, ii = nothing
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
            if abs(relErr) > 1e-4
                flag = true
                println("HU o=", ord, " el=", k1, ",", k2, "/", size(XP,1), ",", size(XP,2), " E = ", relErr, " HP=", maximum(abs.(hessP[:,:,k1,k2])), " A=", maximum(abs.(hessPp[:,:,k1,k2])))
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
    for ord=2:length(PadeUpoint(x1).parts)
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
                    println("HU node=", ii, "/", nr_nodes(MP), " el=", k1, ",", k2, "/", size(XP.parts[ii],1), ",", size(XP.parts[ii],2), " E = ", relErr, " HP=", maximum(abs.(hessP[:,:,k1,k2])), " A=", maximum(abs.(hessPp[:,:,k1,k2])))
#                     println("diff")
#                     display(hessPp[:,:,k1,k2] - hessP[:,:,k1,k2])
#                     println("analytic")
#                     display(hessP[:,:,k1,k2])
#                     println("approximate")
#                     display(hessPp[:,:,k1,k2])
                end
                XPp.parts[ii][k1,k2] = XP.parts[ii][k1,k2]
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
        UpdateCachePartial(Cache)
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
#             qi=1
#             print("minima")
#             display(qmins)
#             print("G->",norm(G_mat))
            # doing the retraction
            if manifold
                Dp = project(Msub, Xsub, reshape(delta[:,qi], size(G)))
                # put it back after projection
                delta[:,qi] .= vec(Dp)
                # it really needs to be in the tangent space
#                 @show norm(Dp .- reshape(delta[:,qi], size(G)))
#                 @show dot(G_mat, delta[:,qi]) + dot(delta[:,qi], H_mat*delta[:,qi])/2, dot(G_mat, vec(Dp)) + dot(vec(Dp), H_mat*vec(Dp))/2
                # with projection
                Xsub .= retract(Msub, Xsub, Dp, retraction)
                # without projection
#                 Xsub .= retract(Msub, Xsub, reshape(delta[:,qi], size(G)), retraction)
#                 print(" Dp->",norm(G_mat))
            else
                Xsub .+= reshape(delta[:,qi], size(G))
            end
            UpdateCachePartial(Cache)
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
        if (norm(G) < 1e-9) || (norm(G)/ng < 2^(-5))
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
# returns: S1 : the linear part of the ROM
#          U1tr: the linear part of U transposed (left invariant subspace)
#          W1: right invariant subspace
#          Wperp: the complement of the left invariant subspace
#          Sperp: the dynamics on the left invariant subspace
function LinearFit(dataIN, dataOUT, Tstep, nearest)
    scale = sqrt.(AmplitudeScaling(dataIN))

    A = ((dataOUT.*scale)*transpose(dataIN)) * inv((dataIN.*scale)*transpose(dataIN))

    # vecs: right eigenvectors
    vals, vecs = eigen(A')
    # ivecs: left eigenvectors
    ivecs = inv(vecs)
    cid = findall(!isreal, vals)
    cvals = vals[cid]
    cvecs = vecs[:,cid]
    civecs = ivecs[:,cid]
    args = abs.(angle.(cvals))
    lst = sortperm(args)
    println("------ START FITTING LINEAR MODEL ------")
    println("All frequencies [Hz]")
    println(unique(args[lst])/Tstep/(2*pi))
    println("All frequencies [1/rad]")
    println(unique(args[lst])/Tstep)
    mn, sel = findmin(abs.(abs.(args/Tstep/(2*pi)) .- nearest))

    # LEFT eigenevctors
    vv = cvecs[:,sel]
    tr0 = hcat(real(vv), imag(vv))
    # orthogonalised vectors
    U1tr = Array(qr(tr0).Q)
    # linear ROM
    S1 = (U1tr'*A)*U1tr

    # RIGHT eigenvectors
    ivv = civecs[:,sel]
    itr0 = hcat(real(ivv), imag(ivv))
    # orthogonalised vectors
    W1 = Array(qr(itr0).Q)

    # rest of the LEFT eigenvectors
    rest = findall(abs.(abs.(args) .- abs(args[sel])) .> 1e-6)
    ims = unique(imag.(vecs[:,rest]) * Diagonal(sign.(imag.(vals[rest]))),dims=2)
    res = unique(real.(vecs[:,rest]),dims=2)
    imid = findall(vec(sum(ims.^2, dims=1)) .> eps(1.0))
    if isempty(imid)
        pvec = res
    else
        pvec = [ims;;res]
    end
    Wperp = transpose(Array(qr(pvec).Q))
    # Linear ROM in the rest of directions
    Sperp = (Wperp*A)*Wperp'

    # rest of the RIGHT eigenvectors
    ims = unique(imag.(ivecs[:,rest]) * Diagonal(sign.(imag.(vals[rest]))),dims=2)
    res = unique(real.(ivecs[:,rest]),dims=2)
    imid = findall(vec(sum(ims.^2, dims=1)) .> eps(1.0))
    if isempty(imid)
        pvec = res
    else
        pvec = [ims;;res]
    end
    W1rest = transpose(Array(qr(pvec).Q))
        
    println("S1 frequencies [Hz] = ", sort(unique(abs.(angle.(eigvals(S1))/(2*pi))))/Tstep)
    println("S1 frequencies [rad/s] = ", sort(unique(abs.(angle.(eigvals(S1)))))/Tstep)
    println("Sperp frequencies [Hz] = ", sort(unique(abs.(angle.(eigvals(Sperp))/(2*pi))))/Tstep)
    println("Sperp frequencies [rad/s] = ", sort(unique(abs.(angle.(eigvals(Sperp)))))/Tstep)
    println("------ END FITTING LINEAR MODEL ------")
    
    return S1, U1tr, W1, Sperp, Wperp, W1rest
end

@doc raw"""
    GaussSouthwellLinearSetup(Misf, Xisf, dataINorig, dataOUTorig, Tstep, nearest; perbox=2000, retbox=4, nbox=10, exclude = false)
    
Sets up the invariant foliation with linear estimates. The linear dynamics is estimated by the matrix ``\boldsymbol{A}``, the left invariant subspace is approximated by the orthogonal matrix ``\boldsymbol{U}_1``, the right invariant subspace is approximated by the orthogonal matrix ``\boldsymbol{W}_1``. The linearised dynamics is the matrix ``\boldsymbol{S}_1``, such that ``\boldsymbol{U}_1 \boldsymbol{A}=\boldsymbol{S}_1 \boldsymbol{U}_1`` and ``\boldsymbol{A} \boldsymbol{W}_1=\boldsymbol{W}_1 \boldsymbol{S}_1``.

The routine then sets ``D \boldsymbol{U} (0) = \boldsymbol{U}_1`` and ``D \boldsymbol{P} (0) = \boldsymbol{S}_1``. It also sets the constrint that ``\boldsymbol{U} (\boldsymbol{W}_1 \boldsymbol{z})`` is linear.
"""
function GaussSouthwellLinearSetup(Misf, Xisf, dataINorig, dataOUTorig, Tstep, nearest; perbox=2000, retbox=4, nbox=10, exclude = false)
    dataINlin, dataOUTlin, linscale = dataPrune(dataINorig, dataOUTorig; perbox=perbox, retbox=retbox, nbox=nbox)
    S1, U1tr, W1, Sperp, Wperp, W1rest = LinearFit(dataINlin, dataOUTlin, Tstep, nearest)
    if ~exclude
        # standard 2dim subspace
        PadeUpoint(Xisf).parts[1] .= U1tr
        setLinearPart!(PadeP(Misf), PadePpoint(Xisf), S1)
        # the restriction is by the remaining eigenvectors
#         ISFPadeSetRestriction(Misf, W1)
        # the restriction is the normal direction
        ISFPadeSetRestriction(Misf, U1tr)
    else
        # standard n-2 dim subspace of the complementary dynamics
        PadeUpoint(Xisf).parts[1] .= Wperp'
        setLinearPart!(PadeP(Misf), PadePpoint(Xisf), Sperp)
        ISFPadeSetRestriction(Misf, W1rest)
    end
    return Sperp, Wperp, W1rest
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
    Cache = makeCache(Misf, Xisf, dataIN, dataOUT)
    loss = ISFPadeLoss(Misf, Xisf, dataIN, dataOUT)
    println("Initial L=", loss)
    UPD_GRAD = 3
    radius = 0.5
    radius_max = 10.0
    grall = ISFPadeRiemannianGradient(Misf, Xisf, dataIN, dataOUT; DV=Cache)
    grall .= project(Misf, Xisf, grall)
    loss_table = ones(2*UPD_GRAD+1)
    id = (1,0,0)
    for it=1:maxit
        if mod(it,UPD_GRAD) == 0
            grall .= ISFPadeRiemannianGradient(Misf, Xisf, dataIN, dataOUT; DV=Cache)
            grall .= project(Misf, Xisf, grall)
        end
#         mx, id = MaximumNorm(grall)
        mx = 1.0
        id = NextMaximumNorm(grall, id)
        if mx < gradstop
            println("reached gradient below threshold. ", mx)
            break
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
            if id[2] == 1
                ord = id[2]
                ii = 1
                grall.parts[3].parts[1] .= 0
                print(it, ". U ", ord, " ii ", ii)
                radius = trust_region!(Misf, Xisf, 
                        (MF, XF) -> ISFPadeLoss(MF, XF, dataIN, dataOUT; DV=Cache),
                        (MF, XF) -> ISFPadeGradientU(MF, XF, dataIN, dataOUT, ord, ii; DV=Cache),
                        (MF, XF) -> ISFPadeGradientHessianU(MF, XF, dataIN, dataOUT, ord, ii; DV=Cache),
                        Cache, (c) -> (),
                        PadeU(Misf).mlist[ord], PadeUpoint(Xisf).parts[ord], PadeU(Misf).R.retractions[ord],
                        radius, radius_max; itmax = 20, manifold = true)
            else
                # this is the most complicated part
                # The old style -- direct hessian calculation
                ord = id[2]
                ii = id[3]
                grall.parts[3].parts[ord].parts[ii] .= 0
                print(it, ". U ", ord, " ii ", ii)
                radius = trust_region!(Misf, Xisf, 
                        (MF, XF) -> ISFPadeLoss(MF, XF, dataIN, dataOUT; DV=Cache),
                        (MF, XF) -> ISFPadeGradientU(MF, XF, dataIN, dataOUT, ord, ii; DV=Cache),
                        (MF, XF) -> ISFPadeGradientHessianU(MF, XF, dataIN, dataOUT, ord, ii; DV=Cache),
                        Cache, (cc) -> updateCachePartial!(cc, Misf, Xisf, dataIN, dataOUT, ord, ii),
                        PadeU(Misf).M.manifolds[ord].manifolds[ii], PadeUpoint(Xisf).parts[ord].parts[ii], PadeU(Misf).R.retractions[ord].retractions[ii],
                        radius, radius_max; itmax = 20, manifold = true)
            end
        end
        infloss = ISFPadeLossInfinity(Misf, Xisf, dataIN, dataOUT)
        println(" Linf=", @sprintf("%.6e", infloss), " mx ", mx)
        @save "ISFdata-$(name).bson" Misf Xisf Tstep scale it grall
        loss_table[1+mod(it,length(loss_table))] = infloss
        if maximum(abs.(loss_table .- loss_table[1])) <= eps(loss_table[1])
            println("Loss did not change, quitting")
            break
        end
    end
    return Misf, Xisf
end
    
mutable struct DebugEig <: DebugAction
    print::Function
    t0
    Tstep
    DebugEig(Tstep, print::Function=print) = new(print, time(), Tstep)
end

function (d::DebugEig)(p::P,o::O,i::Int) where {P <: Problem, O <: Options} 
    d.print(@sprintf("time = %.1f[s] ", time()-d.t0), 
            @sprintf("F(x) = %.4e ", get_cost(p, o.x)), 
            @sprintf("G(x) = %.4e ", norm(p.M,o.x,o.gradient)),
            "|",abs.(eigvals(getLinearPart(p.M.M.manifolds[1],o.x.parts[1]))), 
            angle.(eigvals(getLinearPart(p.M.M.manifolds[1],o.x.parts[1])))/d.Tstep/(2*pi))
end

"""
    This is unused, because it is painfully slow
"""
function TrustRegionOptim(Misf, Xisf, dataINorig, dataOUTorig, datascale, Tstep, nearest; perbox=2000, name = "", maxit=8000, gradstop = 1e-10)
    if perbox > 0
        dataIN, dataOUT, scale = dataPrune(dataINorig, dataOUTorig; perbox=perbox, scale=datascale)
    else
        dataIN = dataINorig
        dataOUT = dataOUTorig
        scale = 1.0
    end
    
#     Xres = quasi_Newton(
#         Misf, 
#         (M, x) -> ISFPadeLoss(M, x, dataIN, dataOUT), 
#         (M, x) -> ISFPadeRiemannianGradient(M, x, dataIN, dataOUT),
#         Xisf;
#         retraction_method = Misf.R,
#         vector_transport_method = Misf.VT,
#         step_size=WolfePowellLineseach(Misf.R, Misf.VT),
#         stopping_criterion=StopAfterIteration(maxit) | StopWhenGradientNormLess(gradstop),
#         debug=[:Iteration, " | ", :Cost, DebugEig(Tstep), "\n", :Stop],
#     )
    
    Xres = trust_regions(Misf, 
                  (M, x) -> ISFPadeLoss(M, x, dataIN, dataOUT), 
                  (M, x) -> ISFPadeRiemannianGradient(M, x, dataIN, dataOUT),
                  ApproxHessianFiniteDifference(
                Misf,
                Xisf,
                (M, x) -> ISFPadeRiemannianGradient(M, x, dataIN, dataOUT);
                steplength=2^(-8),
                retraction_method=Misf.R,
                vector_transport_method=Misf.VT,
            ),
                  Xisf,
                  retraction_method = Misf.R,
#                   max_trust_region_radius=0.5,
                  stopping_criterion=StopWhenAny(
                        StopWhenGradientNormLess(gradstop),
                        StopAfterIteration(maxit)
                        ),
                  debug = [
            :Stop,
            :Iteration,
            " | ",
            DebugEig(Tstep),
            "\n",
            1,
        ])
    
    Xisf .= Xres
    return Misf, Xisf
end

#----------------------------------------------------------------------------------------------------------------------------------------
# END ISFPadeManifold
#----------------------------------------------------------------------------------------------------------------------------------------
