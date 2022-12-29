## ---------------------------------------------------------------------------------------
## Submersion
## 
## ---------------------------------------------------------------------------------------

# U(x) = U0 + U1 x + Unl(x)
# such that U1 is orthogonal
#           Unl(B x) = 0, hence Unl(x) = Unl^hat(P x),
#           where P = B^\perp

function subspaceProjection(B)
    if B == nothing
        return I, 0
    else
        return transpose(qr(B).Q[:,size(B,2)+1:end]), size(B,2)
    end
    nothing
end

struct PolynomialManifold{mdim, ndim, order, nl_start, 𝔽} <: AbstractManifold{𝔽}
    mlist
    M        :: ProductManifold 
    R        :: ProductRetraction
    VT       :: ProductVectorTransport
    P
end

function PolyOrder(M::PolynomialManifold{mdim, ndim, order, nl_start, field}) where {mdim, ndim, order, nl_start, field}
    return order
end

function inner(M::PolynomialManifold{mdim, ndim, order, nl_start, field}, p, X, Y) where {mdim, ndim, order, nl_start, field}
    return inner(M.M, p, X, Y)
end

function project!(M::PolynomialManifold{mdim, ndim, order, nl_start, field}, Y, p, X) where {mdim, ndim, order, nl_start, field}
    return ProductRepr(map(project!, M.mlist, Y.parts, p.parts, X.parts))
end

function retract!(M::PolynomialManifold{mdim, ndim, order, nl_start, field}, q, p, X, method::AbstractRetractionMethod) where {mdim, ndim, order, nl_start, field}
    return retract!(M.M, q, p, X, method)
end

function vector_transport_to!(M::PolynomialManifold{mdim, ndim, order, nl_start, field}, Y, p, X, q, method::AbstractVectorTransportMethod) where {mdim, ndim, order, nl_start, field}
#     println("PolynomialManifold VECTOR TRANSPORT 1")
    return vector_transport_to!(M.M, Y, p, X, q, method)
end

function SubmersionManifold(mdim, ndim, order, B=nothing, field::AbstractNumbers=ℝ; node_rank = 4)
    P, sub_dim = subspaceProjection(B)
    mlist = tuple([ConstantManifold(mdim, field); OrthogonalFlatManifold(ndim, mdim); [HTTensor(repeat([ndim-sub_dim],k), mdim, node_rank = node_rank) for k=2:order]]...)
    nl_start = 3
    M = ProductManifold(map(x->getfield(x,:M), mlist)...)
    R = ProductRetraction(map(x->getfield(x,:R), mlist)...)
    VT = ProductVectorTransport(map(x->getfield(x,:VT), mlist)...)
    return PolynomialManifold{mdim, ndim, order, nl_start, field}(mlist, M, R, VT, P)
end

function PolynomialTallManifold(mdim, ndim, order, B=nothing, field::AbstractNumbers=ℝ; node_rank = 4)
    P, sub_dim = subspaceProjection(B)
    mlist = tuple([LinearTallManifold(ndim, mdim); [HTTensor(repeat([ndim-sub_dim],k), mdim, node_rank = node_rank) for k=2:order]]...)
    nl_start = 2
    M = ProductManifold(map(x->getfield(x,:M), mlist)...)
    R = ProductRetraction(map(x->getfield(x,:R), mlist)...)
    VT = ProductVectorTransport(map(x->getfield(x,:VT), mlist)...)
    return PolynomialManifold{mdim, ndim, order, nl_start, field}(mlist, M, R, VT, P)
end

function zero(M::PolynomialManifold{mdim, ndim, order, nl_start, field}) where {mdim, ndim, order, nl_start, field}
    return ProductRepr(map(zero, M.mlist))
end

function randn(M::PolynomialManifold{mdim, ndim, order, nl_start, field}) where {mdim, ndim, order, nl_start, field}
    return ProductRepr(map(randn, M.mlist))
end

function zero_vector!(M::PolynomialManifold{mdim, ndim, order, nl_start, field}, X, p) where {mdim, ndim, order, nl_start, field}
    return zero_vector!(M.M, X, p)
end

function manifold_dimension(M::PolynomialManifold{mdim, ndim, order, nl_start, field}) where {mdim, ndim, order, nl_start, field}
    return manifold_dimension(M.M)
end

struct PolynomialCache
    projected_data
    tree :: ProductRepr
end

# a) remove the nonsense of encapsulating the data everywhere, even in the tensor
#    it is a remnant when the output was an index, too
#    alternatively think it through and use tuples as they are statically sized
#    what operations would benefit?
# b) cache projected data, to eliminate needless matrix multiplications
function makeCache(M::PolynomialManifold{mdim, ndim, order, nl_start, field}, X, data) where {mdim, ndim, order, nl_start, field}
    projected_data = M.P * data
    tree = ProductRepr((map((x, y) -> makeCache(x, y, data), M.mlist[1:nl_start-1], X.parts[1:nl_start-1])...,
                        map((x, y) -> makeCache(x, y, projected_data), M.mlist[nl_start:end], X.parts[nl_start:end])...))
    return PolynomialCache(projected_data, tree)
#     return tree
end

function updateCache!(DV, M::PolynomialManifold{mdim, ndim, order, nl_start, field}, X, data) where {mdim, ndim, order, nl_start, field}
    DV.projected_data .= M.P * data
    map((c, x, y) -> updateCache!(c, x, y, data), DV.tree.parts[1:nl_start-1], M.mlist[1:nl_start-1], X.parts[1:nl_start-1])
    map((c, x, y) -> updateCache!(c, x, y, DV.projected_data), DV.tree.parts[nl_start:end], M.mlist[nl_start:end], X.parts[nl_start:end])
    return nothing
end

function updateCachePartial!(DV, M::PolynomialManifold{mdim, ndim, order, nl_start, field}, X, data; ord, ii) where {mdim, ndim, order, nl_start, field}
    if ord >= nl_start
        updateCachePartial!(DV.tree.parts[ord], M.mlist[ord], X.parts[ord], DV.projected_data; ii=ii)
    end
    return nothing
end

function Eval(M::PolynomialManifold{mdim, ndim, order, nl_start, field}, X, data; L0 = nothing, DV = makeCache(M, X, data)) where {mdim, ndim, order, nl_start, field}
    # apply projection to only the tensor components
    return (mapreduce((x,y,dv) -> Eval(x, y, data, L0=L0, DV=dv), .+, M.mlist[1:nl_start-1], X.parts[1:nl_start-1], DV.tree.parts[1:nl_start-1]) .+
            mapreduce((x,y,dv) -> Eval(x, y, DV.projected_data, L0=L0, DV=dv), .+, M.mlist[nl_start:end], X.parts[nl_start:end], DV.tree.parts[nl_start:end]))
end

function L0_DF_parts(M::PolynomialManifold{mdim, ndim, order, nl_start, field}, X, data; L0, ord, ii, DV = makeCache(M, X, data)) where {mdim, ndim, order, nl_start, field}
    if ord < nl_start
        return L0_DF(M.mlist[ord], X.parts[ord], data; L0=L0, DV = DV.tree.parts[ord])
    else
        return L0_DF_parts(M.mlist[ord], X.parts[ord], DV.projected_data; L0=L0, ii=ii, DV = DV.tree.parts[ord])
    end
    nothing
end

# L0_DF(M::TensorManifold{field}, X, DV, data, L0, ii)
function L0_DF(M::PolynomialManifold{mdim, ndim, order, nl_start, field}, X, data; L0, DV = makeCache(M, X, data)) where {mdim, ndim, order, nl_start, field}
    return ProductRepr((map((x,y,z) -> L0_DF(x, y, data; L0 = L0, DV = z), M.mlist[1:nl_start-1], X.parts[1:nl_start-1], DV.tree.parts[1:nl_start-1])..., 
                        map((x,y,z) -> L0_DF(x, y, DV.projected_data; L0 = L0, DV = z), M.mlist[nl_start:end], X.parts[nl_start:end], DV.tree.parts[nl_start:end])...))
end

function DF_dt_parts(M::PolynomialManifold{mdim, ndim, order, nl_start, field}, X, data; dt, ord, ii, DV = makeCache(M, X, data)) where {mdim, ndim, order, nl_start, field}
    if ord < nl_start
        return DF_dt_parts(M.mlist[ord], X.parts[ord], data; dt=dt, ii=ii, DV = DV.tree.parts[ord])
    else
        return DF_dt_parts(M.mlist[ord], X.parts[ord], DV.projected_data; dt=dt, ii=ii, DV = DV.tree.parts[ord])
    end
    nothing
end

function DF_dt(M::PolynomialManifold{mdim, ndim, order, nl_start, field}, X, data; dt, DV = makeCache(M, X, data)) where {mdim, ndim, order, nl_start, field}
    return (mapreduce((x,y,z,w) -> DF_dt(x, y, data, dt = z, DV = w), .+, M.mlist[1:nl_start-1], X.parts[1:nl_start-1], dt.parts[1:nl_start-1], DV.tree.parts[1:nl_start-1]) .+ 
            mapreduce((x,y,z,w) -> DF_dt(x, y, DV.projected_data, dt = z, DV = w), .+, M.mlist[nl_start:end], X.parts[nl_start:end], dt.parts[nl_start:end], DV.tree.parts[nl_start:end]))
end

# data: columns are the vectors, same number as size(mexp, 1)
function toDensePolynomial!(Mout::DensePolyManifold{ndim, n, order, field}, Y, Min::PolynomialManifold, X, data) where {ndim, n, order, field}
    @assert size(Mout.mexp, 1) == size(data,2) "Wrong sizes"
    for ord = 1:PolyOrder(Min)
        oid = PolyOrderIndices(Mout, ord)
        for id in oid
            ee = Mout.mexp[:,id]
            cee = cumsum(ee)
            vv = zeros(Int, cee[end])
            vv[1:cee[1]] .= 1
            for k=2:length(cee)
                vv[1+cee[k-1]:cee[k]] .= k
            end
            vperm = unique(permutations(vv))
            res = zero(Y[:,id])
            for p in vperm
                dataARR = [reshape(data[:,ii],:,1) for ii in p]
                @show size(res), size(Eval(Min.mlist[ord], X.parts[ord], dataARR))
                res .+= vec(Eval(Min.mlist[ord], X.parts[ord], dataARR))
            end
            res ./= length(vperm)
            Y[:,id] .= res
        end
    end
    return Y
end
