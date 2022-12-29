## ---------------------------------------------------------------------------------------
## LinearManifold
## represent the liner part of a polynmial map
## ---------------------------------------------------------------------------------------

# ndim: input dimensionality
# n:    output dimensionality
struct LinearManifold{ndim, n, transp, 𝔽} <: AbstractManifold{𝔽}
    M        :: AbstractManifold{𝔽} # Euclidean{Tuple{n, ndim}, 𝔽}
    R        :: AbstractRetractionMethod
    VT       :: AbstractVectorTransportMethod
end

function HessProjection(M::LinearManifold, X, grad, HessV, V)
    HessProjection(M.M, X, grad, HessV, V)
end

function HessFullProjection(M::LinearManifold{ndim, n, transp}, X, grad, hess) where {ndim, n, transp}
    return HessFullProjection(M.M, X, grad, hess)
end

function retract!(M::LinearManifold{ndim, n, transp}, q, p, X, method::AbstractRetractionMethod) where {ndim, n, transp}
    return retract!(M.M, q, p, X, M.R)
end

function retract!(M::LinearManifold{ndim, n, transp}, q, p, X, method::ExponentialRetraction) where {ndim, n, transp}
    return retract!(M.M, q, p, X, M.R)
end

function retract(
    M::LinearManifold,
    p,
    X,
    m::AbstractRetractionMethod = default_retraction_method(M),
)
#     q = allocate_result(M, retract, p, X)
    return retract(M.M, p, X, m)
end

function vector_transport_to!(M::LinearManifold, Y, p, X, q, method::AbstractVectorTransportMethod)
#     println("ISF VECTOR TRANSPORT 1")
    return vector_transport_to!(M.M, Y, p, X, q, method)
end

function vector_transport_to(M::LinearManifold, p, X, q, method::AbstractVectorTransportMethod)
#     println("ISF VECTOR TRANSPORT 1")
    return vector_transport_to(M.M, p, X, q, method)
end

function project!(M::LinearManifold{ndim, n, transp}, Y, p, X) where {ndim, n, transp}
    return project!(M.M, Y, p, X)
end

function LinearFlatManifold(ndim, n, field::AbstractNumbers=ℝ)
    return LinearManifold{ndim, n, true, field}(Euclidean(ndim, n; field), ExponentialRetraction(), ParallelTransport())
end

function LinearTallManifold(ndim, n, field::AbstractNumbers=ℝ)
    return LinearManifold{ndim, n, false, field}(Euclidean(n, ndim; field), ExponentialRetraction(), ParallelTransport())
end

# this is good when doing a submersion of a manifold ndim <= n
function OrthogonalFlatManifold(ndim, n; field::AbstractNumbers=ℝ)
    return LinearManifold{ndim, n, true, field}(Stiefel(ndim, n), PolarRetraction(), DifferentiatedRetractionVectorTransport{PolarRetraction}(PolarRetraction()))
end

# this is good when doing a immersion of a manifold ndim >= n
function OrthogonalTallManifold(ndim, n; field::AbstractNumbers=ℝ)
    return LinearManifold{ndim, n, false, field}(Stiefel(n, ndim), PolarRetraction(), DifferentiatedRetractionVectorTransport{PolarRetraction}(PolarRetraction()))
end

function transpose(M::LinearManifold{ndim, n, transp, field}) where {ndim, n, transp, field}
    return LinearManifold{n, ndim, !transp, field}(M.M, M.R)
end

function getRetraction(M::LinearManifold{ndim, n, transp}) where {ndim, n, transp}
    return M.R
end

function zero(M::LinearManifold{ndim, n, false}) where {ndim, n}
    return project(M.M, zeros(n, ndim))
end

function zero(M::LinearManifold{ndim, n, true}) where {ndim, n}
    return project(M.M, zeros(ndim, n))
end

function zero_vector!(M::LinearManifold, X, p)
    return zero_vector!(M.M, X, p)
end

function manifold_dimension(M::LinearManifold)
    return manifold_dimension(M.M)
end

function inner(M::LinearManifold, p, X, Y)
    return inner(M.M, p, X, Y)
end

function zero_tangent(M::LinearManifold{ndim, n, transp}) where {ndim, n, transp}
    if transp
        return zeros(ndim, n)
    else
        return zeros(n, ndim)
    end
end

function randn(M::LinearManifold{ndim, n, false}) where {ndim, n}
    return project(M.M, randn(n, ndim))
end

function randn(M::LinearManifold{ndim, n, true}) where {ndim, n}
    return project(M.M, randn(ndim, n))
end

function getel(M::LinearManifold{ndim, n, true}, X, idx) where {ndim, n}
    return X[idx[1],idx[2]]
end

function getel(M::LinearManifold{ndim, n, false}, X, idx) where {ndim, n}
    return X[idx[2],idx[1]]
end

function makeCache(M::LinearManifold, X, data; L0 = nothing)
    return nothing
end

function updateCache!(DV::Nothing, M::LinearManifold, X, data; L0 = nothing)
    return nothing
end

function Eval(M::LinearManifold{ndim, n, false}, X, data; L0 = nothing, DV = nothing) where {ndim, n}
    if L0 == nothing
        return X * data
    else
#         @show size(X), size(L0), size(data[1])
        return dropdims(sum((transpose(X) * L0) .* data, dims=1),dims=1)
    end
    return nothing
end

function Eval(M::LinearManifold{ndim, n, true}, X, data; L0 = nothing, DV = nothing) where {ndim, n}
    if L0 == nothing
#         @show size(X), size(data), size(data[1])
#         @show size(transpose(X) * data)
        return transpose(X) * data
    else
#         @show size(X), size(L0), size(data[1])
        return dropdims(sum((X * L0) .* data, dims=1),dims=1)
    end
    return nothing
end

@inline function L0_DF(M::LinearManifold{ndim, n, false}, X, data; L0, ii = 0, DV = nothing) where {ndim, n}
    @assert (size(L0,1) == size(X,1)) && (size(data,1) == size(X,2)) "********* BAD SIZE *********"
    return L0 * transpose(data)
end

@inline function L0_DF(M::LinearManifold{ndim, n, true}, X, data; L0, ii = 0, DV = nothing) where {ndim, n}
    @assert (size(L0,1) == size(X,2)) && (size(data,1) == size(X,1)) "********* BAD SIZE *********"
    return data * transpose(L0)
end

# function DF_dt(M::LinearManifold, X, DV, data; dt, ii)
#     return Eval(M, dt, data)
# end

# the other derivatives
function DF_dt(M::LinearManifold{ndim, n, false}, X, data; dt, ii = 0, DV = nothing) where {ndim, n}
    @assert (n == size(dt,1)) && (size(data,1) == size(dt,2)) "********* BAD SIZE *********"
    return dt * data
end

function DF_dt(M::LinearManifold{ndim, n, true}, X, data; dt, ii = 0, DV = nothing) where {ndim, n}
    @assert (n == size(dt,2)) && (size(data,1) == size(dt,1)) "********* BAD SIZE *********"
    return transpose(dt) * data
end

function Jacobian(M::LinearManifold{ndim, n, false}, X, data) where {ndim, n}
    res = zeros(size(X,1), size(data,1), size(data,2))
    for k=1:size(data,2)
        res[:,:,k] .= X
    end
    return res
end

function Jacobian(M::LinearManifold{ndim, n, true}, X, data) where {ndim, n}
    res = zeros(size(X,2), size(data,1), size(data,2))
    for k=1:size(data,2)
        res[:,:,k] .= transpose(X)
    end
    return res
end

function Hessian(M::LinearManifold{ndim, n, transp}, X, data) where {ndim, n, transp}
    return zeros(ndim, ndim)
end

# DF is the identity
# only works if n >= ndim, which is true
function DFoxT_DFox(M::LinearManifold{ndim, n, transp}, DV, data, ii; scale = alwaysone()) where {ndim, n, transp}
    if transp
        XO = zeros(ndim, n, ndim, n)
        for q=1:n, l=1:size(data,2), p1=1:ndim, p2=1:ndim
            XO[p1,q,p2,q] += data[p1,l] * data[p2,l] / scale[l]
        end
    else
        XO = zeros(n, ndim, n, ndim)
        for q=1:n, l=1:size(data,2), p1=1:ndim, p2=1:ndim
            XO[q,p1,q,p2] += data[p1,l] * data[p2,l] / scale[l]
        end
    end
    return XO
end

# Hessian with respect to parameters in U
# needs JPoUox, JQoUoy, DUox, DUoy
# then  L0J2PoUox, L0J2QoUoy
# results in 
#       DUox^T x JPoUox^T x JPoUox x DUox   -> 1
#     + DUoy^T x JQoUoy^T x JQoUoy x DUoy   -> 2
#     - DUox^T x JPoUox^T x JQoUoy x DUoy   -> 3
#     - DUoy^T x JQoUoy^T x JPoUox x DUox   -> 3^T
#     - DUox^T x L0J2PoUox x DUox
#     + DUoy^T x L0J2QoUoy x DUoy
# The Jacobians JX, JY are evaluated on the whole polynomial!
#        DFT_JFT_JF_DF(MU,                                 N.A., N.A., JPoUox, JQoUoy, L0J2PoUox, L0J2QoUoy, dataIN, dataOUT, N.A.)
function DFT_JFT_JF_DF(M::LinearManifold{ndim, n, transp}, DVX,  DVY,  JX,     JY,     L0J2X,     L0J2Y,     dataX,  dataY,   ii; scale = alwaysone()) where {ndim, n, transp}
    coreX = zeros(n, n, size(dataX,2))
    coreY = zeros(n, n, size(dataY,2))
    coreXY = zeros(n, n, size(dataX,2))

    for q1 = 1:n, q2 = 1:n
        for s = 1:size(JX,1)
            @views coreX[q1,q2,:] .+= JX[s,q1,:] .* JX[s,q2,:] ./ scale[1:size(coreX,3)]
            @views coreY[q1,q2,:] .+= JY[s,q1,:] .* JY[s,q2,:] ./ scale[1:size(coreX,3)]
            @views coreXY[q1,q2,:] .+= JX[s,q1,:] .* JY[s,q2,:] ./ scale[1:size(coreX,3)]
        end
        @views coreX[q1,q2,:] .-= L0J2X[q1,q2,:]
        @views coreY[q1,q2,:] .+= L0J2Y[q1,q2,:]
    end
    if transp
        XO = zeros(ndim, n, ndim, n)
        for l=1:size(coreX,3), q2 = 1:n, p2 = 1:ndim, q1 = 1:n, p1 = 1:ndim
            @inbounds XO[p1,q1,p2,q2] += (
                    dataX[p1,l] * coreX[q1,q2,l] * dataX[p2,l]
                    + dataY[p1,l] * coreY[q1,q2,l] * dataY[p2,l]
                    - dataX[p1,l] * coreXY[q1,q2,l] * dataY[p2,l]
                    - dataX[p2,l] * coreXY[q2,q1,l] * dataY[p1,l] )
        end
    else
        XO = zeros(n, ndim, n, ndim)
        for l=1:size(coreX,3), q2 = 1:n, p2 = 1:ndim, q1 = 1:n, p1 = 1:ndim
            @inbounds XO[q1,p1,q2,p2] += (
                    dataX[p1,l] * coreX[q1,q2,l] * dataX[p2,l]
                    + dataY[p1,l] * coreY[q1,q2,l] * dataY[p2,l]
                    - dataX[p1,l] * coreXY[q1,q2,l] * dataY[p2,l]
                    - dataX[p2,l] * coreXY[q2,q1,l] * dataY[p1,l] )
        end
    end
    return XO
end

struct ConstantManifold{ndim, 𝔽} <: AbstractManifold{𝔽}
    M        :: AbstractManifold{𝔽} # Euclidean{Tuple{n, ndim}, 𝔽}
    R        :: AbstractRetractionMethod
    VT       :: AbstractVectorTransportMethod
end

function ConstantManifold(ndim, field::AbstractNumbers=ℝ)
    return ConstantManifold{ndim, field}(Euclidean(ndim, 1; field), ExponentialRetraction(), ParallelTransport())
end

function HessProjection(M::ConstantManifold)
    HessProjection(M.M)
end

function HessFullProjection(M::ConstantManifold, X, grad, hess)
    return HessFullProjection(M.M, X, grad, hess)
end

function retract!(M::ConstantManifold, q, p, X, method::AbstractRetractionMethod)
    return retract!(M.M, q, p, X, M.R)
end

function retract!(M::ConstantManifold, q, p, X, method::ExponentialRetraction)
    return retract!(M.M, q, p, X, M.R)
end

function retract(
    M::ConstantManifold,
    p,
    X,
    m::AbstractRetractionMethod = default_retraction_method(M),
)
#     q = allocate_result(M, retract, p, X)
    return retract(M.M, p, X, m)
end

function vector_transport_to!(M::ConstantManifold, Y, p, X, q, method::AbstractVectorTransportMethod)
    return vector_transport_to!(M.M, Y, p, X, q, method)
end

function vector_transport_to(M::ConstantManifold, p, X, q, method::AbstractVectorTransportMethod)
    return vector_transport_to(M.M, p, X, q, method)
end

function project!(M::ConstantManifold, Y, p, X)
    return project!(M.M, Y, p, X)
end

function zero(M::ConstantManifold{ndim, field}) where {ndim, field}
    return project(M.M, zeros(ndim,1))
end

function randn(M::ConstantManifold{ndim, field}) where {ndim, field}
    return project(M.M, randn(ndim,1))
end

function zero_vector!(M::ConstantManifold, X, p)
    return zero_vector!(M.M, X, p)
end

function zero_tangent(M::ConstantManifold{ndim, field}) where {ndim, field}
    return zeros(ndim,1)
end

function manifold_dimension(M::ConstantManifold)
    return manifold_dimension(M.M)
end

function inner(M::ConstantManifold, p, X, Y)
    return inner(M.M, p, X, Y)
end

function getel(M::ConstantManifold{ndim, field}, X, idx) where {ndim, field}
    return X[idx[1],1]
end

function makeCache(M::ConstantManifold, X, data; L0 = nothing)
    return nothing
end

function updateCache!(DV::Nothing, M::ConstantManifold, X, data; L0 = nothing)
    return nothing
end

function Eval(M::ConstantManifold, X, data; L0 = nothing, DV = nothing)
    if L0 == nothing
        res = zeros(size(X,1), size(data,2))
#         @show size(res)
        for k=1:size(data,2)
            res[:,k] .= X
        end
        return res
    else
        return transpose(X) * L0
    end
    return nothing
end

@inline function L0_DF(M::ConstantManifold{ndim, field}, X, data; L0, ii = 0, DV = nothing) where {ndim, field}
#     @show "Const_L0_DF"
    return sum(L0, dims=2)
end

# Hessian with respect to parameters in U
# needs JPoUox, JQoUoy, DUox, DUoy
# then  L0J2PoUox, L0J2QoUoy
# results in 
#       DUox^T x JPoUox^T x JPoUox x DUox   -> 1
#     + DUoy^T x JQoUoy^T x JQoUoy x DUoy   -> 2
#     - DUox^T x JPoUox^T x JQoUoy x DUoy   -> 3
#     - DUoy^T x JQoUoy^T x JPoUox x DUox   -> 3^T
#     - DUox^T x L0J2PoUox x DUox
#     + DUoy^T x L0J2QoUoy x DUoy
# The Jacobians JX, JY are evaluated on the whole polynomial!
#        DFT_JFT_JF_DF(MU,                            N.A., N.A., JPoUox, JQoUoy, L0J2PoUox, L0J2QoUoy, dataIN, dataOUT, N.A.)
function DFT_JFT_JF_DF(M::ConstantManifold{n, field}, DVX,  DVY,  JX,     JY,     L0J2X,     L0J2Y,     dataX,  dataY,   ii; scale = alwaysone()) where {n, field}
    H2 = zeros(size(JX,2),size(JX,2))
    for p1=1:size(JX,2), p2=1:size(JX,2), k=1:size(JX,3)
        for j=1:size(JX,1)
            @inbounds H2[p1,p2] += (JX[j,p1,k] * JX[j,p2,k] + JY[j,p1,k] * JY[j,p2,k] - JX[j,p1,k] * JY[j,p2,k] - JY[j,p1,k] * JX[j,p2,k])/scale[k] 
        end
        @inbounds H2[p1,p2] += - L0J2X[p1,p2,k] + L0J2Y[p1,p2,k]
    end
    return reshape(H2, size(H2,1), 1, size(H2,2), 1)
end

function DF_dt(M::ConstantManifold{n, field}, X, DV, data, dt, ii) where {n, field}
    res = zeros(size(X,1), size(data,2))
    for k=1:size(data,2)
        res[:,k] .= X
    end
    return res
end

function HessProjection(M::ConstantManifold, X, grad, HessV, V)
    HessProjection(M.M, X, grad, HessV, V)
end

function tensorVecsInvalidate(DV, M::Union{ConstantManifold,LinearManifold}, ii)
    return nothing
end

function tensorBVecsInvalidate(DV, M::Union{ConstantManifold,LinearManifold}, ii)
    return nothing
end
