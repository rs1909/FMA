
# it consists of
# p1 \in (n,m); p2 \in (n,l}, where l is calculated to represent nonlinear mononomials up to (and including) order 'order'
# mexp \in (m,l) the exponents of the polynomials
struct OAEManifold{ndim, mdim, order, orthogonal, 𝔽} <: AbstractManifold{𝔽}
    mlist
    M        :: ProductManifold 
    R        :: ProductRetraction 
    VT       :: ProductVectorTransport
end

# this is a flat and tall orthogonal manifold at the same time
# This means that different routines will apply to it at given times
function OAEManifold(ndim, mdim, order, orthogonal = true, field::AbstractNumbers=ℝ)
    if orthogonal
        # only include the non linear part: p1 will act as the nonlinear part and p1^T is the projection
        mlist = (OrthogonalFlatManifold(ndim, mdim; field = field), DensePolyManifold(mdim, ndim, order; min_order = 2, field = field))
    else
        # include the linear part: p2 is the full immersion, p1^T is the projection
        mlist = (OrthogonalFlatManifold(ndim, mdim; field = field), DensePolyManifold(mdim, ndim, order; min_order = 1, field = field))
    end
    M = ProductManifold(mlist...)
    R = ProductRetraction(map(x->getfield(x,:R), mlist)...)
    VT = ProductVectorTransport(map(x->getfield(x,:VT), mlist)...)
    return OAEManifold{ndim, mdim, order, orthogonal, field}(mlist, M, R, VT)
end

function OAE_U(M::OAEManifold)
    return M.mlist[1]
end

function OAE_W(M::OAEManifold)
    return M.mlist[2]
end

function OAE_Upoint(X)
    return X.parts[1]
end

function OAE_Wpoint(X)
    return X.parts[2]
end

function zero_vector!(M::OAEManifold, X, p)
    X .= ProductRepr(map(zero, M.mlist))
    return X
end

function manifold_dimension(M::OAEManifold)
    return manifold_dimension(M.M)
end

function inner(M::OAEManifold, p, X, Y)
    return inner(M.M, p, X, Y)
end

function project!(M::OAEManifold{ndim, mdim, order, orthogonal, field}, Y, p, X) where {ndim, mdim, order, orthogonal, field}
    p1 = p.parts[1]
    p2 = p.parts[2]
    X1 = X.parts[1]
    X2 = X.parts[2]
    F = qr(p1)
    p1perp = F.Q[:,mdim+1:end]
    K1 = (X1' * p1 + p1' * X1)/2
    K2 = (X1' * p2 + p1' * X2) * inv(I + p2' * p1perp * p1perp' * p2)
    Y1 = X1 - p1*K1 - p1perp * p1perp' * p2 * K2'
    Y2 = X2 - p1*K2
    # testing
#     @show norm(Y1' * p1 + p1' * Y1)
#     @show norm(Y1' * p2 + p1' * Y2)
    Y.parts[1] .= Y1
    Y.parts[2] .= Y2
    return Y
end

# the retraction should normally be the converged iteration
# However we only take the first step, as it is more accurate than the QR retraction
# function retr(q1,q2)
#     G = svd(q1)
#     p1 = G.U * G.Vt
#     beta = p1' * q2
#     sigma = p1'* q1
#     p2 = q2 - p1 * beta
#     println("2->", norm(p1*sigma + p2*beta' - q1))
#     println("3->", norm(p2 + p1*beta - q2))
#     println(0," SVD->", norm([q1;;q2] - [p1;;p2]))
#     for k=1:10
#         A = p1' * (q1 - q2 * q2' * p1);
#         F = svd(reshape((kron(A',Diagonal(I,size(q2,1))) + kron(Diagonal(I,size(q1,2)),q2 * q2'))\vec(q1),size(q1)));
#         p1 = F.U*F.Vt
#         println("1->", norm(p1*A + q2 * q2' * p1 - q1))
#         beta = p1' * q2
#         sigma = p1'* q1
#         p2 .= q2 - p1 * beta
#         println("2->", norm(p1*sigma + p2*beta' - q1))
#         println("3->", norm(p2 + p1*beta - q2))
#         println(k," SVD->", norm([q1;;q2] - [p1;;p2]))
#     end
#     return p1,p2
# end
# result is in 'p', input is 'q'
# This is a QR based retraction as nothing else works consistently
# function retract!(M::OAEManifold{ndim, mdim, order, true, field}, p, q, X, alpha::Number, method::AbstractRetractionMethod = QRRetraction()) where {ndim, mdim, order, field}
#     q1 = q.parts[1] + alpha * X.parts[1]
#     q2 = q.parts[2] + alpha * X.parts[2]
#     # zero-th iteration
#     G = svd(q1)
#     p1 = G.U * G.Vt
#     # first iteration
#     for k=1:5
#         A = p1' * (q1 - q2 * q2' * p1);
#         F = svd(reshape((kron(A',Diagonal(I,size(q2,1))) + kron(Diagonal(I,size(q1,2)),q2 * q2'))\vec(q1),size(q1)));
#         p1 = F.U*F.Vt
#     end
#     
#     beta = p1' * q2
#     p2 = q2 - p1 * beta
# 
#     p.parts[1] .= p1
#     p.parts[2] .= p2
#     return p
# end

# THE OLD VERSION
# function retract!(M::OAEManifold{ndim, mdim, order, false, field}, p, q, X, alpha::Number, method::AbstractRetractionMethod = QRRetraction()) where {ndim, mdim, order, field}
#     q1 = q.parts[1] + alpha * X.parts[1]
#     q2 = q.parts[2] + alpha * X.parts[2]
#     # linear indices
#     linid = findall(isequal(1), dropdims(sum(M.mlist[2].mexp,dims=1),dims=1))
#     MM = zeros(size(M.mlist[2].mexp))
#     MM[:,linid] .=  M.mlist[2].mexp[:,linid]
# #     set-up
#     AA = q2 * q2'
#     BB = q1 + q2 * MM'
#     # zero-th iteration
#     G = svd(q1)
#     p1 = G.U * G.Vt
#     # first iteration
#     for k=1:15
#         A = p1' * (q1 + q2*MM' - q2 * q2' * p1);
#         p1t = reshape((kron(A',Diagonal(I,size(q2,1))) + kron(Diagonal(I,size(q1,2)),q2 * q2'))\vec(q1 + q2*MM'),size(q1))
#     #     @show norm(p1t*A + q2 * q2' * p1t - (q1 + q2*MM'))
#         F = svd(p1t)
#         p1 = F.U*F.Vt
#         a = p1' * (BB - AA * p1)
#         @show norm(p1 * a + AA * p1 - BB)
#     end
#     beta = p1' * q2 - MM
#     p2 = q2 - p1 * beta
# 
#     p.parts[1] .= p1
#     p.parts[2] .= p2
#     return p
# end

# function retres(xx, AA, BB, MM, ndim, mdim)
#     p1 = reshape(xx[1:ndim * mdim], ndim, mdim)
#     a = reshape(xx[ndim * mdim + 1 : end], mdim, mdim)
#     return [vec(BB - p1 * a - AA * p1); vec(I - p1' * p1)]
# end
# 
# function retjac(xx, AA, BB, MM, ndim, mdim)
#     xxp = copy(xx)
#     epsilon = 1e-9
#     JAC = zeros(length(xx), length(xx))
#     for k=1:length(xx)
#         xxp[k] += epsilon
#         JAC[:,k] .= -(retres(xxp, AA, BB, MM, ndim, mdim) .- retres(xx, AA, BB, MM, ndim, mdim)) / epsilon
#         xxp[k] = xx[k]
#     end
#     return JAC
# end
# 
function retract!(M::OAEManifold{ndim, mdim, order, orthogonal, field}, p, q, X, alpha::Number, method::AbstractRetractionMethod = QRRetraction()) where {ndim, mdim, order, orthogonal, field}
#     println("Newton orthoProj retraction")
    q1 = q.parts[1] + alpha * X.parts[1]
    q2 = q.parts[2] + alpha * X.parts[2]
    # use Newton iteration for solving the constraint equation
    #   p1 a + q2 q2^T p1 = q1 + q2 M
    #   p1^T p1 = I
    linid = findall(isequal(1), dropdims(sum(M.mlist[2].mexp,dims=1),dims=1))
    MM = zeros(eltype(q1), size(M.mlist[2].mexp)...)
    MM[:,linid] .=  M.mlist[2].mexp[:,linid]
    # Try the equations instead
    #   p1 a + AA p1 = BB
    #   p1^T p1 = I
    AA = q2 * q2'
    BB = q1 + q2 * MM'
    CF = zeros(eltype(q1), ndim * mdim + mdim * mdim, ndim * mdim + mdim * mdim)
    # initialise
    p1 = copy(q1)
    a = q1' * (BB - AA * q1)
    Idn = Diagonal(I,ndim)
    Idm = Diagonal(I,mdim)
    # the loop
    for k = 1:100
        # [i,j,k,l] := I[i,k] * a[l,j] + A[i,k] * I[j,l]
        CF[1:ndim * mdim, 1:ndim * mdim] .= kron(a, Idn) .+ kron(Idm, AA)
        # [i,j,k,l] := p1[i,k] * I[j,l]
        CF[1:ndim * mdim, ndim * mdim + 1 : end] .= kron(Idm, p1)
        @views CFp = reshape(CF[ndim * mdim + 1 : end, 1:ndim * mdim], mdim, mdim, ndim, mdim)
        @tullio CFp[i,j,k,l] = p1[k,j] * Idm[i,l] + p1[k,i] * Idm[j,l]
#         JJ = kron(Idm, transpose(p1))
#         JJ = JJ + transpose(JJ)
#         @show norm(CF[ndim * mdim + 1 : end, 1:ndim * mdim] - JJ)
        F = svd(CF)
        ids = findall(F.S .> 20*eps(eltype(CF)))
        res_t = F.U' * [vec(BB - p1 * a - AA * p1); vec(I - p1' * p1)]
        res = F.V[:,ids] * Diagonal( one(eltype(CF)) ./ F.S[ids]) * res_t[ids]
        idz = findall(F.S .<= 20*eps(eltype(CF)))
#         @show norm(res)
        p1 .+= reshape(res[1:ndim * mdim], size(p1)...)
        a .+= reshape(res[ndim * mdim + 1 : end], size(a)...)
        if norm(res) < 20*eps(eltype(CF))
#             print(":$(k):")
            break
        end
    end
    # calculating the rest of the solution
    beta = p1' * q2 - MM
    p2 = q2 - p1 * beta

    p.parts[1] .= p1
    p.parts[2] .= p2
    return p
end

function retract(M::OAEManifold{ndim, mdim, order, orthogonal, field}, q, X, alpha::Number, method::AbstractRetractionMethod) where {ndim, mdim, order, orthogonal, field}
    p = zero(q)
    return retract!(M, p, q, X, alpha, method)
end

function vector_transport_to!(M::OAEManifold, Y, p, X, q, VT::AbstractVectorTransportMethod)
    project!(M, Y, q, X)
    return Y
end

function zero(M::OAEManifold{ndim, mdim, order, orthogonal, field}) where {ndim, mdim, order, orthogonal, field}
    out = ProductRepr(map(zero, M.mlist))
    return out
end

function randn(M::OAEManifold{ndim, mdim, order, orthogonal, field}) where {ndim, mdim, order, orthogonal, field}
    X = ProductRepr(map(randn, M.mlist))
    q1 = X.parts[1]
    q2 = X.parts[2]
    linid = findall(isequal(1), dropdims(sum(M.mlist[2].mexp,dims=1),dims=1))
    MM = zeros(eltype(q1), size(M.mlist[2].mexp)...)
    MM[:,linid] .=  M.mlist[2].mexp[:,linid]

    beta = q1' * q2 - MM
    p2 = q2 - q1 * beta

    X.parts[2] .= p2    
    return X
end

# Now the full autoencoder manifold with conjugate dynamics

struct AENCManifold{ndim, mdim, Worder, Sorder, orthogonal, 𝔽} <: AbstractManifold{𝔽}
    mlist
    M        :: ProductManifold 
    R        :: ProductRetraction 
    VT       :: ProductVectorTransport
end

@doc raw"""
    M = AENCManifold(ndim, mdim, Sorder, Worder, orthogonal = true, field::AbstractNumbers=ℝ)
    
Creates an autoencoder as a matrix manifold.

The parameters are
  * `ndim` the dimansionality of the problem ``\boldsymbol{F}``
  * `mdim` dimensionality of the low-dimensional map ``\boldsymbol{S}``
  * `Sorder` polynomial order of map ``\boldsymbol{S}``
  * `Worder` polynomial order of decoder ``\boldsymbol{W}``
  * `orthogonal` whether ``D\boldsymbol{W}(0)`` is orthogonal to ``\boldsymbol{U}``. When the data is on a manifold `true` is the good answer.
"""
function AENCManifold(ndim, mdim, Sorder, Worder, orthogonal = true, field::AbstractNumbers=ℝ)
    mlist = (DensePolyManifold(mdim, mdim, Sorder; min_order = 1), OAEManifold(ndim, mdim, Worder, orthogonal, field))
    M = ProductManifold(mlist...)
    R = ProductRetraction(map(x->getfield(x,:R), mlist)...)
    VT = ProductVectorTransport(map(x->getfield(x,:VT), mlist)...)
    return AENCManifold{ndim, mdim, Worder, Sorder, orthogonal, field}(mlist, M, R, VT)
end

@doc raw"""
    X = zero(M::AENCManifold)
    
Creates a zero valued representation of an autoencoder.
"""
function zero(M::AENCManifold)
    return ProductRepr(map(zero, M.mlist))
end

function randn(M::AENCManifold)
    return ProductRepr(map(randn, M.mlist))
end

function zero_vector!(M::AENCManifold, X::ProductRepr, p::ProductRepr)
    zero_vector!(M.M, X, p)
    return X
end

function manifold_dimension(M::AENCManifold)
    return manifold_dimension(M.M)
end

function inner(M::AENCManifold, p, X, Y)
    return inner(M.M, p, X, Y)
end

function project!(M::AENCManifold, Y, p, X)
    project!(M.M, Y, p, X)
    return X
end

function retract!(M::AENCManifold, q, p, X, t::Number, method::AbstractRetractionMethod)
    return retract!(M.M, q, p, X, t, method)
end

function retract(M::AENCManifold, p, X, t::Number, m::AbstractRetractionMethod = default_retraction_method(M))
    q = allocate_result(M, retract, p, X)
    return retract!(M, q, p, X, t, m)
end

function vector_transport_to!(M::AENCManifold, Y, p, X, q, method::AbstractVectorTransportMethod)
    return vector_transport_to!(M.M, Y, p, X, q, method)
end

function vector_transport_to(M::AENCManifold, p, X, q, m::AbstractVectorTransportMethod)
    Y = allocate_result(M, vector_transport_to, X, p)
    return vector_transport_to!(M, Y, p, X, q, m)
end

function AENC_S(M::AENCManifold)
    return M.mlist[1]
end

function AENC_Spoint(X)
    return X.parts[1]
end

# function AENC_OAE(M::AENCManifold)
#     return M.mlist[2]
# end
# 
# function AENC_OAEpoint(X)
#     return X.parts[2]
# end

function AENC_U(M::AENCManifold{ndim, mdim, Worder, Sorder, orthogonal, field}) where {ndim, mdim, Worder, Sorder, orthogonal, field}
    return M.mlist[2].mlist[1]
end

function AENC_Upoint(X)
    return X.parts[2].parts[1]
end

function AENC_Wl(M::AENCManifold{ndim, mdim, Worder, Sorder, orthogonal, field}) where {ndim, mdim, Worder, Sorder, orthogonal, field}
    return OrthogonalTallManifold(ndim, mdim; field = field)
end

function AENC_Wlpoint(X)
    return X.parts[2].parts[1]
end

function AENC_Wnl(M::AENCManifold{ndim, mdim, Worder, Sorder, orthogonal, field}) where {ndim, mdim, Worder, Sorder, orthogonal, field}
    return M.mlist[2].mlist[2]
end

function AENC_Wnlpoint(X)
    return X.parts[2].parts[2]
end

function AENCLoss(M::AENCManifold{ndim, mdim, Worder, Sorder, true, field}, X, dataIN, dataOUT) where {ndim, mdim, Worder, Sorder, field}
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN, zeros(size(dataIN,1)))
    #  Wl * S( U * x ) + Wnl( S( U * x ) ) - y according to the commutative diagram
    Ux = Eval(AENC_U(M), AENC_Upoint(X), dataIN)
    SUx = Eval(AENC_S(M), AENC_Spoint(X), Ux)
    WSUx = Eval(AENC_Wl(M), AENC_Wlpoint(X), SUx) + Eval(AENC_Wnl(M), AENC_Wnlpoint(X), SUx)
    L0 = WSUx .- dataOUT
    return sum( (L0 .^ 2) ./ scale )/2/datalen
end

function AENCGradient(M::AENCManifold{ndim, mdim, Worder, Sorder, true, field}, X, dataIN, dataOUT) where {ndim, mdim, Worder, Sorder, field}
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN, zeros(size(dataIN,1)))
    # Wl * S( U * x ) + Wnl( S( U * x ) ) - y according to the commutative diagram
    # and U = Wl^T
    Ux = Eval(AENC_U(M), AENC_Upoint(X), dataIN)
    SUx = Eval(AENC_S(M), AENC_Spoint(X), Ux)
    WSUx = Eval(AENC_Wl(M), AENC_Wlpoint(X), SUx) + Eval(AENC_Wnl(M), AENC_Wnlpoint(X), SUx)
    L0 = (WSUx .- dataOUT) ./ scale / datalen
    # need to apply the chain rule
    # Wl:  L0 _*_ D Wl * S( U * x )
    # Wnl: L0 _*_ D Wnl( S( U * x ) )
    # S:   L0 _*_ [J Wl * S( U * x ) + J Wnl( S( U * x ) )] D S( U * x )
    # U:   L0 _*_ [J Wl * S( U * x ) + J Wnl( S( U * x ) )] J S( U * x ) D U * x
    # [J Wl * S( U * x ) + J Wnl( S( U * x ) )]
    JWSUx = Jacobian(AENC_Wl(M), AENC_Wlpoint(X), SUx) + Jacobian(AENC_Wnl(M), AENC_Wnlpoint(X), SUx)
    # J S( U * x )
    JSUx = Jacobian(AENC_S(M), AENC_Spoint(X), Ux)
    L0_JWSUx_JSUx = @views [dot(L0[:,p]' * JWSUx[:,:,p], JSUx[:,j,p]) for j=1:size(JSUx,2), p=1:size(JSUx,3)]
    L0_JWSUx = @views [dot(L0[:,p], JWSUx[:,j,p]) for j=1:size(JWSUx,2), p=1:size(JWSUx,3)] #dropdims(sum(JWSUx .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    # W
    Wl = L0_DF(AENC_Wl(M), AENC_Wlpoint(X), SUx, L0 = L0)
    Wnl = L0_DF(AENC_Wnl(M), AENC_Wnlpoint(X), SUx, L0 = L0)
    S = L0_DF(AENC_S(M), AENC_Spoint(X), Ux, L0 = L0_JWSUx)
    U = L0_DF(AENC_U(M), AENC_Upoint(X), dataIN, L0 = L0_JWSUx_JSUx)
    return ProductRepr(S, ProductRepr(Wl + U, Wnl))
end

function AENCLoss(M::AENCManifold{ndim, mdim, Worder, Sorder, false, field}, X, dataIN, dataOUT) where {ndim, mdim, Worder, Sorder, field}
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN, zeros(size(dataIN,1)))
    #  Wnl( S( U * x ) ) - y according to the commutative diagram
    Ux = Eval(AENC_U(M), AENC_Upoint(X), dataIN)
    SUx = Eval(AENC_S(M), AENC_Spoint(X), Ux)
    WSUx = Eval(AENC_Wnl(M), AENC_Wnlpoint(X), SUx)
    L0 = WSUx .- dataOUT
    return sum( (L0 .^ 2) ./ scale )/2/datalen
end

function AENCGradient(M::AENCManifold{ndim, mdim, Worder, Sorder, false, field}, X, dataIN, dataOUT) where {ndim, mdim, Worder, Sorder, field}
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN, zeros(size(dataIN,1)))
    # Wnl( S( U * x ) ) - y according to the commutative diagram
    # and U = Wl^T
    Ux = Eval(AENC_U(M), AENC_Upoint(X), dataIN)
    SUx = Eval(AENC_S(M), AENC_Spoint(X), Ux)
    WSUx = Eval(AENC_Wnl(M), AENC_Wnlpoint(X), SUx)
    L0 = (WSUx .- dataOUT) ./ scale / datalen
    # need to apply the chain rule
    # Wl:  L0 _*_ D Wl * S( U * x )
    # Wnl: L0 _*_ D Wnl( S( U * x ) )
    # S:   L0 _*_ J Wnl( S( U * x ) ) D S( U * x )
    # U:   L0 _*_ J Wnl( S( U * x ) ) J S( U * x ) D U * x
    # [J Wl * S( U * x ) + J Wnl( S( U * x ) )]
    JWSUx = Jacobian(AENC_Wnl(M), AENC_Wnlpoint(X), SUx)
    # J S( U * x )
    JSUx = Jacobian(AENC_S(M), AENC_Spoint(X), Ux)
    L0_JWSUx_JSUx = @views [dot(L0[:,p]' * JWSUx[:,:,p], JSUx[:,j,p]) for j=1:size(JSUx,2), p=1:size(JSUx,3)]
    L0_JWSUx = @views [dot(L0[:,p], JWSUx[:,j,p]) for j=1:size(JWSUx,2), p=1:size(JWSUx,3)] #dropdims(sum(JWSUx .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    # W
    Wnl = L0_DF(AENC_Wnl(M), AENC_Wnlpoint(X), SUx, L0 = L0)
    S = L0_DF(AENC_S(M), AENC_Spoint(X), Ux, L0 = L0_JWSUx)
    U = L0_DF(AENC_U(M), AENC_Upoint(X), dataIN, L0 = L0_JWSUx_JSUx)
    return ProductRepr(S, ProductRepr(U, Wnl))
end

# works for both as W is not featured!
function AENCROMLoss(M::AENCManifold{ndim, mdim, Worder, Sorder, orthogonal, field}, X, dataIN, dataOUT) where {ndim, mdim, Worder, Sorder, orthogonal, field}
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN, zeros(size(dataIN,1)))
    #  S( U * x ) - U * y
    Ux = Eval(AENC_U(M), AENC_Upoint(X), dataIN)
    Uy = Eval(AENC_U(M), AENC_Upoint(X), dataOUT)
    SUx = Eval(AENC_S(M), AENC_Spoint(X), Ux)
    L0 = SUx .- Uy
    return sum( (L0 .^ 2) ./ scale )/2/datalen
end

# works for both as W is not featured!
function AENCROMGradient(M::AENCManifold{ndim, mdim, Worder, Sorder, orthogonal, field}, X, dataIN, dataOUT) where {ndim, mdim, Worder, Sorder, orthogonal, field}
    datalen = size(dataIN,2)
    scale = AmplitudeScaling(dataIN, zeros(size(dataIN,1)))
    #  S( U * x ) - U * y
    Ux = Eval(AENC_U(M), AENC_Upoint(X), dataIN)
    Uy = Eval(AENC_U(M), AENC_Upoint(X), dataOUT)
    SUx = Eval(AENC_S(M), AENC_Spoint(X), Ux)
    L0 = (SUx .- Uy) ./ scale / datalen
    # need to apply the chain rule
    S = L0_DF(AENC_S(M), AENC_Spoint(X), Ux, L0 = L0)
    return ProductRepr(S, ProductRepr(zero(AENC_Upoint(X)), zero(AENC_Wnlpoint(X))))
end

# ORTHOGONAL!
function AENCFitLoss(M::AENCManifold{ndim, mdim, Worder, Sorder, true, field}, X, data) where {ndim, mdim, Worder, Sorder, field}
    datalen = size(data,2)
    scale = AmplitudeScaling(data, zeros(size(data,1)))
    #  Wl * U * x + Wnl( U * x ) - y according to the commutative diagram
    Ux = Eval(AENC_U(M), AENC_Upoint(X), data)
    WUx = Eval(AENC_Wl(M), AENC_Wlpoint(X), Ux) + Eval(AENC_Wnl(M), AENC_Wnlpoint(X), Ux)
    L0 = WUx .- data
    return sum( (L0 .^ 2) ./ scale )/2/datalen
end

# ORTHOGONAL!
function AENCFitGradient(M::AENCManifold{ndim, mdim, Worder, Sorder, true, field}, X, data) where {ndim, mdim, Worder, Sorder, field}
    datalen = size(data,2)
    scale = AmplitudeScaling(data, zeros(size(data,1)))
    #  Wl * U * x + Wnl( U * x ) - y according to the commutative diagram
    Ux = Eval(AENC_U(M), AENC_Upoint(X), data)
    WUx = Eval(AENC_Wl(M), AENC_Wlpoint(X), Ux) + Eval(AENC_Wnl(M), AENC_Wnlpoint(X), Ux)
    L0 = (WUx .- data) ./ scale / datalen
    # need to apply the chain rule
    # Wl:  L0 _*_ D Wl * S( U * x )
    # Wnl: L0 _*_ D Wnl( S( U * x ) )
    # S:   L0 _*_ [J Wl * S( U * x ) + J Wnl( S( U * x ) )] D S( U * x )
    # U:   L0 _*_ [J Wl * S( U * x ) + J Wnl( S( U * x ) )] J S( U * x ) D U * x
    # [J Wl * S( U * x ) + J Wnl( S( U * x ) )]
    JWUx = Jacobian(AENC_Wl(M), AENC_Wlpoint(X), Ux) + Jacobian(AENC_Wnl(M), AENC_Wnlpoint(X), Ux)
    # J S( U * x )
    L0_JWUx = @views [dot(L0[:,p], JWUx[:,j,p]) for j=1:size(JWUx,2), p=1:size(JWUx,3)] #dropdims(sum(JWSUx .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    # W
    Wl = L0_DF(AENC_Wl(M), AENC_Wlpoint(X), Ux, L0 = L0)
    Wnl = L0_DF(AENC_Wnl(M), AENC_Wnlpoint(X), Ux, L0 = L0)
    U = L0_DF(AENC_U(M), AENC_Upoint(X), data, L0 = L0_JWUx)
    return ProductRepr(zero(AENC_Spoint(X)), ProductRepr(Wl + U, Wnl))

end

function AENCFitLoss(M::AENCManifold{ndim, mdim, Worder, Sorder, false, field}, X, data) where {ndim, mdim, Worder, Sorder, field}
    datalen = size(data,2)
    scale = AmplitudeScaling(data, zeros(size(data,1)))
    #  Wnl(U * x) - x
    Ux = Eval(AENC_U(M), AENC_Upoint(X), data)
    WUx = Eval(AENC_Wnl(M), AENC_Wnlpoint(X), Ux)
    L0 = WUx .- data
    return sum( (L0 .^ 2) ./ scale )/2/datalen
end

function AENCFitGradient(M::AENCManifold{ndim, mdim, Worder, Sorder, false, field}, X, data) where {ndim, mdim, Worder, Sorder, field}
    datalen = size(data,2)
    scale = AmplitudeScaling(data, zeros(size(data,1)))
    #  Wnl(U * x) - x
    Ux = Eval(AENC_U(M), AENC_Upoint(X), data)
    WUx = Eval(AENC_Wnl(M), AENC_Wnlpoint(X), Ux)
    L0 = (WUx .- data) ./ scale / datalen
    # need to apply the chain rule
    # Wnl: L0 _*_ D Wnl( U )
    # U:   L0 _*_ J Wnl( U * x ) D U * x
    JWUx = Jacobian(AENC_Wnl(M), AENC_Wnlpoint(X), Ux)
    # J S( U * x )
    L0_JWUx = @views [dot(L0[:,p], JWUx[:,j,p]) for j=1:size(JWUx,2), p=1:size(JWUx,3)] #dropdims(sum(JWSUx .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    # W
    Wnl = L0_DF(AENC_Wnl(M), AENC_Wnlpoint(X), Ux, L0 = L0)
    U = L0_DF(AENC_U(M), AENC_Upoint(X), data, L0 = L0_JWUx)
    return ProductRepr(zero(AENC_Spoint(X)), ProductRepr(U, Wnl))
end

function AENCRiemannianGradient(M::AENCManifold{ndim, mdim, Worder, Sorder, orthogonal, field}, X, dataIN, dataOUT) where {ndim, mdim, Worder, Sorder, orthogonal, field}
    return project(M, X, AENCGradient(M, X, dataIN, dataOUT))
end

function AENCFitRiemannianGradient(M::AENCManifold{ndim, mdim, Worder, Sorder, orthogonal, field}, X, data) where {ndim, mdim, Worder, Sorder, orthogonal, field}
    return project(M, X, AENCFitGradient(M, X, data))
end

function AENCROMRiemannianGradient(M::AENCManifold{ndim, mdim, Worder, Sorder, orthogonal, field}, X, dataIN, dataOUT) where {ndim, mdim, Worder, Sorder, orthogonal, field}
    return project(M, X, AENCROMGradient(M, X, dataIN, dataOUT))
end

mutable struct AENCDebug <: DebugAction
    print::Function
    t0
    Tstep
    AENCDebug(Tstep, print::Function=print) = new(print, time(), Tstep)
end

function (d::AENCDebug)(mp,trs,i::Int)
    d.print(@sprintf("time = %.1f[s] ", time()-d.t0), 
            @sprintf("F(x) = %.4e ", get_cost(mp, trs.p)), 
            @sprintf("G(x) = %.4e ", norm(get_manifold(mp), trs.p, trs.X)),
            "| S:", 
            unique(sort(abs.(eigvals(getLinearPart(AENC_S(get_manifold(mp)),AENC_Spoint(trs.p)))))),
            unique(sort(abs.(angle.(eigvals(getLinearPart(AENC_S(get_manifold(mp)),AENC_Spoint(trs.p))))/d.Tstep))))
end


@doc raw"""
    AENCIndentify(dataIN, dataOUT, Tstep, embedscales, freq, orders = (S=7,W=7))
    
Input:

  * `dataIN` is a two dimensional array, each column is an ``\boldsymbol{x}_k`` value, 
  * `dataOUT` is a two dimensional array, each column is an ``\boldsymbol{y}_k``
  * `Tstep` is the time step between ``\boldsymbol{x}_k`` and ``\boldsymbol{y}_k``
  * `embedscales`, denoted by ``\boldsymbol{w}`` is a matrix applied to ``\boldsymbol{x}_k``, ``\boldsymbol{y}_k``, used to calculate the amplitude of a signal
  * `freq` is the frequency that the invariant foliation is calculated for.
  * `orders` is a named tuple, specifies the polynomial order of ``\boldsymbol{S}`` and ``\boldsymbol{W}``.

Output is a tuple with elements
  1. vector of instantaneous frequencies
  2. vector of instantaneous damping
  3. vector of instantaneous amplitudes
  4. the scaling factor that was used to fit all data into the unit ball
  5. uncorrected vector of instantaneous frequencies
  6. uncorrected vector of instantaneous damping
  7. uncorrected vector of instantaneous amplitudes
"""
function AENCIndentify(dataINorig, dataOUTorig, Tstep, embedscales, freq, orders = (S=7,W=7); iteration = (aenc=100, map=100))

    NDIM = size(dataINorig,1)
    din = NDIM
    dout = 2
    
    dataINlin, dataOUTlin, linscale = dataPrune(dataINorig, dataOUTorig; perbox=14000, retbox=6, nbox=6)
    X, S1, U1, W1, S2, U2, W2 = LinearFit(dataINlin, dataOUTlin, Tstep, freq)
    
    # if orthogonal
    M = AENCManifold(NDIM, dout, orders.S, orders.W)
    X = zero(M)
    AENC_Wlpoint(X) .= W1
    setLinearPart!(AENC_S(M), AENC_Spoint(X), S1)

    # if non-orthogonal
#     M = AENCManifold(NDIM, dout, orders.S, orders.W, false)
#     X = zero(M)
#     AENC_Upoint(X) .= U1tr
#     setLinearPart!(AENC_Wnl(M), AENC_Wnlpoint(X), W1)
#     setLinearPart!(AENC_S(M), AENC_Spoint(X), S1)
    
    scale = maximum(sqrt.(sum(dataINorig .^ 2, dims=1)))
    dataINorig ./= scale
    dataOUTorig ./= scale

    # fiting the autoencoder
    Xres = quasi_Newton(M, 
                        (Mp, x) -> AENCFitLoss(Mp, x, dataOUTorig), 
                        (Mp, x) -> AENCFitRiemannianGradient(Mp, x, dataOUTorig), 
                        X,
                        retraction_method=M.R,
                        vector_transport_method=M.VT,
                        stepsize = WolfePowellBinaryLinesearch(M, retraction_method=M.R, vector_transport_method=M.VT),
#                         stepsize = WolfePowellBinaryLinesearch(M.R,M.VT, 0.0, 0.99),
                        stopping_criterion=StopWhenAny(
                            StopWhenGradientNormLess(1e-8),
                            StopAfterIteration(iteration.aenc)),
                        cautious_update = true,
                        debug = [
                        :Stop,
                        :Iteration,
                        :Cost,
                        " | ",
                        AENCDebug(Tstep),
                        "\n",
                        1,
                        ])
    X .= Xres
    # fitting the dynamics
    Xres = quasi_Newton(M, 
                        (Mp, x) -> AENCROMLoss(Mp, x, dataINorig, dataOUTorig), 
                        (Mp, x) -> AENCROMRiemannianGradient(Mp, x, dataINorig, dataOUTorig), 
                        X,
                        retraction_method=M.R,
                        vector_transport_method=M.VT,
                        stepsize = WolfePowellBinaryLinesearch(M, retraction_method=M.R, vector_transport_method=M.VT),
#                         stepsize = WolfePowellBinaryLinesearch(M.R,M.VT),
                        stopping_criterion=StopWhenAny(
                            StopWhenGradientNormLess(1e-8),
                            StopAfterIteration(iteration.map)),
                        cautious_update = true,
                        debug = [
                        :Stop,
                        :Iteration,
                        :Cost,
                        " | ",
                        AENCDebug(Tstep),
                        "\n",
                        1,
                        ])
    X .= Xres
    # fitting both together
#     Xres = conjugate_gradient_descent(M,
#                         (Mp, x) -> AENCLoss(Mp, x, dataINorig, dataOUTorig), 
#                         (Mp, x) -> AENCRiemannianGradient(Mp, x, dataINorig, dataOUTorig), 
#                         X,
#                         retraction_method=M.R,
#                         vector_transport_method=M.VT,
#                         stepsize = WolfePowellLinesearch(M.R,M.VT),
#                         stopping_criterion=StopWhenAny(
#                             StopWhenGradientNormLess(1e-8),
#                             StopAfterIteration(iteration.full)),
#                         debug = [
#                         :Stop,
#                         :Iteration,
#                         :Cost,
#                         " | ",
#                         AENCDebugEig(Tstep),
#                         "\n",
#                         1,
#                         ])
#     X .= Xres
    # this is for the non-orthogonal one
#     W = PolyModel(AENC_Wnl(M).mexp, AENC_Wnlpoint(X))
#     S = PolyModel(AENC_S(M).mexp, AENC_Spoint(X))
    # for orthogonal case
    # setting a full immersion: need to do this because the derivatives do not trickle down
    MW, XW = toFullDensePolynomial(AENC_Wnl(M), AENC_Wnlpoint(X))
    MS, XS = toFullDensePolynomial(AENC_S(M), AENC_Spoint(X))
    setLinearPart!(MW, XW, AENC_Wlpoint(X))
    
    # needs a normal form transformation!
    MWr, XWr, MRr, XRr = iManifoldMAP(MS, XS, [1; 2], [])
    XWt = zero(MW)
    DensePolySubstitute!(MW, XWt, MW, XW, MWr, XWr)
    
    @show r_max = maximum(sqrt.(sum((reshape(embedscales,1,:) * dataINorig) .^ 2, dims=1)))
    That, Rhat_r = MAPFrequencyDamping(MW, XWt, MRr, XRr, 0.55, output = reshape(embedscales,1,:))
    
    r = range(0, domain(That).right, length=1000)
    omega = abs.(That.(r)/Tstep)
    zeta = -log.(abs.(Rhat_r.(r))) ./ abs.(That.(r))
    return omega, zeta, collect(r * scale)
end

function testAENCManifold()
    M = OAEManifold(10, 2, 3, false)
    p1 = randn(M)
    p2 = randn(M)

    project!(M, p1, p2, p1)
    
    println("p1 constraints")
    display(p2.parts[1]' * p2.parts[1])
    display(p2.parts[1]' * p2.parts[2])

    X1 = zero_vector!(M, p1, p2)

    retract!(M, p1, p2, X1)
    println("p1 constraints")
    display(p1.parts[1]' * p1.parts[1])
    display(p1.parts[1]' * p1.parts[2])

    retract!(M, p1, p1, X1)
    println("p1 constraints")
    display(p1.parts[1]' * p1.parts[1])
    display(p1.parts[1]' * p1.parts[2])
    return
    # M = AENCManifold(ndim, mdim, Sorder, Worder, field::AbstractNumbers=ℝ)
    M = AENCManifold(10, 2, 5, 5, false)
    X = randn(M)
    dataIN = randn(10,100)/10
    dataOUT = randn(10,100)/10
    AENCLoss(M, X, dataIN, dataOUT)
    @time AENCGradient(M, X, dataIN, dataOUT);

    # AENC_Wnlpoint(X) .= 0
    
    let b=1
        # checking S derivatives
        println("S")
        errflag = false
        Xp = deepcopy(X)
        G = AENCGradient(M, X, dataIN, dataOUT)
        Gp = deepcopy(G)
        eps = 1e-7
        S = AENC_Spoint(X)
        Sp = AENC_Spoint(Xp)
        GS = AENC_Spoint(G)
        GSp = AENC_Spoint(Gp)
        for k1=1:size(S,1), k2=1:size(S,2)
            Sp[k1,k2] += eps
            GSp[k1,k2] = (AENCLoss(M, Xp, dataIN, dataOUT) - AENCLoss(M, X, dataIN, dataOUT)) / eps
            relErr = (GSp[k1,k2] - GS[k1,k2]) / GS[k1,k2]
            if abs(relErr) > 1e-4
                errflag = true
                println("S el=", k1, ",", k2, "/", size(Sp,1), ",", size(Sp,2), " E = ", relErr, " G=", GS[k1,k2], " Gp=", GSp[k1,k2])
            end
            Sp[k1,k2] = S[k1,k2]
        end

        # checking Wl derivatives
        println("U, Wl")
        errflag = false
        Xp = deepcopy(X)
        G = AENCGradient(M, X, dataIN, dataOUT)
        Gp = deepcopy(G)
        eps = 1e-7
        S = AENC_Wlpoint(X)
        Sp = AENC_Wlpoint(Xp)
        GS = AENC_Wlpoint(G)
        GSp = AENC_Wlpoint(Gp)
        for k1=1:size(S,1), k2=1:size(S,2)
            Sp[k1,k2] += eps
            GSp[k1,k2] = (AENCLoss(M, Xp, dataIN, dataOUT) - AENCLoss(M, X, dataIN, dataOUT)) / eps
            relErr = (GSp[k1,k2] - GS[k1,k2]) / GS[k1,k2]
            if abs(relErr) > 1e-4
                errflag = true
                println("Wl el=(", k1, "),", k2, "/(", size(Sp,1), "),", size(Sp,2), " E = ", relErr, " G=", GS[k1,k2], " Gp=", GSp[k1,k2])
            end
            Sp[k1,k2] = S[k1,k2]
        end

        # checking Wnl derivatives
        println("Wnl")
        errflag = false
        Xp = deepcopy(X)
        G = AENCGradient(M, X, dataIN, dataOUT)
        Gp = deepcopy(G)
        eps = 1e-7
        S = AENC_Wnlpoint(X)
        Sp = AENC_Wnlpoint(Xp)
        GS = AENC_Wnlpoint(G)
        GSp = AENC_Wnlpoint(Gp)
        for k1=1:size(S,1), k2=1:size(S,2)
            Sp[k1,k2] += eps
            GSp[k1,k2] = (AENCLoss(M, Xp, dataIN, dataOUT) - AENCLoss(M, X, dataIN, dataOUT)) / eps
            relErr = (GSp[k1,k2] - GS[k1,k2]) / GS[k1,k2]
            if abs(relErr) > 1e-4
                errflag = true
                println("Wnl el=", k1, ",", k2, "/", size(Sp,1), ",", size(Sp,2), " E = ", relErr, " G=", GS[k1,k2], " Gp=", GSp[k1,k2])
            end
            Sp[k1,k2] = S[k1,k2]
        end
    end
    
    let b=1
        # checking S derivatives
        println("S")
        errflag = false
        Xp = deepcopy(X)
        G = AENCFitGradient(M, X, dataOUT)
        Gp = deepcopy(G)
        eps = 1e-7
        S = AENC_Spoint(X)
        Sp = AENC_Spoint(Xp)
        GS = AENC_Spoint(G)
        GSp = AENC_Spoint(Gp)
        for k1=1:size(S,1), k2=1:size(S,2)
            Sp[k1,k2] += eps
            GSp[k1,k2] = (AENCFitLoss(M, Xp, dataOUT) - AENCFitLoss(M, X, dataOUT)) / eps
            relErr = (GSp[k1,k2] - GS[k1,k2]) / GS[k1,k2]
            if abs(relErr) > 1e-4
                errflag = true
                println("S el=", k1, ",", k2, "/", size(Sp,1), ",", size(Sp,2), " E = ", relErr, " G=", GS[k1,k2], " Gp=", GSp[k1,k2])
            end
            Sp[k1,k2] = S[k1,k2]
        end

        # checking Wl derivatives
        println("U, Wl")
        errflag = false
        Xp = deepcopy(X)
        G = AENCFitGradient(M, X, dataOUT)
        Gp = deepcopy(G)
        eps = 1e-7
        S = AENC_Wlpoint(X)
        Sp = AENC_Wlpoint(Xp)
        GS = AENC_Wlpoint(G)
        GSp = AENC_Wlpoint(Gp)
        for k1=1:size(S,1), k2=1:size(S,2)
            Sp[k1,k2] += eps
            GSp[k1,k2] = (AENCFitLoss(M, Xp, dataOUT) - AENCFitLoss(M, X, dataOUT)) / eps
            relErr = (GSp[k1,k2] - GS[k1,k2]) / GS[k1,k2]
            if abs(relErr) > 1e-4
                errflag = true
                println("Wl el=(", k1, "),", k2, "/(", size(Sp,1), "),", size(Sp,2), " E = ", relErr, " G=", GS[k1,k2], " Gp=", GSp[k1,k2])
            end
            Sp[k1,k2] = S[k1,k2]
        end

        # checking Wnl derivatives
        println("Wnl")
        errflag = false
        Xp = deepcopy(X)
        G = AENCFitGradient(M, X, dataOUT)
        Gp = deepcopy(G)
        eps = 1e-7
        S = AENC_Wnlpoint(X)
        Sp = AENC_Wnlpoint(Xp)
        GS = AENC_Wnlpoint(G)
        GSp = AENC_Wnlpoint(Gp)
        for k1=1:size(S,1), k2=1:size(S,2)
            Sp[k1,k2] += eps
            GSp[k1,k2] = (AENCFitLoss(M, Xp, dataOUT) - AENCFitLoss(M, X, dataOUT)) / eps
            relErr = (GSp[k1,k2] - GS[k1,k2]) / GS[k1,k2]
            if abs(relErr) > 1e-4
                errflag = true
                println("Wnl el=", k1, ",", k2, "/", size(Sp,1), ",", size(Sp,2), " E = ", relErr, " G=", GS[k1,k2], " Gp=", GSp[k1,k2])
            end
            Sp[k1,k2] = S[k1,k2]
        end
    end
end
