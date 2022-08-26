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

function HessProjection(M::LinearManifold{ndim, n, transp}) where {ndim, n, transp}
    HessProjection(M.M)
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

function makeCache(M::LinearManifold, X, data, topdata = nothing)
    return nothing
end

function updateCache!(DV::Nothing, M::LinearManifold, X, data, topdata = nothing)
    return nothing
end

function Eval(M::LinearManifold{ndim, n, false}, X, data, topdata = nothing; DV = nothing) where {ndim, n}
    if topdata == nothing
        return X * data[1]
    else
#         @show size(X), size(topdata), size(data[1])
        return dropdims(sum((transpose(X) * topdata) .* data[1], dims=1),dims=1)
    end
    return nothing
end

function Eval(M::LinearManifold{ndim, n, true}, X, data, topdata = nothing; DV = nothing) where {ndim, n}
    if topdata == nothing
#         @show size(X), size(data[1]), size(data[1][1])
        return transpose(X) * data[1]
    else
#         @show size(X), size(topdata), size(data[1])
        return dropdims(sum((X * topdata) .* data[1], dims=1),dims=1)
    end
    return nothing
end

function wDF(M::LinearManifold{ndim, n, false}, X, data, topdata) where {ndim, n}
    @assert (size(topdata,1) == size(X,1)) && (size(data[1],1) == size(X,2)) "********* BAD SIZE *********"
    return topdata * transpose(data[1])
end

function wDF(M::LinearManifold{ndim, n, true}, X, data, topdata) where {ndim, n}
    @assert (size(topdata,1) == size(X,2)) && (size(data[1],1) == size(X,1)) "********* BAD SIZE *********"
    return data[1] * transpose(topdata)
end

# the other derivatives
function DFdt(M::LinearManifold{ndim, n, false}, X, data, dt) where {ndim, n}
    @assert (n == size(dt,1)) && (size(data[1],1) == size(dt,2)) "********* BAD SIZE *********"
    return dt * data[1]
end

function DFdt(M::LinearManifold{ndim, n, true}, X, data, dt) where {ndim, n}
    @assert (n == size(dt,2)) && (size(data[1],1) == size(dt,1)) "********* BAD SIZE *********"
    return transpose(dt) * data[1]
end

# this is zero
function DwDFdt(M::LinearManifold{ndim, n, transp}, X, data, w, dt) where {ndim, n, transp}
    return zero_tangent(M)
end

# this is zero
function vD2Fw(M::LinearManifold{ndim, n, transp}, X, data, topdata) where {ndim, n, transp}
    return zeros(ndim, size(data[1],2))
end

function Gradient(M::LinearManifold{ndim, n, transp}, X, data, topdata) where {ndim, n, transp}
    deri = wDF(M, X, data, topdata)
    return project!(M.M, deri, X, deri)
end

function wJF(M::LinearManifold{ndim, n, false}, X, data, topdata) where {ndim, n}
    return transpose(X) * topdata
end

function wJF(M::LinearManifold{ndim, n, true}, X, data, topdata) where {ndim, n}
    return X * topdata
end

function DwJFv(M::LinearManifold{ndim, n, false}, X, data, topdata) where {ndim, n}
    # wDF = topdata * transpose(data[1])
    # J(wDF)*v
    return topdata * transpose(data[2])
end

function DwJFv(M::LinearManifold{ndim, n, true}, X, data, topdata) where {ndim, n}
    # wDF = data[1] * transpose(topdata)
    # J(wDF)*v
    return data[2] * transpose(topdata)
end

function DwJFdt(M::LinearManifold{ndim, n, false}, X, data, topdata, dt) where {ndim, n}
    # wJF = transpose(X) * topdata
    # DwJFdt = D(wJF)*dt
    return transpose(dt) * topdata
end

function DwJFdt(M::LinearManifold{ndim, n, true}, X, data, topdata, dt) where {ndim, n}
    # wJF = X * topdata
    # DwJFdt = D(wJF)*dt
    return dt * topdata
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

# same as wDF
function L0_DF(M::LinearManifold{ndim, n, transp}, X, DV, data, L0, ii) where {ndim, n, transp}
    return wDF(M, X, [data], L0)
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

function testLin()
    M2 = LinearFlatManifold(4, 3)
    x2 = randn(M2)
    dataIN = randn(4,10)
    dataIN2 = randn(4,10)
    dataOUT = randn(3,10)

    getel(M2, x2, (1, 3))
    Eval(M2, x2, [dataIN], dataOUT)
    wDF(M2, x2, [dataIN], dataOUT)
    Gradient(M2, x2, [dataIN], dataOUT)
    wJF(M2, x2, [dataIN], dataOUT)
    
    grad = wDF(M2, x2, [dataIN], dataOUT)
    xp = deepcopy(x2)
    gradp = deepcopy(grad)
    eps = 1e-6
    flag = false
    for l=1:length(x2)
        xp[l] += eps
        gradp[l] = sum(Eval(M2, xp, [dataIN], dataOUT) - Eval(M2, x2, [dataIN], dataOUT)) / eps
        relErr = (gradp[l] - grad[l]) / grad[l]
        if abs(relErr) > 1e-4
            flag = true
        end
        xp[l] = x2[l]
    end
    if flag
        println("Lin wDF")
        @show diff = gradp - grad
    end
    
    # DFdt
    w = randn(M2)
    grad = DFdt(M2, x2, [dataIN], w)

    xp = deepcopy(x2)
    gradp = zero(grad)
    eps = 1e-6
    for l=1:length(x2)
        xp[l] += eps
        tmp = (Eval(M2, xp, [dataIN]) - Eval(M2, x2, [dataIN])) / eps
        gradp .+= tmp * w[l]
        xp[l] = x2[l]
    end
#     if flag
        println("Lin DFdt")
        @show diff = gradp - grad
        @show gradp
        @show grad
#     end
    
    # now the hessian
    w = randn(M2)
    hess = DwDFdt(M2, x2, [dataIN], dataOUT, w)
    
    # test accuracy
    xp = deepcopy(x2)
    hessp = deepcopy(hess)
    eps = 1e-6
    flag = false
    for l=1:length(x2)
        xp[l] += eps
        hessp[l] = inner(M2.M, x2, wDF(M2, xp, [dataIN], dataOUT) .- wDF(M2, x2, [dataIN], dataOUT), w)/eps
        relErr = (hessp[l] - hess[l]) / hess[l]
        if abs(relErr) > 1e-4
            flag = true
            println("k = ", k, "/", length(x2.parts), " leaf=", is_leaf(M2,k), " l = ", l, " E = ", relErr)
        end
        xp[l] = x2[l]
    end
    if flag
        println("Lin DwDFdt")
        @show diff = hessp - hess
        @show diff.parts[3]
        @show hessp.parts[3]
        @show hess.parts[3]
    end

    # test wJF
    # the Jacobian is a list of matrices
    # that is Eval differentiated with respect to dimensions, but for each element in the list at the same time
    println("wJF")
    res_orig = wJF(M2, x2, [dataIN], dataOUT)
    eps = 1e-6
#     xp = deepcopy(x2)
    res = deepcopy(res_orig)
    dataINp = deepcopy(dataIN)
    for k=1:size(dataIN,1)
        dataINp[k,:] .+= eps
        res[k,:] = (Eval(M2, x2, [dataINp], dataOUT) - Eval(M2, x2, [dataIN], dataOUT)) / eps
        dataINp[k,:] = dataIN[k,:]
    end
    @show res_orig .- res
    
    println("vD2Fw")
    res_orig = vD2Fw(M2, x2, [dataIN, dataIN2], dataOUT)
    eps = 1e-6
#     xp = deepcopy(x2)
    res = deepcopy(res_orig)
    dataINp = deepcopy(dataIN)
    for k=1:size(dataIN,1)
        dataINp[k,:] .+= eps
        tmp = (wJF(M2, x2, [dataINp], dataOUT) - wJF(M2, x2, [dataIN], dataOUT)) / eps
        res[k,:] = dropdims(sum(tmp .* dataIN2,dims=1),dims=1)
        dataINp[k,:] = dataIN[k,:]
    end
    @show res_orig .- res
    @show res_orig
    @show res
end
