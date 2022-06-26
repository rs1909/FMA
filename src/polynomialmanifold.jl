## ---------------------------------------------------------------------------------------
## Submersion
## 
## ---------------------------------------------------------------------------------------

struct PolynomialManifold{mdim, ndim, order, 𝔽} <: AbstractManifold{𝔽}
    mlist
    M        :: ProductManifold 
    R        :: ProductRetraction
    VT       :: ProductVectorTransport
end

function PolyOrder(M::PolynomialManifold{mdim, ndim, order, field}) where {mdim, ndim, order, field}
    return order
end

function inner(M::PolynomialManifold{mdim, ndim, order, field}, p, X, Y) where {mdim, ndim, order, field}
    return inner(M.M, p, X, Y)
end

function project!(M::PolynomialManifold{mdim, ndim, order, field}, Y, p, X) where {mdim, ndim, order, field}
    return ProductRepr(map(project!, M.mlist, Y.parts, p.parts, X.parts))
end

function retract!(M::PolynomialManifold{mdim, ndim, order, field}, q, p, X, method::AbstractRetractionMethod) where {mdim, ndim, order, field}
    return retract!(M.M, q, p, X, method)
end

function vector_transport_to!(M::PolynomialManifold{mdim, ndim, order, field}, Y, p, X, q, method::AbstractVectorTransportMethod) where {mdim, ndim, order, field}
#     println("PolynomialManifold VECTOR TRANSPORT 1")
    return vector_transport_to!(M.M, Y, p, X, q, method)
end

function SubmersionManifold(mdim, ndim, order, B=nothing, field::AbstractNumbers=ℝ)
#     mlist = tuple([MultiSphereManifold(ndim, mdim); [RandomTensorManifold(repeat([ndim],k), mdim) for k=2:order]]...)
    mlist = tuple([OrthogonalFlatManifold(ndim, mdim); [RandomTensorManifold(repeat([ndim],k), mdim, B) for k=2:order]]...)
    M = ProductManifold(map(x->getfield(x,:M), mlist)...)
    R = ProductRetraction(map(x->getfield(x,:R), mlist)...)
    VT = ProductVectorTransport(map(x->getfield(x,:VT), mlist)...)
    return PolynomialManifold{mdim, ndim, order, field}(mlist, M, R, VT)
end

# here mdim is the high-dimensional output and ndim is the low dimensional input
function ImmersionManifold(mdim, ndim, order, B=nothing, field::AbstractNumbers=ℝ)
    mlist = tuple([OrthogonalTallManifold(ndim, mdim); [RandomTensorManifold(repeat([ndim],k), mdim, B) for k=2:order]]...)
    M = ProductManifold(map(x->getfield(x,:M), mlist)...)
    R = ProductRetraction(map(x->getfield(x,:R), mlist)...)
    VT = ProductVectorTransport(map(x->getfield(x,:VT), mlist)...)
    return PolynomialManifold{mdim, ndim, order, field}(mlist, M, R, VT)
end

function PolynomialFlatManifold(mdim, ndim, order, B=nothing, field::AbstractNumbers=ℝ)
    mlist = tuple([LinearFlatManifold(ndim, mdim); [RandomTensorManifold(repeat([ndim],k), mdim, B) for k=2:order]]...)
    M = ProductManifold(map(x->getfield(x,:M), mlist)...)
    R = ProductRetraction(map(x->getfield(x,:R), mlist)...)
    VT = ProductVectorTransport(map(x->getfield(x,:VT), mlist)...)
    return PolynomialManifold{mdim, ndim, order, field}(mlist, M, R, VT)
end

function PolynomialTallManifold(mdim, ndim, order, B=nothing, field::AbstractNumbers=ℝ)
    mlist = tuple([LinearTallManifold(ndim, mdim); [RandomTensorManifold(repeat([ndim],k), mdim, B) for k=2:order]]...)
    M = ProductManifold(map(x->getfield(x,:M), mlist)...)
    R = ProductRetraction(map(x->getfield(x,:R), mlist)...)
    VT = ProductVectorTransport(map(x->getfield(x,:VT), mlist)...)
    return PolynomialManifold{mdim, ndim, order, field}(mlist, M, R, VT)
end

function zero(M::PolynomialManifold{mdim, ndim, order, field}) where {mdim, ndim, order, field}
    return ProductRepr(map(zero, M.mlist))
end

function randn(M::PolynomialManifold{mdim, ndim, order, field}) where {mdim, ndim, order, field}
    return ProductRepr(map(randn, M.mlist))
end

function zero_vector!(M::PolynomialManifold{mdim, ndim, order, field}, X, p) where {mdim, ndim, order, field}
    return zero_vector!(M.M, X, p)
end

function manifold_dimension(M::PolynomialManifold{mdim, ndim, order, field}) where {mdim, ndim, order, field}
    return manifold_dimension(M.M)
end

function Eval(M::PolynomialManifold{mdim, ndim, order, field}, X, data, topdata = nothing) where {mdim, ndim, order, field}
#     return mapreduce((x,y) -> begin tmp = Eval(x, y, data, topdata); @show size(tmp); return tmp; end, .+, M.mlist, X.parts)
    return mapreduce((x,y) -> Eval(x, y, data, topdata), .+, M.mlist, X.parts)
end

function DF(M::PolynomialManifold{mdim, ndim, order, field}, X, data) where {mdim, ndim, order, field}
    return ProductRepr(map((x,y) -> DF(x, y, data), M.mlist, X.parts))
end

function wDF(M::PolynomialManifold{mdim, ndim, order, field}, X, data, topdata) where {mdim, ndim, order, field}
    return ProductRepr(map((x,y) -> wDF(x, y, data, topdata), M.mlist, X.parts))
end

function DFdt(M::PolynomialManifold{mdim, ndim, order, field}, X, data, dt) where {mdim, ndim, order, field}
    return mapreduce((x,y,z) -> DFdt(x, y, data, z), .+, M.mlist, X.parts, dt.parts)
end

function DwDFdt(M::PolynomialManifold{mdim, ndim, order, field}, X, data, w, dt) where {mdim, ndim, order, field}
    return ProductRepr(map((x,y,z) -> DwDFdt(x, y, data, w, z), M.mlist, X.parts, dt.parts))
end

function vD2Fw(M::PolynomialManifold{mdim, ndim, order, field}, X, data, topdata) where {mdim, ndim, order, field}
#     return mapreduce((x,y) -> begin tmp = vD2Fw(x, y, data, topdata); @show size(tmp); return tmp; end, .+, M.mlist, X.parts)
    return mapreduce((x,y) -> vD2Fw(x, y, data, topdata), .+, M.mlist, X.parts)
end

function Gradient(M::PolynomialManifold{mdim, ndim, order, field}, X, data, topdata = nothing) where {mdim, ndim, order, field}
    return ProductRepr(map((x,y) -> Gradient(x, y, data, topdata), M.mlist, X.parts))
end

function wJF(M::PolynomialManifold{mdim, ndim, order, field}, X, data, topdata = nothing) where {mdim, ndim, order, field}
    return mapreduce((x,y) -> wJF(x, y, data, topdata), .+, M.mlist, X.parts)
end

function DwJFv(M::PolynomialManifold{mdim, ndim, order, field}, X, data, topdata) where {mdim, ndim, order, field}
    return ProductRepr(map((x,y) -> DwJFv(x, y, data, topdata), M.mlist, X.parts))
end

function DwJFdt(M::PolynomialManifold{mdim, ndim, order, field}, X, data, topdata, dt) where {mdim, ndim, order, field}
    return mapreduce((x,y,z) -> DwJFdt(x, y, data, topdata, z), .+, M.mlist, X.parts, dt.parts)
end

# note that data is a simple array, NOT an array of arrays
function Jacobian(M::PolynomialManifold{mdim, ndim, order, field}, X, data) where {mdim, ndim, order, field}
    return mapreduce((x,y) -> Jacobian(x, y, data), .+, M.mlist, X.parts)
end

function Hessian(M::PolynomialManifold{mdim, ndim, order, field}, X, data) where {mdim, ndim, order, field}
    return mapreduce((x,y) -> Hessian(x, y, data), .+, M.mlist, X.parts)
end

function PolynomialLoss(M::PolynomialManifold{mdim, ndim, order, field}, X, dataIN, dataOUT) where {mdim, ndim, order, field}
    return sum( ((Eval(M, X, [dataIN]) .- dataOUT ) .^2) ./ sum(abs.(dataIN) .^ 2, dims = 1))
end

function PolynomialDerivative(M::PolynomialManifold{mdim, ndim, order, field}, X, dataIN, dataOUT) where {mdim, ndim, order, field}
    tmp = 2*(Eval(M, X, [dataIN]) .- dataOUT) ./ sum(abs.(dataIN) .^ 2, dims = 1)
    return wDF(M, X, [dataIN], tmp)
end

function PolynomialHessian(M::PolynomialManifold{mdim, ndim, order, field}, X, dt, dataIN, dataOUT) where {mdim, ndim, order, field}
    G0 = 2*(Eval(M, X, [dataIN]) .- dataOUT) ./ sum(abs.(dataIN) .^ 2, dims = 1)
    G1_delta = 2*DFdt(M, X, [dataIN], dt) ./ sum(abs.(dataIN) .^ 2, dims = 1)
    H1 = wDF(M, X, [dataIN], G1_delta)
    H2 = DwDFdt(M, X, [dataIN], G0, dt)
    return H1 + H2
end

function PolynomialGradient(M::PolynomialManifold{mdim, ndim, order, field}, X, dataIN, dataOUT) where {mdim, ndim, order, field}
    tmp = 2*(Eval(M, X, [dataIN]) .- dataOUT) ./ sum(abs.(dataIN) .^ 2, dims = 1)
    return Gradient(M, X, [dataIN], tmp)
end

# S(U(x)) - y
function PolyCompLoss(SM, SX, UM, UX, dataIN, dataOUT)
    Uox = Eval(UM, UX, [dataIN])
    SoUox = Eval(SM, SX, [Uox])
    return sum((SoUox .- dataOUT) .^2 )
end

function PolyCompDerivative(SM, SX, UM, UX, dataIN, dataOUT)
    Uox = Eval(UM, UX, [dataIN])
    SoUox_m_y = Eval(SM, SX, [Uox]) .- dataOUT
    G_S = wDF(SM, SX, [Uox], 2*SoUox_m_y)
    DS = wJF(SM, SX, [Uox], 2*SoUox_m_y)
    G_U = wDF(UM, UX, [dataIN], DS)
    return G_S, G_U
end

function PolyCompHessian(SM, SX, UM, UX, dS, dU, dataIN, dataOUT)
    Uox = Eval(UM, UX, [dataIN])
    SoUox_m_y = Eval(SM, SX, [Uox]) .- dataOUT
    G_S = wDF(SM, SX, [Uox], 2*SoUox_m_y)
    DS = wJF(SM, SX, [Uox], 2*SoUox_m_y)
    G_U = wDF(UM, UX, [dataIN], DS)
    return G_S, G_U
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

function testPoly()
    din = 5
    dout = 3
    M1 = SubmersionManifold(dout, din, 3) # output 3, input 4, order 4
    M2 = PolynomialTallManifold(dout, din, 3) # output 3, input 4, order 4
#     M2 = SubmersionManifold(3, 4, 3) # output 3, input 4, order 4
    zero(M1)
    x1 = randn(M1)
    zero(M2)
    x2 = randn(M2)
    dataIN = randn(din,1000)
    dataIN2 = randn(din,1000)
    dataOUT = randn(dout,1000)
    Eval(M1, x1, [dataIN], dataOUT)
    Eval(M2, x2, [dataIN], dataOUT)
    wDF(M1, x1, [dataIN], dataOUT)
    wDF(M2, x2, [dataIN], dataOUT)
    Gradient(M1, x1, [dataIN], dataOUT)
    Gradient(M2, x2, [dataIN], dataOUT)
    wJF(M1, x1, [dataIN], dataOUT)
    wJF(M2, x2, [dataIN], dataOUT)
    @time PolynomialLoss(M1, x1, dataIN, dataOUT)
    @time PolynomialLoss(M1, x1, dataIN, dataOUT)
    @time PolynomialGradient(M1, x1, dataIN, dataOUT)
    @time PolynomialGradient(M1, x1, dataIN, dataOUT)
    
    grad = PolynomialDerivative(M1, x1, dataIN, dataOUT)
    xp = deepcopy(x1)
    gradp = deepcopy(grad)
    eps = 1e-6
    flag = false
    for l=1:length(x1.parts[1])
        xp.parts[1][l] += eps
        gradp.parts[1][l] = sum(PolynomialLoss(M1, xp, dataIN, dataOUT) - PolynomialLoss(M1, x1, dataIN, dataOUT)) / eps
        relErr = (gradp.parts[1][l] - grad.parts[1][l]) / grad.parts[1][l]
        if abs(relErr) > 1e-4
            flag = true
            print("|Poly|")
            @show relErr
        end
        xp.parts[1][l] = x1.parts[1][l]
    end
    for k1=2:length(x1.parts)
        for k2=1:length(x1.parts[k1].parts)
            for l=1:length(x1.parts[k1].parts[k2])
                xp.parts[k1].parts[k2][l] += eps
                gradp.parts[k1].parts[k2][l] = sum(PolynomialLoss(M1, xp, dataIN, dataOUT) - PolynomialLoss(M1, x1, dataIN, dataOUT)) / eps
                relErr = (gradp.parts[k1].parts[k2][l] - grad.parts[k1].parts[k2][l]) / grad.parts[k1].parts[k2][l]
                if abs(relErr) > 1e-4
                    flag = true
                    print("|Poly|")
                    @show relErr
                end
                xp.parts[k1].parts[k2][l] = x1.parts[k1].parts[k2][l]
            end
        end
    end
    # -------------------------------------------------------
    println("Poly Hessian")
    dt = randn(M1)
    hess = PolynomialHessian(M1, x1, dt, dataIN, dataOUT)
    xp = deepcopy(x1)
    hessp = deepcopy(hess)
    eps = 1e-6
    flag = false
    for l=1:length(x1.parts[1])
        xp.parts[1][l] += eps
        hessp.parts[1][l] = inner(M1.M, x1, PolynomialDerivative(M1, xp, dataIN, dataOUT) .- PolynomialDerivative(M1, x1, dataIN, dataOUT), dt)/eps
        relErr = (hessp.parts[1][l] - hess.parts[1][l]) / hess.parts[1][l]
        if abs(relErr) > 1e-4
            flag = true
            print("|Poly H|")
            @show relErr
        end
        xp.parts[1][l] = x1.parts[1][l]
    end
    for k1=2:length(x1.parts)
        for k2=1:length(x1.parts[k1].parts)
            for l=1:length(x1.parts[k1].parts[k2])
                xp.parts[k1].parts[k2][l] += eps
                hessp.parts[k1].parts[k2][l] = inner(M1.M, x1, PolynomialDerivative(M1, xp, dataIN, dataOUT) .- PolynomialDerivative(M1, x1, dataIN, dataOUT), dt)/eps
                relErr = (hessp.parts[k1].parts[k2][l] - hess.parts[k1].parts[k2][l]) / hess.parts[k1].parts[k2][l]
                if abs(relErr) > 1e-4
                    flag = true
                    print("|Poly H|")
                    @show relErr
                end
                xp.parts[k1].parts[k2][l] = x1.parts[k1].parts[k2][l]
            end
        end
    end
    
    # -------------------------------------------------------
    
    let dataIN = randn(5,1000), dataOUT = randn(3,1000)

        println("Poly Comp Derivative")
        SM = PolynomialTallManifold(3, 3, 3)
        UM = SubmersionManifold(3, 5, 3) 
        SX = randn(SM)
        UX = randn(UM)
        
        dS, dU = PolyCompDerivative(SM, SX, UM, UX, dataIN, dataOUT)
        # S part
        SXp = deepcopy(SX)
        dSp = deepcopy(dS)
        eps = 1e-6
        flag = false
        for l=1:length(SX.parts[1])
            SXp.parts[1][l] += eps
            dSp.parts[1][l] = sum(PolyCompLoss(SM, SXp, UM, UX, dataIN, dataOUT) - PolyCompLoss(SM, SX, UM, UX, dataIN, dataOUT)) / eps
            relErr = (dSp.parts[1][l] - dS.parts[1][l]) / dS.parts[1][l]
            if abs(relErr) > 1e-4
                flag = true
                print("|PolyComp S|")
                @show relErr
            end
            SXp.parts[1][l] = SX.parts[1][l]
        end
        for k1=2:length(SX.parts)
            for k2=1:length(SX.parts[k1].parts)
                for l=1:length(SX.parts[k1].parts[k2])
                    SXp.parts[k1].parts[k2][l] += eps
                    dSp.parts[k1].parts[k2][l] = sum(PolyCompLoss(SM, SXp, UM, UX, dataIN, dataOUT) - PolyCompLoss(SM, SX, UM, UX, dataIN, dataOUT)) / eps
                    relErr = (dSp.parts[k1].parts[k2][l] - dS.parts[k1].parts[k2][l]) / dS.parts[k1].parts[k2][l]
                    if abs(relErr) > 1e-4
                        flag = true
                        print("|PolyComp S|")
                        @show relErr
                    end
                    SXp.parts[k1].parts[k2][l] = SX.parts[k1].parts[k2][l]
                end
            end
        end
        # U part
        UXp = deepcopy(UX)
        dUp = deepcopy(dU)
        eps = 1e-6
        flag = false
        for l=1:length(UX.parts[1])
            UXp.parts[1][l] += eps
            dUp.parts[1][l] = sum(PolyCompLoss(SM, SX, UM, UXp, dataIN, dataOUT) - PolyCompLoss(SM, SX, UM, UX, dataIN, dataOUT)) / eps
            relErr = (dUp.parts[1][l] - dU.parts[1][l]) / dU.parts[1][l]
            if abs(relErr) > 1e-4
                flag = true
                print("|PolyComp U|")
                @show relErr
            end
            UXp.parts[1][l] = UX.parts[1][l]
        end
        for k1=2:length(UX.parts)
            for k2=1:length(UX.parts[k1].parts)
                for l=1:length(UX.parts[k1].parts[k2])
                    UXp.parts[k1].parts[k2][l] += eps
                    dUp.parts[k1].parts[k2][l] = sum(PolyCompLoss(SM, SX, UM, UXp, dataIN, dataOUT) - PolyCompLoss(SM, SX, UM, UX, dataIN, dataOUT)) / eps
                    relErr = (dUp.parts[k1].parts[k2][l] - dU.parts[k1].parts[k2][l]) / dU.parts[k1].parts[k2][l]
                    if abs(relErr) > 1e-4
                        flag = true
                        print("|PolyComp U|")
                        @show relErr
                    end
                    UXp.parts[k1].parts[k2][l] = UX.parts[k1].parts[k2][l]
                end
            end
        end
    end #let
    
#     return nothing
    
    # -------------------------------------------------------
    grad = wDF(M2, x2, [dataIN], dataOUT)
    xp = deepcopy(x2)
    gradp = deepcopy(grad)
    eps = 1e-6
    flag = false
    for l=1:length(x2.parts[1])
        xp.parts[1][l] += eps
        gradp.parts[1][l] = sum(Eval(M2, xp, [dataIN], dataOUT) - Eval(M2, x2, [dataIN], dataOUT)) / eps
        relErr = (gradp.parts[1][l] - grad.parts[1][l]) / grad.parts[1][l]
        if abs(relErr) > 1e-4
            flag = true
        end
        xp.parts[1][l] = x2.parts[1][l]
    end
    for r=2:length(x2.parts)
        for k=1:length(x2.parts[r].parts)
            for l=1:length(x2.parts[r].parts[k])
                xp.parts[r].parts[k][l] += eps
                gradp.parts[r].parts[k][l] = sum(Eval(M2, xp, [dataIN], dataOUT) - Eval(M2, x2, [dataIN], dataOUT)) / eps
                relErr = (gradp.parts[r].parts[k][l] - grad.parts[r].parts[k][l]) / grad.parts[r].parts[k][l]
                if abs(relErr) > 1e-4
                    flag = true
                end
                xp.parts[r].parts[k][l] = x2.parts[r].parts[k][l]
            end
        end
    end
    if flag
        println("Poly wDF")
        @show diff = gradp - grad
    end
    
    # DFdt
    w = randn(M2)
    grad = DFdt(M2, x2, [dataIN], w)

    xp = deepcopy(x2)
    gradp = zero(grad)
    eps = 1e-6
    for l=1:length(x2.parts[1])
        xp.parts[1][l] += eps
        tmp = (Eval(M2, xp, [dataIN]) - Eval(M2, x2, [dataIN])) / eps
        gradp .+= tmp * w.parts[1][l]
        xp.parts[1][l] = x2.parts[1][l]
    end
    for r=2:length(x2.parts)
        for k=1:length(x2.parts[r].parts)
            for l=1:length(x2.parts[r].parts[k])
                xp.parts[r].parts[k][l] += eps
                tmp = (Eval(M2, xp, [dataIN]) - Eval(M2, x2, [dataIN])) / eps
                gradp .+= tmp * w.parts[r].parts[k][l]
                xp.parts[r].parts[k][l] = x2.parts[r].parts[k][l]
            end
        end
    end
#     if flag
        println("Poly DFdt")
        @show maximum(abs.(gradp - grad))
#         @show gradp
#         @show grad
#     end
    
    # now the hessian
    w = randn(M2)
    hess = DwDFdt(M2, x2, [dataIN], dataOUT, w)
    
    # test accuracy
    xp = deepcopy(x2)
    hessp = deepcopy(hess)
    eps = 1e-6
    flag = false
    for l=1:length(x2.parts[1])
        xp.parts[1][l] += eps
        hessp.parts[1][l] = inner(M2.M, x2, wDF(M2, xp, [dataIN], dataOUT) .- wDF(M2, x2, [dataIN], dataOUT), w)/eps
        relErr = (hessp.parts[1][l] - hess.parts[1][l]) / hess.parts[1][l]
        if abs(relErr) > 1e-4
            flag = true
            println("r = 1, l = ", l, " E = ", relErr, " fd=", hessp.parts[1][l], " an=", hess.parts[1][l])
        end
        xp.parts[1][l] = x2.parts[1][l]
    end
    for r=2:length(x2.parts)
        for k=1:length(x2.parts[r].parts)
            for l=1:length(x2.parts[r].parts[k])
                xp.parts[r].parts[k][l] += eps
                hessp.parts[r].parts[k][l] = inner(M2.M, x2, wDF(M2, xp, [dataIN], dataOUT) .- wDF(M2, x2, [dataIN], dataOUT), w)/eps
                relErr = (hessp.parts[r].parts[k][l] - hess.parts[r].parts[k][l]) / hess.parts[r].parts[k][l]
                if abs(relErr) > 1e-4
                    flag = true
                    println("r = ", r, "k = ", k, "/", length(x2.parts[r].parts), " l = ", l, " E = ", relErr)
                end
                xp.parts[r].parts[k][l] = x2.parts[r].parts[k][l]
            end
        end
    end
    if flag
        println("Poly DwDFdt")
        @show diff = hessp - hess
    end

    grad = DwJFv(M2, x2, [dataIN, dataIN2], dataOUT)
    xp = deepcopy(x2)
    gradp = deepcopy(grad)
    eps = 1e-6
    flag = false
    for l=1:length(x2.parts[1])
        xp.parts[1][l] += eps
        gradp.parts[1][l] = sum( (wJF(M2, xp, [dataIN], dataOUT) - wJF(M2, x2, [dataIN], dataOUT)) .* dataIN2 ) / eps
        relErr = (gradp.parts[1][l] - grad.parts[1][l]) / grad.parts[1][l]
        if abs(relErr) > 1e-4
            flag = true
            println("DwJFv r = 1, l = ", l, " E = ", relErr, " fd=", gradp.parts[1][l], " an=", grad.parts[1][l])
        end
        xp.parts[1][l] = x2.parts[1][l]
    end
    for r=2:length(x2.parts)
        for k=1:length(x2.parts[r].parts)
            for l=1:length(x2.parts[r].parts[k])
                xp.parts[r].parts[k][l] += eps
                gradp.parts[r].parts[k][l] = sum( (wJF(M2, xp, [dataIN], dataOUT) - wJF(M2, x2, [dataIN], dataOUT)) .* dataIN2 ) / eps
                relErr = (gradp.parts[r].parts[k][l] - grad.parts[r].parts[k][l]) / grad.parts[r].parts[k][l]
                if abs(relErr) > 1e-4
                    flag = true
                    println("DwJFv r = ", r, " k = ", k, "/", length(x2.parts[r].parts), " l = ", l, "/", length(x2.parts[r].parts[k]), " E = ", relErr)
                end
                xp.parts[r].parts[k][l] = x2.parts[r].parts[k][l]
            end
        end
    end
    if flag
        println("Errors in Poly DwJFv")
#         @show diff = gradp - grad
#         @show diff.parts[1]
#         @show gradp.parts[1]
#         @show grad.parts[1]
    end

    #-----------------------------------------


# function DwJFdt(M::PolynomialManifold{mdim, ndim, order, field}, X, data, topdata, dt) where {mdim, ndim, order, field}
#     return mapreduce((x,y,z) -> DFdt(x, y, data, data, z), .+, M.mlist, X.parts, dt.parts)
# end
    
    println("Poly DwJFdt")
    dt = randn(M2)
    res_orig = DwJFdt(M2, x2, [dataIN], dataOUT, dt)
#     @show size(res_orig)
    eps = 1e-6
    res = deepcopy(res_orig)
    dataINp = deepcopy(dataIN)
    for k=1:size(dataIN,1)
        dataINp[k,:] .+= eps
#         @show size(Eval(M2, x2, [dataINp], dataOUT)), size(dataINp), size(dataOUT)
        res[k,:] .= dropdims(sum((DFdt(M2, x2, [dataINp], dt) .- DFdt(M2, x2, [dataIN], dt)) .* dataOUT, dims=1),dims=1) / eps
        dataINp[k,:] = dataIN[k,:]
    end
    @show maximum(abs.(res_orig .- res))

    #--------------------------------------------------------
    
    
    println("Poly wJF")
    res_orig = wJF(M2, x2, [dataIN], dataOUT)
#     @show size(res_orig)
    eps = 1e-6
    res = deepcopy(res_orig)
    dataINp = deepcopy(dataIN)
    for k=1:size(dataIN,1)
        dataINp[k,:] .+= eps
#         @show size(Eval(M2, x2, [dataINp], dataOUT)), size(dataINp), size(dataOUT)
        res[k,:] .= (Eval(M2, x2, [dataINp], dataOUT) .- Eval(M2, x2, [dataIN], dataOUT)) / eps
        dataINp[k,:] = dataIN[k,:]
    end
    @show maximum(abs.(res_orig .- res))
    
    println("Poly Jacobian")
    res_orig = Jacobian(M2, x2, dataIN)
    eps = 1e-6
    res = deepcopy(res_orig)
    dataINp = deepcopy(dataIN)
    for k=1:size(dataIN,1)
        dataINp[k,:] .+= eps
#         @show size(Eval(M2, x2, [dataINp], dataOUT))
        res[:,k,:] = (Eval(M2, x2, [dataINp]) - Eval(M2, x2, [dataIN])) / eps
        dataINp[k,:] = dataIN[k,:]
    end
    @show maximum(abs.(res_orig .- res))

    
    println("Poly vD2Fw")
    res_orig = vD2Fw(M2, x2, [dataIN, dataIN2], dataOUT)
    eps = 1e-6
    res = deepcopy(res_orig)
    dataINp = deepcopy(dataIN)
    for k=1:size(dataIN,1)
        dataINp[k,:] .+= eps
        tmp = (wJF(M2, x2, [dataINp], dataOUT) - wJF(M2, x2, [dataIN], dataOUT)) / eps
        res[k,:] = dropdims(sum(tmp .* dataIN2,dims=1),dims=1)
        dataINp[k,:] = dataIN[k,:]
    end
    @show maximum(abs.(res_orig .- res))
    return nothing
end
