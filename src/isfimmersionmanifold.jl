
struct ISFImmersionManifold{mdim, ndim, deficiency, Worder, C, 𝔽} <: AbstractManifold{𝔽}
    mlist
    M        :: ProductManifold 
    R        :: ProductRetraction 
    VT       :: ProductVectorTransport
    Xstar    :: AbstractArray
end

# mdim : number of immersion parameters
# ndim : system dimensions
@doc raw"""
    M = ISFImmersionManifold(mdim, ndim, Worder, kappa=0.0, field::AbstractNumbers=ℝ)

Creates a manifold representation for the encoder
```math
\hat{\boldsymbol{U}}\left(\boldsymbol{x}\right)=\boldsymbol{U}^{\perp}\boldsymbol{x}-\boldsymbol{W}_{0}\left(\boldsymbol{U}\left(\boldsymbol{x}\right)\right),
```
where ``\boldsymbol{W}_{0}:Z\to\hat{Z}`` with ``D\boldsymbol{W}_{0}\left(\boldsymbol{0}\right)=\boldsymbol{0}``, ``\boldsymbol{U}^{\perp}:X\to\hat{Z}`` is a linear map, ``\boldsymbol{U}^{\perp}\left(\boldsymbol{U}^{\perp}\right)^{T}=\boldsymbol{I}`` and ``\boldsymbol{U}^{\perp}\boldsymbol{W}_{0}\left(\boldsymbol{z}\right)=\boldsymbol{0}``.

Function arguments arguments are
  * `mdim`: dimensionality of the manifold
  * `ndim`: system dimensionality
  * `Worder`: polynomial order of ``\boldsymbol{W}_{0}``
"""
function ISFImmersionManifold(mdim, ndim, Worder, kappa=0.0, field::AbstractNumbers=ℝ; rank_deficiency = 0, X = zeros(ndim))
    # to solve the equation
    #   Uh(y) - B Uh(x) = 0, where
    #   Uh(x) = Up x - W0( U(x) ), where
    #   U(x) is already known
    # To initialise:
    #   Assume that 
    #       - A is the full linear dynamics
    #       - Vs . A = C . Vs represents the dynamics transversal to the manifold
    # In case C has no (nearly) zero singular value, set
    #   B  = C
    #   Up = Vs
    # In case C has singular values, that is C = Cu . Cs . Cv^T, set
    #   B  = (Cu^T . Cv)^{-1} . Cs
    #   Up = Bv^T . Vs
    mlist = (LinearTallManifold(ndim - mdim, ndim - mdim, field), DenseNonlinearManifold(mdim, ndim - mdim, Worder), ConstantManifold(ndim - mdim, field), OrthogonalFlatManifold(ndim, ndim - mdim; field=field))
    M = ProductManifold(mlist...)
    R = ProductRetraction(map(x->getfield(x,:R), mlist)...)
    VT = ProductVectorTransport(map(x->getfield(x,:VT), mlist)...)
    if kappa == 0.0
        return ISFImmersionManifold{mdim, ndim, rank_deficiency, Worder, 0.0, field}(mlist, M, R, VT, X)
    else
        println("C = ", -1/(2*kappa^2))
        return ISFImmersionManifold{mdim, ndim, rank_deficiency, Worder, -1/(2*kappa^2), field}(mlist, M, R, VT, X)
    end
end

function ImmersionB(M::ISFImmersionManifold)
    return M.mlist[1]
end

function ImmersionBpoint(X)
    return X.parts[1]
end

function ImmersionW0(M::ISFImmersionManifold)
    return M.mlist[2]
end

function ImmersionW0point(X)
    return X.parts[2]
end

function ImmersionU0(M::ISFImmersionManifold)
    return M.mlist[3]
end

function ImmersionU0point(X)
    return X.parts[3]
end

function ImmersionWp(M::ISFImmersionManifold)
    return M.mlist[4]
end

function ImmersionWppoint(X)
    return X.parts[4]
end

@doc raw"""
    X = zero(M::ISFImmersionManifold)
    
Creates a zero data structure for a local foliation `ISFImmersionManifold`.
"""
function zero(M::ISFImmersionManifold)
    return ProductRepr(map(zero, M.mlist)...)
end

function randn(M::ISFImmersionManifold)
    return ProductRepr(map(randn, M.mlist)...)
end

function zero_vector!(M::ISFImmersionManifold, X, p)
    return zero_vector!(M.M, X, p)
end

function zero_tangent(M::ISFImmersionManifold)
    return zero_tangent(M.M)
end

function manifold_dimension(M::ISFImmersionManifold)
    return manifold_dimension(M.M)
end

function inner(M::ISFImmersionManifold, p, X, Y)
    return inner(M.M, p, X, Y)
end

function project!(M::ISFImmersionManifold, Y, p, X)
    return project!(M.M, Y, p, X)
end

function retract!(M::ISFImmersionManifold, q, p, X, method::AbstractRetractionMethod)
    return retract!(M.M, q, p, X, M.R)
end

function retract(M::ISFImmersionManifold, p, X, method::AbstractRetractionMethod)
    return retract(M.M, p, X, M.R)
end

function vector_transport_to!(M::ISFImmersionManifold, Y, p, X, q, method::AbstractVectorTransportMethod)
#     println("ISF VECTOR TRANSPORT 1")
    return vector_transport_to!(M.M, Y, p, X, q, method)
end

function vector_transport_to(M::ISFImmersionManifold, p, X, q, method::AbstractVectorTransportMethod)
#     println("ISF VECTOR TRANSPORT 1")
    return vector_transport_to(M.M, p, X, q, method)
end

function EvalUhat(Mimm::ISFImmersionManifold, Ximm, MU, XU, Misf::ISFPadeManifold, Xisf, data)
    dataPar = Eval(MU, XU, Eval(PadeU(Misf), PadeUpoint(Xisf), data) )
    Uox = ImmersionU0point(Ximm) .+ Eval(ImmersionWp(Mimm), ImmersionWppoint(Ximm), data) .- Eval(ImmersionW0(Mimm), ImmersionW0point(Ximm), dataPar)
    return Uox
end

function ISFImmersionLoss(M::ISFImmersionManifold{mdim, ndim, deficiency, Worder, C, field}, X, dataIN, dataOUT, dataParIN, dataParOUT) where {mdim, ndim, deficiency, Worder, C, field}
    datalen = size(dataIN,2)
    # U(x) = Wp.x - W0(z)
    scale = AmplitudeScaling(dataIN, M.Xstar)

    Uox = Eval(ImmersionU0(M), ImmersionU0point(X), dataIN) .+ Eval(ImmersionWp(M), ImmersionWppoint(X), dataIN) .- Eval(ImmersionW0(M), ImmersionW0point(X), dataParIN)
    Uoy = Eval(ImmersionU0(M), ImmersionU0point(X), dataOUT)  .+ Eval(ImmersionWp(M), ImmersionWppoint(X), dataOUT) .- Eval(ImmersionW0(M), ImmersionW0point(X), dataParOUT)
    UoxSq = sum(Uox .^ 2, dims=1)
    L0 = copy(Uoy)
    L0[1:end-deficiency,:] .-= ImmersionBpoint(X)[1:end-deficiency,1:end-deficiency] * Uox[1:end-deficiency,:]
    return sum( (L0 .^ 2) .* exp.(C*UoxSq) ./ scale )/2/datalen
end

function ISFImmersionGradient(M::ISFImmersionManifold{mdim, ndim, deficiency, Worder, C, field}, X, dataIN, dataOUT, dataParIN, dataParOUT) where {mdim, ndim, deficiency, Worder, C, field}
    datalen = size(dataIN,2)
    # U(x) = Wp.x - W0(z)
    scale = AmplitudeScaling(dataIN, M.Xstar)

    Uox = ImmersionU0point(X) .+ Eval(ImmersionWp(M), ImmersionWppoint(X), dataIN) .- Eval(ImmersionW0(M), ImmersionW0point(X), dataParIN)
    Uoy = ImmersionU0point(X) .+ Eval(ImmersionWp(M), ImmersionWppoint(X), dataOUT) .- Eval(ImmersionW0(M), ImmersionW0point(X), dataParOUT)
    UoxSq = sum(Uox .^ 2, dims=1)
    # objfun = sum( (L0 .^ 2) .* exp.(C * UoxSq) ./ scale )/2/datalen
    L0 = Uoy .- ImmersionBpoint(X) * Uox
    L0 = copy(Uoy)
    L0[1:end-deficiency,:] .-= ImmersionBpoint(X)[1:end-deficiency,1:end-deficiency] * Uox[1:end-deficiency,:]
    #
    L0deri1 = L0 .* exp.(C*UoxSq) ./ scale
    L0deri2 = C .* Uox .* sum(L0 .^ 2, dims=1) .* exp.(C*UoxSq) ./ scale
    # L0 otimes Uox
    DB = L0_DF(ImmersionB(M), ImmersionBpoint(X), Uox, L0 = -1.0*L0deri1)/datalen
    DB[end+1-deficiency:end,:] .= 0
    DB[:,end+1-deficiency:end] .= 0
    # 
    DW0 = ( L0_DF(ImmersionW0(M), ImmersionW0point(X), dataParOUT, L0 = -L0deri1) .+ 
            L0_DF(ImmersionW0(M), ImmersionW0point(X), dataParIN, L0 = transpose(ImmersionBpoint(X))*L0deri1 .- L0deri2) )/datalen
    #
#     @show size(ImmersionWppoint(X)), size(dataOUT), size(L0)
    DWp = ( L0_DF(ImmersionWp(M), ImmersionWppoint(X), dataOUT, L0 = L0deri1) .+
            L0_DF(ImmersionWp(M), ImmersionWppoint(X), dataIN, L0 = -1.0*transpose(ImmersionBpoint(X))*L0deri1 .+ L0deri2) )/datalen
    DU0 = ( L0_DF(ImmersionU0(M), ImmersionU0point(X), dataOUT, L0 = L0deri1) .+
            L0_DF(ImmersionU0(M), ImmersionU0point(X), dataIN, L0 = -1.0*transpose(ImmersionBpoint(X))*L0deri1 .+ L0deri2) )/datalen
    return ProductRepr(DB, DW0, DU0, DWp)
end

function ISFImmersionRiemannianGradient(M::ISFImmersionManifold, X, dataIN, dataOUT, dataParIN, dataParOUT)
    gr = ISFImmersionGradient(M::ISFImmersionManifold, X, dataIN, dataOUT, dataParIN, dataParOUT)
    return project(M, X, gr)
end

@doc raw"""
    MWt, XWt = ImmersionReconstruct(Mimm, Ximm, Misf, Xisf, MU, XU)
    
Creates a manifold immersion from the locally accurate foliation represented by `Mimm, Ximm`, 
the full foliation represented by `Misf, Xisf` and the normal form transformation `MU, XU`.
"""
function ImmersionReconstruct(Mimm, Xres, Misf, Xisf, MU, XU; rank_deficiency = 0)
    #---------------------------------------------------------
    #
    # Reconstructing the manifold immersion
    #
    #---------------------------------------------------------
    # Solving the equation
    #   U( W( z ) )      = z
    #   u0 + W^p W( z )  = W0( z )
    # which is
    #   W(z) = Wc + Wlin z + Wt(z)
    #   U(x) = Uc + Ulin x + Ut(x)
    # define
    #   Q = [Wp Ulin]^T
    #   DW(0) = Q^{-1} (0, I)^T
    # iterate
    #   Wt(z) = Q^{-1} [ W0(z), -Ut( Wlin z + Wt(z) ) ]^T
    # note: 'dout' is out for U, but 'in' for W
    
    # U = Uout o PadeU(Xisf)
    Ulin = getLinearPart(MU, XU)*PadeUpoint(Xisf).parts[2]'
    dout, din = size(Ulin)
    Q = zeros(din, din)
    Q[1:din-dout,:] .= ImmersionWppoint(Xres)'
    Q[1+din-dout:end,:] .= Ulin

    function UNewt(Mimm, Xres, Misf, Xisf, MU, XU, MW, XW, z)
        X = [ EvalUhat(Mimm, Xres, MU, XU, Misf, Xisf, reshape(Eval(MW, XW, z),:,1));
              Eval(MU, XU, Eval(PadeU(Misf), PadeUpoint(Xisf), reshape(Eval(MWt, XWt, z),:,1)) ) - z]
        return vec(X)
    end

    # checking for singularity
    if minimum(abs.(eigvals(Q))) < 0.1
        println("Matrix Q is nearly singular, the two encoders U \\hat{U} do not define a good coordinate system")
    end

    # Newton iteration
    MW0, XW0 = toFullDensePolynomial(ImmersionW0(Mimm), ImmersionW0point(Xres))
    order = max(PolyOrder(MU), PolyOrder(MW0)) # could be the sum of them, too!
        
    MWt = DensePolyManifold(dout, din, order)
    XWt = zero(MWt)
    for k=1:2*order
        XWt2 = XWt - Q \ fromFunction(MWt, (z) -> UNewt(Mimm, Xres, Misf, Xisf, MU, XU, MWt, XWt, z))
        XWt .= XWt2
    end
    println("Steady state = ", Eval(MWt, XWt, zeros(dout)))
    
    return MWt, XWt
end

@doc raw"""
    Xres, dataParIN, dataParOUT = ISFImmersionSolve!(Mimm, Ximm, Misf, Xisf, Uout, Wperp, Sperp, dataIN, dataOUT; maxit = 25)
    
Solves the optimisation problem
```math
\arg\min_{\boldsymbol{S},\boldsymbol{U}}\sum_{k=1}^{N}\left\Vert \boldsymbol{x}_{k}\right\Vert ^{-2}\exp\left(-\frac{1}{2\kappa^{2}}\left\Vert \hat{\boldsymbol{U}}\left(\boldsymbol{x}_{k}\right)\right\Vert ^{2}\right)\left\Vert \boldsymbol{B}\hat{\boldsymbol{U}}\left(\boldsymbol{x}_{k}\right)-\hat{\boldsymbol{U}}\left(\boldsymbol{y}_{k}\right)\right\Vert ^{2}
```
"""
function ISFImmersionSolve!(Mimm, Ximm, Misf, Xisf, MU, XU, S2, U2, dataIN, dataOUT; maxit = 25, rank_deficiency = 0, Tstep = 1.0)
    #---------------------------------------------------------
    #
    # Fitting a manifold as an immersion
    #
    #---------------------------------------------------------
    # the parameters of the manifold using the submersion we have calculated
    dout, din = size(PadeUpoint(Xisf).parts[1]')
    
    dataParIN = Eval(MU, XU, Eval(PadeU(Misf), PadeUpoint(Xisf), dataIN) )
    dataParOUT = Eval(MU, XU, Eval(PadeU(Misf), PadeUpoint(Xisf), dataOUT) )

    # To initialise:
    #   Assume that 
    #       - A is the full linear dynamics
    #       - Vs . A = C . Vs represents the dynamics transversal to the manifold
    # In case C has no (nearly) zero singular value, set
    #   B  = C
    #   Up = Vs
    # In case C has singular values, that is C = Cu . Cs . Cv^T, set
    #   B  = (Cu^T . Cv)^{-1} . Cs
    #   Up = Bv^T . Vs

    # calculate the invariant subspace corresponding to the largest (in magnitude) eigenvalues
    F0 = schur(S2)
    p = sortperm(abs.(F0.values), order=Base.Order.Reverse)
    sel = zeros(Bool, length(p))
    sel[p[1:end-rank_deficiency]] .= true
    F = ordschur(F0,sel)
    # debug display
    if rank_deficiency > 0
        println("Min included modulus ", minimum(abs.(F.values[1:end-rank_deficiency])))
        println("Max excluded modulus ", maximum(abs.(F.values[end-rank_deficiency+1:end])))
    end
    println("Included frequencies [Hz] = ", sort(unique(abs.(angle.(F.values[1:end-rank_deficiency])/(2*pi))))/Tstep)
    println("Included frequencies [rad/s] = ", sort(unique(abs.(angle.(F.values[1:end-rank_deficiency]))))/Tstep)

    Sperp = F.T
    Sperp[end+1-rank_deficiency:end,:] .= 0
    Sperp[:,end+1-rank_deficiency:end] .= 0
    Wperp = F.Z' * U2
    
    # setting up the linear part
    @show size(ImmersionWppoint(Ximm)), size(U2)
    ImmersionBpoint(Ximm) .= Sperp
    ImmersionWppoint(Ximm) .= Wperp'
    ImmersionW0point(Ximm) .= 0.0
    
    # checking if problem is well defined
    Ulin = getLinearPart(MU, XU)*PadeUpoint(Xisf).parts[2]'
    dout, din = size(Ulin)
    Q = zeros(din, din)
    Q[1:din-dout,:] .= ImmersionWppoint(Ximm)'
    Q[1+din-dout:end,:] .= Ulin
    if minimum(abs.(eigvals(Q))) < 0.1
        println("Matrix Q is nearly singular, the two encoders U \\hat{U} do not define a good coordinate system")
    end

    # do the actual optimisation
    Xres = trust_regions(Mimm, 
                (M, x) -> ISFImmersionLoss(M, x, dataIN, dataOUT, dataParIN, dataParOUT), 
                (M, x) -> ISFImmersionRiemannianGradient(M, x, dataIN, dataOUT, dataParIN, dataParOUT),
                ApproxHessianFiniteDifference(
            Mimm,
            Ximm,
            (M, x) -> ISFImmersionRiemannianGradient(M, x, dataIN, dataOUT, dataParIN, dataParOUT);
            steplength=2^(-22),
            retraction_method=Mimm.R,
            vector_transport_method=Mimm.VT,
        ),
                Ximm,
                retraction_method = Mimm.R,
                  max_trust_region_radius=0.5,
                  stopping_criterion=StopWhenAny(
                        StopWhenGradientNormLess(1e-8),
                        StopAfterIteration(maxit)
                        ),
                debug = [
        :Stop,
        :Iteration,
        :Cost,
        " | ",
#         DebugEig(Tstep),
        "\n",
        1,
    ])
    Ximm .= Xres
    return Xres, dataParIN, dataParOUT
end

function testImmersion()
    mdim = 2
    ndim = 10
    order = 3
    samples = 10000
    Wperp = randn(ndim - mdim, ndim)
    dataIN = randn(ndim, samples)
    dataOUT = randn(ndim, samples)
    dataParIN = randn(mdim, samples)
    dataParOUT = randn(mdim, samples)
    
    M = ISFImmersionManifold(mdim, ndim, order, 1.0; X = randn(ndim))
    X = randn(M)
#     ImmersionWppoint(X) .= Wperp
#     ImmersionBpoint(X) .= 0
#     display(M)
    display(X)
#     return
    ISFImmersionLoss(M, X, dataIN, dataOUT, dataParIN, dataParOUT)
    grad = ISFImmersionGradient(M, X, dataIN, dataOUT, dataParIN, dataParOUT)
    
    # test the gradient
    Xp = deepcopy(X)
    gradp = deepcopy(grad)
    eps = 1e-6
    for k=1:length(ImmersionBpoint(X))
        ImmersionBpoint(Xp)[k] += eps
        ImmersionBpoint(gradp)[k] = (ISFImmersionLoss(M, Xp, dataIN, dataOUT, dataParIN, dataParOUT) 
                                        - ISFImmersionLoss(M, X, dataIN, dataOUT, dataParIN, dataParOUT))/eps
        ImmersionBpoint(Xp)[k] = ImmersionBpoint(X)[k]
        @show ImmersionBpoint(grad)[k], ImmersionBpoint(gradp)[k] - ImmersionBpoint(grad)[k]
    end
    for k=1:length(ImmersionW0point(X))
        ImmersionW0point(Xp)[k] += eps
        ImmersionW0point(gradp)[k] = (ISFImmersionLoss(M, Xp, dataIN, dataOUT, dataParIN, dataParOUT) 
                                        - ISFImmersionLoss(M, X, dataIN, dataOUT, dataParIN, dataParOUT))/eps
        ImmersionW0point(Xp)[k] = ImmersionW0point(X)[k]
        @show ImmersionW0point(grad)[k], ImmersionW0point(gradp)[k] - ImmersionW0point(grad)[k]
    end
    for k=1:length(ImmersionU0point(X))
        ImmersionU0point(Xp)[k] += eps
        ImmersionU0point(gradp)[k] = (ISFImmersionLoss(M, Xp, dataIN, dataOUT, dataParIN, dataParOUT) 
                                        - ISFImmersionLoss(M, X, dataIN, dataOUT, dataParIN, dataParOUT))/eps
        ImmersionU0point(Xp)[k] = ImmersionU0point(X)[k]
        @show ImmersionU0point(grad)[k], ImmersionU0point(gradp)[k] - ImmersionU0point(grad)[k]
    end
    for k=1:length(ImmersionWppoint(X))
        ImmersionWppoint(Xp)[k] += eps
        ImmersionWppoint(gradp)[k] = (ISFImmersionLoss(M, Xp, dataIN, dataOUT, dataParIN, dataParOUT) 
                                        - ISFImmersionLoss(M, X, dataIN, dataOUT, dataParIN, dataParOUT))/eps
        ImmersionWppoint(Xp)[k] = ImmersionWppoint(X)[k]
        @show ImmersionWppoint(grad)[k], ImmersionWppoint(gradp)[k] - ImmersionWppoint(grad)[k]
    end

    
    grad = ISFImmersionRiemannianGradient(M, X, dataIN, dataOUT, dataParIN, dataParOUT)
    
    Misf = M
    Xisf = X
    Xres = trust_regions(Misf, 
                (M, x) -> ISFImmersionLoss(M, x, dataIN, dataOUT, dataParIN, dataParOUT), 
                (M, x) -> ISFImmersionRiemannianGradient(M, x, dataIN, dataOUT, dataParIN, dataParOUT),
                ApproxHessianFiniteDifference(
            Misf,
            Xisf,
            (M, x) -> ISFImmersionRiemannianGradient(M, x, dataIN, dataOUT, dataParIN, dataParOUT);
            steplength=2^(-8),
            retraction_method=Misf.R,
            vector_transport_method=Misf.VT,
        ),
                Xisf,
                retraction_method = Misf.R,
                  max_trust_region_radius=0.5,
                  stopping_criterion=StopWhenAny(
                        StopWhenGradientNormLess(1e-10),
                        StopAfterIteration(2000)
                        ),
                debug = [
        :Stop,
        :Iteration,
        :Cost,
        " | ",
#         DebugEig(Tstep),
        "\n",
        1,
    ])

    grad = ISFImmersionRiemannianGradient(Misf, Xres, dataIN, dataOUT, dataParIN, dataParOUT)
end
