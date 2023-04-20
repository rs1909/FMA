@doc raw"""
    FoliationIdentify(dataIN, dataOUT, Tstep, 
                      embedscales, SysName, freq;
                      orders = (P=9,Q=1,U=5,W=3), 
                      iterations = (f=4000, l=30), kappa = 0.2)

This is a convenience method that integrates all steps of calculating a reduced order model.
  1. Estimates the linear dynamics about the origin, only using part of the data.
  2. Calculates an invariant foliation about the invariant subspace corresponding, which has the closest frequancy to `freq` in Hz.
  3. Performs a normal form transformation of the invariant foliation
  4. Calculates a locally accurate complementary invariant foliation, containing the invariant manifold that has the same dynamics as the foliation calculated in point 2.
  5. Extracts the invariant manifold from the locally accurate invariant foliation.
  6. Performs a correction of the instantaneous frequency and damping values using the invariant manifold calculated in point 5.

Input:

  * `dataIN` is a two dimensional array, each column is an ``\boldsymbol{x}_k`` value, 
  * `dataOUT` is a two dimensional array, each column is an ``\boldsymbol{y}_k``
  * `Tstep` is the time step between ``\boldsymbol{x}_k`` and ``\boldsymbol{y}_k``
  * `embedscales`, denoted by ``\boldsymbol{w}`` is a matrix applied to ``\boldsymbol{x}_k``, ``\boldsymbol{y}_k``, used to calculate the amplitude of a signal
  * `freq` is the frequency that the invariant foliation is calculated for.
  * `orders` is a named tuple, specifies the polynomial order of ``\boldsymbol{P}``, ``\boldsymbol{Q}`` and ``\boldsymbol{U}``.
  * `iterations` is a named tuple, `f` is the maximum itearion when solving for a foliation, 
    `l` is the maximum number of iterations when solving for a locally accurate foliation.
  * `kappa` is the size of the neighbourhood of a the invariant manifold considered, when calculating a locally accurate foliation.
  
Output:
is a tuple with elements
  1. vector of instantaneous frequencies
  2. vector of instantaneous damping
  3. vector of instantaneous amplitudes
  4. 1
  5. 1
  6. 1
  7. 1
  8. 1
  9. amplitude of each data point using ``\left\| \boldsymbol{w}\, \boldsymbol{U}\left( \boldsymbol{W} \left( \boldsymbol{x}_k\right)\right) \right\|``
"""
function FoliationIdentify(dataINorig, dataOUTorig, Tstep, embedscales, SysName, freq; orders = (P=7,Q=1,U=5,W=5), iterations = (f=4000, l=30), kappa = 0.2, rank_deficiency = 0, node_rank = 4)
    # need # of iterations for each optimisation
    # need list of tolerances, gradient, cost function
    disable_cache()
    NDIM = size(dataINorig,1)
    nonlin_perbox = 120000
    # 1. pruning, scaling
    # 2. identify a linear system. Use a neighbourhood of the supposed equilibrium with bump function
    #    phi(x - xs) |y - xs - A (x - xs)|
    # 3. Calculate invariant subspaces of matrix A -> S1, U1, W1; S2, U2, W2
    # 4. Set up the compressed foliation and optimise
    # 5. Set up the local foliation and optimise
    # 6. Reconstruct the invariant manifold
    # 7. Calculate normal form (manifold style) of the foliation
    # 8. Identify frequencies and damping ratios
    #---------------------------------------------------------
    # Scaling the data. Is it really needed?
    #---------------------------------------------------------
    dataIN, dataOUT, scale, scalePTS = dataPrune(dataINorig, dataOUTorig; perbox=nonlin_perbox)
    #---------------------------------------------------------
    # ISF
    #---------------------------------------------------------
    din = NDIM
    dout = 2

    # ISF
    Misf = ISFPadeManifold(dout, din, orders.P, orders.Q, orders.U, zeros(din, dout), node_rank = node_rank)
    Xisf = zero(Misf)
    S2, U2, W2, Xstar = GaussSouthwellLinearSetup(Misf, Xisf, dataIN, dataOUT, Tstep, freq; perbox=140000, retbox=6, nbox=6, exclude = false)
    Misf, Xisf = GaussSouthwellOptim(Misf, Xisf, dataIN, dataOUT, scale, Tstep, freq; name = SysName, maxit=iterations.f)

    scaleOLD = scale
#     @load "ISFdata-$(SysName).bson" Misf Xisf Tstep scale
    # ignoring  Misf...
    bs = BSON.parse("ISFdata-$(SysName).bson")
    Xisf = BSON.raise_recursive(bs[:Xisf], Main)
    Tstep = BSON.raise_recursive(bs[:Tstep], Main)
    scale = BSON.raise_recursive(bs[:scale], Main)
    @show scale, scaleOLD
    
    #---------------------------------------------------------
    # Fitting a manifold as an immersion
    #---------------------------------------------------------
    Mimm = ISFImmersionManifold(2, NDIM, orders.W, kappa; rank_deficiency = rank_deficiency, X=Xstar)
    Ximm = zero(Mimm)
    
    Xres, dataParIN, dataParOUT = ISFImmersionSolve!(Mimm, Ximm, Misf, Xisf, S2, U2, dataIN, dataOUT; maxit=iterations.l, rank_deficiency = rank_deficiency, Tstep = Tstep)
    @save "ManifoldImmersion-$(SysName).bson" Mimm Xres Tstep scale dataParIN dataParOUT
#     @load "ManifoldImmersion-$(SysName).bson" Mimm Xres Tstep scale dataParIN dataParOUT
    
    println("B frequencies [Hz] = ", sort(unique(abs.(angle.(eigvals(ImmersionBpoint(Xres)))/(2*pi))))/Tstep)
    println("B frequencies [rad/s] = ", sort(unique(abs.(angle.(eigvals(ImmersionBpoint(Xres))))))/Tstep)
    @show scale, scaleOLD

    MWt, XWt = ImmersionReconstruct(Mimm, Xres, Misf, Xisf)
    
    println("Reconstruction residuals")
    rd = EvalUhat(Mimm, Xres, Misf, Xisf, Eval(MWt, XWt, dataParIN))
    rz = Eval(PadeU(Misf), PadeUpoint(Xisf), Eval(MWt, XWt, dataParIN)) .- dataParIN
    # this really should be more accurate ...
    @show maximum( sqrt.(sum(rd .^ 2,dims=2)) )
    @show maximum( sqrt.(sum(rz .^ 2,dims=2)) )
    
    #---------------------------------------------------------
    # Normal form (manifold style)
    #---------------------------------------------------------
#     MU, XU, MS, XS = ISFNormalForm(Misf, Xisf)
    MS, XS = toFullDensePolynomial(PadeP(Misf), PadePpoint(Xisf))
    MWr, XWr, MRr, XRr, MW, XW, MR, XR = iManifoldMAP(MS, XS, collect(1:2), [], order = PolyOrder(MS))

    MWc, XWc0 = toComplex(MWt, XWt)
    XWc = zero(MWc)
    DensePolySubstitute!(MWc, XWc, MWc, XWc0, MW, XW)
    println("Reconstruction residuals")
    cdat = zeros(Complex{eltype(dataParIN)}, size(dataParIN)...)
    cdat[1,:] .= (dataParIN[1,:] .+ 1im*dataParIN[2,:]) ./ 2
    cdat[2,:] .= (dataParIN[1,:] .- 1im*dataParIN[2,:]) ./ 2
    rd = EvalUhat(Mimm, Xres, Misf, Xisf, Eval(MWc, XWc, cdat))
    rz = Eval(PadeU(Misf), PadeUpoint(Xisf), Eval(MWc, XWc, cdat)) .- Eval(MW, XW, cdat)
    # this really should be more accurate ...
    @show maximum( sqrt.(abs.(sum(rd .* conj.(rd),dims=2))) )
    @show maximum( sqrt.(abs.(sum(rz .* conj.(rz),dims=2))) )
    
    #---------------------------------------------------------
    # Frequency, damping calculation
    #---------------------------------------------------------
    # maximum amplitude of W o U o data
    amp_max = maximum(sqrt.(sum((embedscales * Eval(MWt, XWt, dataParIN)) .^ 2, dims=1)))
    # create the composite immersion

    That, Rhat_r = MAPFrequencyDamping(MWc, XWc, MR, XR, amp_max, output = reshape(embedscales,1,:))
#     XWt_f = zero(MWt)
#     DensePolySubstitute!(MWt, XWt_f, MWt, XWt, MWr, XWr)
#     That, Rhat_r = MAPFrequencyDamping(MWt, XWt_f, MRr, XRr, r_max, output = reshape(embedscales,1,:))
    
    ROMdata = (transpose(vec(embedscales)) * Eval(MWt, XWt, dataParIN))
    dataAmps = sqrt.(dropdims(sum(ROMdata .^ 2, dims=1),dims=1)) .* sign.(ROMdata[1,:])
    r = range(0, domain(That).right, length=1000)
    omega = abs.(That.(r)/Tstep)
    zeta = -log.(abs.(Rhat_r.(r))) ./ abs.(That.(r))
    return omega, zeta, collect(r * scale), 1, 1, 1, 1, 1, dataAmps*scale, MW, XW
end
