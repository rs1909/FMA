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
  4. the scaling factor that was used to fit all data into the unit ball
  5. uncorrected vector of instantaneous frequencies
  6. uncorrected vector of instantaneous damping
  7. uncorrected vector of instantaneous amplitudes
  8. something mysterious
  9. amplitude of each data point using ``\left\| \boldsymbol{w}\, \boldsymbol{U}\left( \boldsymbol{W} \left( \boldsymbol{x}_k\right)\right) \right\|``
"""
function FoliationIdentify(dataINorig, dataOUTorig, Tstep, embedscales, SysName, freq; orders = (P=7,Q=1,U=5,W=5), iterations = (f=4000, l=30), kappa = 0.2, rank_deficiency = 0, node_rank = 4)
    # need # of iterations for each optimisation
    # need list of tolerances, gradient, cost function
    disable_cache()
    NDIM = size(dataINorig,1)
    nonlin_perbox = 12000
    
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
    S2, U2, W2, Xstar = GaussSouthwellLinearSetup(Misf, Xisf, dataIN, dataOUT, Tstep, freq; perbox=14000, retbox=3, nbox=6, exclude = false)
    Misf, Xisf = GaussSouthwellOptim(Misf, Xisf, dataIN, dataOUT, scale, Tstep, freq; name = SysName, maxit=iterations.f)

    scaleOLD = scale
    @load "ISFdata-$(SysName).bson" Misf Xisf Tstep scale
    @show scale, scaleOLD
    
    #---------------------------------------------------------
    # Normal form
    #---------------------------------------------------------
    MU, XU, MS, XS = ISFNormalForm(Misf, Xisf)

    #---------------------------------------------------------
    # Fitting a manifold as an immersion
    #---------------------------------------------------------
    Mimm = ISFImmersionManifold(2, NDIM, orders.W, kappa; rank_deficiency = rank_deficiency, X=Xstar)
    Ximm = zero(Mimm)
    
    Xres, dataParIN, dataParOUT = ISFImmersionSolve!(Mimm, Ximm, Misf, Xisf, MU, XU, S2, U2, dataIN, dataOUT; maxit=iterations.l, rank_deficiency = rank_deficiency, Tstep = Tstep)
    println("B frequencies [Hz] = ", sort(unique(abs.(angle.(eigvals(ImmersionBpoint(Xres)))/(2*pi))))/Tstep)
    println("B frequencies [rad/s] = ", sort(unique(abs.(angle.(eigvals(ImmersionBpoint(Xres))))))/Tstep)
    
    @save "ManifoldImmersion-$(SysName).bson" Mimm Xres Tstep scale dataParIN dataParOUT
#     @load "ManifoldImmersion-$(SysName).bson" Mimm Xres Tstep scale dataParIN dataParOUT

    MWt, XWt = ImmersionReconstruct(Mimm, Xres, Misf, Xisf, MU, XU)

    #---------------------------------------------------------
    # Frequency, damping calculation
    #---------------------------------------------------------
    Dr = 0.0001
    r = range(0,1,step=Dr)

    M_A, A00 = ManifoldAmplitudeSquare(MWt, XWt)

    ROMdata = (transpose(vec(embedscales)) * Eval(MWt, XWt, dataParIN))
    dataAmps2 = sqrt.(dropdims(sum(ROMdata .^ 2, dims=1),dims=1)) .* sign.(ROMdata[1,:])
    @show size(sign.(ROMdata[1,:])), size(sqrt.(sum(ROMdata .^ 2, dims=1))), size(dataAmps2)
    
    data_freq, data_damp, data_r, freq_old, damp_old, r_old = MAPManifoldFrequencyDamping(MWt, XWt, MS, XS, r, Tstep; output = transpose(vec(embedscales)))
    Wamp = [ sqrt(Eval(M_A, A00, [x])[1]) for x in r_old]
    return data_freq, data_damp, data_r*scale, scale, freq_old, damp_old, r_old*scale, Wamp, dataAmps2*scale
end
