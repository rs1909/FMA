var documenterSearchIndex = {"docs":
[{"location":"freqdamp.html#Instantaneous-frequencies,-damping-ratios","page":"Oscillations","title":"Instantaneous frequencies, damping ratios","text":"","category":"section"},{"location":"freqdamp.html","page":"Oscillations","title":"Oscillations","text":"ODEManifoldFrequencyDamping","category":"page"},{"location":"freqdamp.html#FoliationsManifoldsAutoencoders.ODEManifoldFrequencyDamping","page":"Oscillations","title":"FoliationsManifoldsAutoencoders.ODEManifoldFrequencyDamping","text":"freq, damp, r = ODEManifoldFrequencyDamping(W, R, r0; output=nothing)\n\nCalculates the instantaneous frequencies and damping ratios of the ROM boldsymbolW boldsymbolR, where boldsymbolR is in the real normal form\n\nbeginpmatrixdotz_1\ndotz_2\nendpmatrix = beginpmatrixz_1f_rleft(z_1^2+z_2^2right)-z_2f_rleft(z_1^2+z_2^2right)\nz_1f_ileft(z_1^2+z_2^2right)+z_2f_rleft(z_1^2+z_2^2right)\nendpmatrix\n\nThe input parameters are\n\nW is boldsymbolW: decoder or manifold immersion\nS is boldsymbolS: nonlinear map\nr0: a vector of amplitudes that result is calculated at. This must be a reasonably fine mesh, because a finite difference is taken with respect to this mesh.\noutput: an optional linear map, that \n\n\n\n\n\n","category":"function"},{"location":"freqdamp.html","page":"Oscillations","title":"Oscillations","text":"MAPManifoldFrequencyDamping","category":"page"},{"location":"freqdamp.html#FoliationsManifoldsAutoencoders.MAPManifoldFrequencyDamping","page":"Oscillations","title":"FoliationsManifoldsAutoencoders.MAPManifoldFrequencyDamping","text":"freq, damp, r, freq_old, damp_old, r_old = MAPManifoldFrequencyDamping(W, S, r0, Tstep; output=nothing)\n\nCalculates the instantaneous frequencies and damping ratios of the ROM boldsymbolW boldsymbolS, where boldsymbolS is in the real normal form\n\nbeginpmatrixz_1\nz_2\nendpmatrix mapsto beginpmatrixz_1f_rleft(z_1^2+z_2^2right)-z_2f_rleft(z_1^2+z_2^2right)\nz_1f_ileft(z_1^2+z_2^2right)+z_2f_rleft(z_1^2+z_2^2right)\nendpmatrix\n\nThe input parameters are\n\nW is boldsymbolW: decoder or manifold immersion\nS is boldsymbolS: nonlinear map\nTstep is the time step that one application of boldsymbolS represents\nr0: a vector of amplitudes that result is calculated at. This must be a reasonably fine mesh, because a finite difference is taken with respect to this mesh.\noutput: an optional linear map, that \n\n\n\n\n\n","category":"function"},{"location":"invariancecalculations.html#Map-and-Vector-Field-Transformations","page":"Direct methods","title":"Map and Vector Field Transformations","text":"","category":"section"},{"location":"invariancecalculations.html","page":"Direct methods","title":"Direct methods","text":"The following methods calculate invariant foliations and invariant manifolds of maps and vectopr fields.","category":"page"},{"location":"invariancecalculations.html","page":"Direct methods","title":"Direct methods","text":"Let us consider the following four-dimensional map","category":"page"},{"location":"invariancecalculations.html","page":"Direct methods","title":"Direct methods","text":"function trivialMAP(z)\n    om1 = 0.12\n    om2 = 1\n    dm1 = 0.002\n    dm2 = 0.003\n    return [(exp(dm1)*z[4]*(z[1]^3*z[2] + z[4]^2 + z[4]^4) + 2*z[1]*cos(om1) - 2*z[2]*sin(om1))/(2.0*exp(dm1)),\n            -0.5*z[2]^3 - z[1]^2*z[2]^2*z[4] - (3*z[2]*z[3]*z[4])/4. + (z[2]*cos(om1) + z[1]*sin(om1))/exp(dm1),\n            (-3*z[1]*z[2]^3*z[3])/4.0 + z[3]^5 - (3*z[1]^3*z[2]*z[4])/4.0 + (z[3]*cos(om2) - z[4]*sin(om2))/exp(dm2),\n            (z[2]*z[3]^4)/4.0 + (z[2]*z[3]^3*z[4])/2.0 - z[4]^5 + (z[4]*cos(om2) + z[3]*sin(om2))/exp(dm2)]\nend","category":"page"},{"location":"invariancecalculations.html","page":"Direct methods","title":"Direct methods","text":"To calculate the invariant foliation corresponding to a two-simensional invariant subspace we use","category":"page"},{"location":"invariancecalculations.html","page":"Direct methods","title":"Direct methods","text":"using FoliationsManifoldsAutoencoders\n\nMF = DensePolyManifold(4, 4, 5)\nXF = fromFunction(MF, trivial)\n\nMWt, XWt, MS, XS = iFoliationMAP(MF, XF, [3, 4], [])","category":"page"},{"location":"invariancecalculations.html","page":"Direct methods","title":"Direct methods","text":"To calculate the invariant manifold corresponding to the same subspace we use","category":"page"},{"location":"invariancecalculations.html","page":"Direct methods","title":"Direct methods","text":"MWt, XWt, MR, XR = iManifoldMAP(MF, XF, [3, 4], [])","category":"page"},{"location":"invariancecalculations.html","page":"Direct methods","title":"Direct methods","text":"One can also calculate the frequency and damping curves using","category":"page"},{"location":"invariancecalculations.html","page":"Direct methods","title":"Direct methods","text":"Dr = 0.0001\nr = range(0,1,step=Dr)\nopscal = ones(1,4)/4\nfrequency, damping, amplitude = MAPManifoldFrequencyDamping(MWt, XWt, MR, XR, r, 1.0; output = opscal)","category":"page"},{"location":"invariancecalculations.html","page":"Direct methods","title":"Direct methods","text":"The relevant functions are the following:","category":"page"},{"location":"invariancecalculations.html","page":"Direct methods","title":"Direct methods","text":"iFoliationMAP","category":"page"},{"location":"invariancecalculations.html#FoliationsManifoldsAutoencoders.iFoliationMAP","page":"Direct methods","title":"FoliationsManifoldsAutoencoders.iFoliationMAP","text":"MUr, XUr, MSr, XSr, MU, XU, MS, XS = iFoliationMAP(MF::DensePolyManifold, XF, vars, par; order = PolyOrder(MF))\n\nCalculates the invariant foliation of nonlinear map boldsymbolF with respect to selected eigenvectors of the Jacobian DboldsymbolF(boldsymbol0).\n\nWe solve the invariance equation\n\nboldsymbolU circ boldsymbolF = boldsymbolS circ boldsymbolU\n\nfor boldsymbolU and boldsymbolS using polynomial expansion. \n\nThe input are \n\nMF, XF represent the dense polynomial expansion of function \\boldsymbol{F}`. \nvars is a vector of integers, selecting the eigenvectors of DboldsymbolF(boldsymbol0).\npar are the indices of input variable of boldsymbolF, which do not have dynamics and can be treated as parameters.\norder is the polynomial order for which the calculation is carried out. The default is the same order as the order of MF, XF, but it can be greater for increased accuracy.\n\nThe output is\n\nMUr, XUr represent boldsymbolU in real coordinates\nMSr, XSr represent boldsymbolS in real coordinates\nMU, XU represent boldsymbolU in complex coordinates\nMS, XS represent boldsymbolS in complex coordinates\n\nNote that it calculates the complex result first, which has the minimal number of parameters, then it transforms the result into real form.\n\n\n\n\n\n","category":"function"},{"location":"invariancecalculations.html","page":"Direct methods","title":"Direct methods","text":"iFoliationVF","category":"page"},{"location":"invariancecalculations.html#FoliationsManifoldsAutoencoders.iFoliationVF","page":"Direct methods","title":"FoliationsManifoldsAutoencoders.iFoliationVF","text":"MUr, XUr, MSr, XSr, MU, XU, MS, XS = iFoliationVF(MF::DensePolyManifold, XF, vars, par; order = PolyOrder(MF))\n\nCalculates the invariant foliation of nonlinear vector field boldsymbolF with respect to selected eigenvectors of the Jacobian DboldsymbolF(boldsymbol0).\n\nWe solve the invariance equation\n\nD boldsymbolU boldsymbolF = boldsymbolS circ boldsymbolU\n\nfor boldsymbolU and boldsymbolS using polynomial expansion. \n\nThe input are \n\nMF, XF represent the dense polynomial expansion of function \\boldsymbol{F}`. \nvars is a vector of integers, selecting the eigenvectors of DboldsymbolF(boldsymbol0).\npar are the indices of input variable of boldsymbolF, which do not have dynamics and can be treated as parameters.\norder is the polynomial order for which the calculation is carried out. The default is the same order as the order of MF, XF, but it can be greater for increased accuracy.\n\nThe output is\n\nMUr, XUr represent boldsymbolU in real coordinates\nMSr, XSr represent boldsymbolS in real coordinates\nMU, XU represent boldsymbolU in complex coordinates\nMS, XS represent boldsymbolS in complex coordinates\n\nNote that it calculates the complex result first, which has the minimal number of parameters, then it transforms the result into real form.\n\n\n\n\n\n","category":"function"},{"location":"invariancecalculations.html","page":"Direct methods","title":"Direct methods","text":"iManifoldMAP","category":"page"},{"location":"invariancecalculations.html#FoliationsManifoldsAutoencoders.iManifoldMAP","page":"Direct methods","title":"FoliationsManifoldsAutoencoders.iManifoldMAP","text":"MWr, XWr, MRr, XRr, MW, XW, MR, XR = iManifoldMAP(MF0::DensePolyManifold, XF0, vars, par; order = PolyOrder(MF0))\n\nCalculates the invariant manifold of nonlinear map boldsymbolF with respect to selected eigenvectors of the Jacobian DboldsymbolF(boldsymbol0).\n\nWe solve the invariance equation\n\nboldsymbolW circ boldsymbolR = boldsymbolF circ boldsymbolW\n\nfor boldsymbolU and boldsymbolS using polynomial expansion. \n\nThe input are \n\nMF, XF represent the dense polynomial expansion of function \\boldsymbol{F}`. \nvars is a vector of integers, selecting the eigenvectors of DboldsymbolF(boldsymbol0).\npar are the indices of input variable of boldsymbolF, which do not have dynamics and can be treated as parameters.\norder is the polynomial order for which the calculation is carried out. The default is the same order as the order of MF, XF, but it can be greater for increased accuracy.\n\nThe output is\n\nMWr, XUr represent boldsymbolW in real coordinates\nMWr, XSr represent boldsymbolR in real coordinates\nMW, XW represent boldsymbolW in complex coordinates\nMR, XR represent boldsymbolR in complex coordinates\n\nNote that it calculates the complex result first, which has the minimal number of parameters, then it transforms the result into real form.\n\n\n\n\n\n","category":"function"},{"location":"invariancecalculations.html","page":"Direct methods","title":"Direct methods","text":"iManifoldVF","category":"page"},{"location":"invariancecalculations.html#FoliationsManifoldsAutoencoders.iManifoldVF","page":"Direct methods","title":"FoliationsManifoldsAutoencoders.iManifoldVF","text":"MWr, XWr, MRr, XRr, MW, XW, MR, XR = iManifoldVF(MF0::DensePolyManifold, XF0, vars, par; order = PolyOrder(MF0))\n\nCalculates the invariant manifold of nonlinear vector field boldsymbolF with respect to selected eigenvectors of the Jacobian DboldsymbolF(boldsymbol0).\n\nWe solve the invariance equation\n\nD boldsymbolW boldsymbolR = boldsymbolF circ boldsymbolW\n\nfor boldsymbolU and boldsymbolS using polynomial expansion. \n\nThe input are \n\nMF, XF represent the dense polynomial expansion of function \\boldsymbol{F}`. \nvars is a vector of integers, selecting the eigenvectors of DboldsymbolF(boldsymbol0).\npar are the indices of input variable of boldsymbolF, which do not have dynamics and can be treated as parameters.\norder is the polynomial order for which the calculation is carried out. The default is the same order as the order of MF, XF, but it can be greater for increased accuracy.\n\nThe output is\n\nMWr, XUr represent boldsymbolW in real coordinates\nMWr, XSr represent boldsymbolR in real coordinates\nMW, XW represent boldsymbolW in complex coordinates\nMR, XR represent boldsymbolR in complex coordinates\n\nNote that it calculates the complex result first, which has the minimal number of parameters, then it transforms the result into real form.\n\n\n\n\n\n","category":"function"},{"location":"localfoliation.html#Locally-Invariant-Foliations","page":"Locally accurate foliation","title":"Locally Invariant Foliations","text":"","category":"section"},{"location":"localfoliation.html","page":"Locally accurate foliation","title":"Locally accurate foliation","text":"These are methods that find a locally accurate invariant foliation in the neighbourhood of an invariant manifold.","category":"page"},{"location":"localfoliation.html","page":"Locally accurate foliation","title":"Locally accurate foliation","text":"ISFImmersionManifold","category":"page"},{"location":"localfoliation.html#FoliationsManifoldsAutoencoders.ISFImmersionManifold","page":"Locally accurate foliation","title":"FoliationsManifoldsAutoencoders.ISFImmersionManifold","text":"M = ISFImmersionManifold(mdim, ndim, Worder, kappa=0.0, field::AbstractNumbers=ℝ)\n\nCreates a manifold representation for the encoder\n\nhatboldsymbolUleft(boldsymbolxright)=boldsymbolU^perpboldsymbolx-boldsymbolW_0left(boldsymbolUleft(boldsymbolxright)right)\n\nwhere boldsymbolW_0ZtohatZ with DboldsymbolW_0left(boldsymbol0right)=boldsymbol0, boldsymbolU^perpXtohatZ is a linear map, boldsymbolU^perpleft(boldsymbolU^perpright)^T=boldsymbolI and boldsymbolU^perpboldsymbolW_0left(boldsymbolzright)=boldsymbol0.\n\nFunction arguments arguments are\n\nmdim: dimensionality of the manifold\nndim: system dimensionality\nWorder: polynomial order of boldsymbolW_0\n\n\n\n\n\n","category":"type"},{"location":"localfoliation.html","page":"Locally accurate foliation","title":"Locally accurate foliation","text":"zero(M::ISFImmersionManifold)","category":"page"},{"location":"localfoliation.html#Base.zero-Tuple{ISFImmersionManifold}","page":"Locally accurate foliation","title":"Base.zero","text":"X = zero(M::ISFImmersionManifold)\n\nCreates a zero data structure for a local foliation ISFImmersionManifold.\n\n\n\n\n\n","category":"method"},{"location":"localfoliation.html","page":"Locally accurate foliation","title":"Locally accurate foliation","text":"ImmersionReconstruct","category":"page"},{"location":"localfoliation.html#FoliationsManifoldsAutoencoders.ImmersionReconstruct","page":"Locally accurate foliation","title":"FoliationsManifoldsAutoencoders.ImmersionReconstruct","text":"MWt, XWt = ImmersionReconstruct(Mimm, Ximm, Misf, Xisf, MU, XU)\n\nCreates a manifold immersion from the locally accurate foliation represented by Mimm, Ximm,  the full foliation represented by Misf, Xisf and the normal form transformation MU, XU.\n\n\n\n\n\n","category":"function"},{"location":"localfoliation.html","page":"Locally accurate foliation","title":"Locally accurate foliation","text":"ISFImmersionSolve!","category":"page"},{"location":"localfoliation.html#FoliationsManifoldsAutoencoders.ISFImmersionSolve!","page":"Locally accurate foliation","title":"FoliationsManifoldsAutoencoders.ISFImmersionSolve!","text":"Xres, dataParIN, dataParOUT = ISFImmersionSolve!(Mimm, Ximm, Misf, Xisf, Uout, Wperp, Sperp, dataIN, dataOUT; maxit = 25)\n\nSolves the optimisation problem\n\nargmin_boldsymbolSboldsymbolUsum_k=1^NleftVert boldsymbolx_krightVert ^-2expleft(-frac12kappa^2leftVert hatboldsymbolUleft(boldsymbolx_kright)rightVert ^2right)leftVert boldsymbolBhatboldsymbolUleft(boldsymbolx_kright)-hatboldsymbolUleft(boldsymboly_kright)rightVert ^2\n\n\n\n\n\n","category":"function"},{"location":"foliationidentify.html#Invariant-Foliations","page":"Sparse foliation","title":"Invariant Foliations","text":"","category":"section"},{"location":"foliationidentify.html","page":"Sparse foliation","title":"Sparse foliation","text":"Here we lay out the API for invariant foliations","category":"page"},{"location":"foliationidentify.html","page":"Sparse foliation","title":"Sparse foliation","text":"FoliationIdentify","category":"page"},{"location":"foliationidentify.html#FoliationsManifoldsAutoencoders.FoliationIdentify","page":"Sparse foliation","title":"FoliationsManifoldsAutoencoders.FoliationIdentify","text":"FoliationIdentify(dataIN, dataOUT, Tstep, \n                  embedscales, SysName, freq;\n                  orders = (P=9,Q=1,U=5,W=3), \n                  iterations = (f=4000, l=30), kappa = 0.2)\n\nThis is a convenience method that integrates all steps of calculating a reduced order model.\n\nEstimates the linear dynamics about the origin, only using part of the data.\nCalculates an invariant foliation about the invariant subspace corresponding, which has the closest frequancy to freq in Hz.\nPerforms a normal form transformation of the invariant foliation\nCalculates a locally accurate complementary invariant foliation, containing the invariant manifold that has the same dynamics as the foliation calculated in point 2.\nExtracts the invariant manifold from the locally accurate invariant foliation.\nPerforms a correction of the instantaneous frequency and damping values using the invariant manifold calculated in point 5.\n\nInput:\n\ndataIN is a two dimensional array, each column is an boldsymbolx_k value, \ndataOUT is a two dimensional array, each column is an boldsymboly_k\nTstep is the time step between boldsymbolx_k and boldsymboly_k\nembedscales, denoted by boldsymbolw is a matrix applied to boldsymbolx_k, boldsymboly_k, used to calculate the amplitude of a signal\nfreq is the frequency that the invariant foliation is calculated for.\norders is a named tuple, specifies the polynomial order of boldsymbolP, boldsymbolQ and boldsymbolU.\niterations is a named tuple, f is the maximum itearion when solving for a foliation,  l is the maximum number of iterations when solving for a locally accurate foliation.\nkappa is the size of the neighbourhood of a the invariant manifold considered, when calculating a locally accurate foliation.\n\nOutput: is a tuple with elements\n\nvector of instantaneous frequencies\nvector of instantaneous damping\nvector of instantaneous amplitudes\nthe scaling factor that was used to fit all data into the unit ball\nuncorrected vector of instantaneous frequencies\nuncorrected vector of instantaneous damping\nuncorrected vector of instantaneous amplitudes\nsomething mysterious\namplitude of each data point using left boldsymbolw boldsymbolUleft( boldsymbolW left( boldsymbolx_kright)right) right\n\n\n\n\n\n","category":"function"},{"location":"foliationidentify.html","page":"Sparse foliation","title":"Sparse foliation","text":"ISFPadeManifold","category":"page"},{"location":"foliationidentify.html#FoliationsManifoldsAutoencoders.ISFPadeManifold","page":"Sparse foliation","title":"FoliationsManifoldsAutoencoders.ISFPadeManifold","text":"ISFPadeManifold(mdim, ndim, Porder, Qorder, Uorder, B=nothing, field::AbstractNumbers=ℝ)\n\nReturns an ISFPadeManifold object, which provides the matrix manifold structure for an invariant foliation. The invariance equation, where these appear is\n\nboldsymbolPleft(boldsymbolUleft(boldsymbolxright)right) = boldsymbolQleft(boldsymbolUleft( boldsymbolF(boldsymbolx_k)right)right)\n\nwhere boldsymbolU is a polynomial with HT tensor coefficients, boldsymbolP is a general dense polynomial and boldsymbolQ is a near identity polynomial, that is D boldsymbolQ(boldsymbol0) = boldsymbolI. The purpose of polynomial boldsymbolQ is to have a Pade approximated nonlinear map, boldsymbolS = boldsymbolQ^-1 circ boldsymbolP. This can balance polynomials on both sides of the invariance equation. In practice, we did not find much use for it yet.\n\nThe parameters are\n\nmdim: co-dimension of the foliation\nndim: the dimesnion of the underlying phase space\nPorder: order of polynomial boldsymbolP\nQorder: order of polynomial boldsymbolQ\nUorder: order of polynomial boldsymbolU\nB: the matrix boldsymbolW_1, such that boldsymbolU (boldsymbolW_1 boldsymbolz) is constraing to be linear.\nfields: dummy, a standard parameter of Manifolds.jl\n\n\n\n\n\n","category":"type"},{"location":"foliationidentify.html","page":"Sparse foliation","title":"Sparse foliation","text":"zero(M::ISFPadeManifold)","category":"page"},{"location":"foliationidentify.html#Base.zero-Tuple{ISFPadeManifold}","page":"Sparse foliation","title":"Base.zero","text":"zero(M::ISFPadeManifold)\n\nCreates a zero ISFPadeManifold data representation.\n\n\n\n\n\n","category":"method"},{"location":"foliationidentify.html","page":"Sparse foliation","title":"Sparse foliation","text":"GaussSouthwellLinearSetup","category":"page"},{"location":"foliationidentify.html#FoliationsManifoldsAutoencoders.GaussSouthwellLinearSetup","page":"Sparse foliation","title":"FoliationsManifoldsAutoencoders.GaussSouthwellLinearSetup","text":"GaussSouthwellLinearSetup(Misf, Xisf, dataINorig, dataOUTorig, Tstep, nearest; perbox=2000, retbox=4, nbox=10, exclude = false)\n\nSets up the invariant foliation with linear estimates. The linear dynamics is estimated by the matrix boldsymbolA, the left invariant subspace is approximated by the orthogonal matrix boldsymbolU_1, the right invariant subspace is approximated by the orthogonal matrix boldsymbolW_1. The linearised dynamics is the matrix boldsymbolS_1, such that boldsymbolU_1 boldsymbolA=boldsymbolS_1 boldsymbolU_1 and boldsymbolA boldsymbolW_1=boldsymbolW_1 boldsymbolS_1.\n\nThe routine then sets D boldsymbolU (0) = boldsymbolU_1 and D boldsymbolP (0) = boldsymbolS_1. It also sets the constrint that boldsymbolU (boldsymbolW_1 boldsymbolz) is linear.\n\n\n\n\n\n","category":"function"},{"location":"foliationidentify.html","page":"Sparse foliation","title":"Sparse foliation","text":"GaussSouthwellOptim","category":"page"},{"location":"foliationidentify.html#FoliationsManifoldsAutoencoders.GaussSouthwellOptim","page":"Sparse foliation","title":"FoliationsManifoldsAutoencoders.GaussSouthwellOptim","text":"GaussSouthwellOptim(Misf, Xisf, dataIN, dataOUT, scale, Tstep, nearest; name = \"\", maxit=8000, gradstop = 1e-10)\n\nSolves the optimisation problem\n\nargmin_boldsymbolSboldsymbolUsum_k=1^NleftVert boldsymbolx_krightVert ^-2leftVert boldsymbolPleft(boldsymbolUleft(boldsymbolx_kright)right)-boldsymbolQleft(boldsymbolUleft(boldsymboly_kright)right)rightVert ^2\n\nThe method is block coordinate descent, where we optimise for matrix coefficients of the representation in a cyclic manner and also by choosing the coefficient whose gradient is the gratest for optimisation.\n\n\n\n\n\n","category":"function"},{"location":"foliationidentify.html","page":"Sparse foliation","title":"Sparse foliation","text":"ISFNormalForm","category":"page"},{"location":"foliationidentify.html#FoliationsManifoldsAutoencoders.ISFNormalForm","page":"Sparse foliation","title":"FoliationsManifoldsAutoencoders.ISFNormalForm","text":"Wout, Rout, PW, PR = ISFNormalForm(M, X)\n\nCalculates the normal form of the polynomial map represented by M, X.  It returns the real normal form Wout, Rout  and the complex normal form PW, PR\n\n\n\n\n\n","category":"function"},{"location":"autoencoders.html#Autoencoders","page":"Autoencoders","title":"Autoencoders","text":"","category":"section"},{"location":"autoencoders.html","page":"Autoencoders","title":"Autoencoders","text":"We use the autoencoder introduced in [1]. We assume an orthogonal matrix boldsymbolUinmathbbR^ntimesnu (boldsymbolU^TboldsymbolU=boldsymbolI), a polynomial boldsymbolWmathbbR^nuto mathbbR^n that starts with quadratic terms up to order d. The encoder is the linear map boldsymbolU^T and the decoder is","category":"page"},{"location":"autoencoders.html","page":"Autoencoders","title":"Autoencoders","text":"boldsymbolWleft(boldsymbolzright)=boldsymbolUboldsymbolz+boldsymbolWleft(boldsymbolzright)","category":"page"},{"location":"autoencoders.html","page":"Autoencoders","title":"Autoencoders","text":"To find the autoencoder we first solve","category":"page"},{"location":"autoencoders.html","page":"Autoencoders","title":"Autoencoders","text":"argmin_boldsymbolUboldsymbolWsum_k=1^NleftVert boldsymboly_krightVert ^-2leftVert boldsymbolWleft(boldsymbolUleft(boldsymboly_kright)right)-boldsymboly_krightVert ^2","category":"page"},{"location":"autoencoders.html","page":"Autoencoders","title":"Autoencoders","text":"then solve","category":"page"},{"location":"autoencoders.html","page":"Autoencoders","title":"Autoencoders","text":"argmin_boldsymbolSsum_k=1^NleftVert boldsymbolx_krightVert ^-2leftVert boldsymbolWleft(boldsymbolSleft(boldsymbolUleft(boldsymbolx_kright)right)right)-boldsymboly_krightVert ^2","category":"page"},{"location":"autoencoders.html","page":"Autoencoders","title":"Autoencoders","text":"The methods are","category":"page"},{"location":"autoencoders.html","page":"Autoencoders","title":"Autoencoders","text":"AENCManifold","category":"page"},{"location":"autoencoders.html#FoliationsManifoldsAutoencoders.AENCManifold","page":"Autoencoders","title":"FoliationsManifoldsAutoencoders.AENCManifold","text":"M = AENCManifold(ndim, mdim, Sorder, Worder, orthogonal = true, field::AbstractNumbers=ℝ)\n\nCreates an autoencoder as a matrix manifold.\n\nThe parameters are\n\nndim the dimansionality of the problem boldsymbolF\nmdim dimensionality of the low-dimensional map boldsymbolS\nSorder polynomial order of map boldsymbolS\nWorder polynomial order of decoder boldsymbolW\northogonal whether DboldsymbolW(0) is orthogonal to boldsymbolU. When the data is on a manifold true is the good answer.\n\n\n\n\n\n","category":"type"},{"location":"autoencoders.html","page":"Autoencoders","title":"Autoencoders","text":"zero(M::AENCManifold)","category":"page"},{"location":"autoencoders.html#Base.zero-Tuple{AENCManifold}","page":"Autoencoders","title":"Base.zero","text":"X = zero(M::AENCManifold)\n\nCreates a zero valued representation of an autoencoder.\n\n\n\n\n\n","category":"method"},{"location":"autoencoders.html","page":"Autoencoders","title":"Autoencoders","text":"AENCIndentify","category":"page"},{"location":"autoencoders.html#FoliationsManifoldsAutoencoders.AENCIndentify","page":"Autoencoders","title":"FoliationsManifoldsAutoencoders.AENCIndentify","text":"AENCIndentify(dataIN, dataOUT, Tstep, embedscales, freq, orders = (S=7,W=7))\n\nInput:\n\ndataIN is a two dimensional array, each column is an boldsymbolx_k value, \ndataOUT is a two dimensional array, each column is an boldsymboly_k\nTstep is the time step between boldsymbolx_k and boldsymboly_k\nembedscales, denoted by boldsymbolw is a matrix applied to boldsymbolx_k, boldsymboly_k, used to calculate the amplitude of a signal\nfreq is the frequency that the invariant foliation is calculated for.\norders is a named tuple, specifies the polynomial order of boldsymbolS and boldsymbolW.\n\nOutput is a tuple with elements\n\nvector of instantaneous frequencies\nvector of instantaneous damping\nvector of instantaneous amplitudes\nthe scaling factor that was used to fit all data into the unit ball\nuncorrected vector of instantaneous frequencies\nuncorrected vector of instantaneous damping\nuncorrected vector of instantaneous amplitudes\n\n\n\n\n\n","category":"function"},{"location":"autoencoders.html","page":"Autoencoders","title":"Autoencoders","text":"[1]: M. Cenedese, J. Axås, B. Bäuerlein, K. Avila, and G. Haller. Data-driven modeling and prediction of non-linearizable dynamics via spectral submanifolds. Nat Commun, 13(872), 2022.","category":"page"},{"location":"polymethods.html#Polynomials","page":"Dense polynomials","title":"Polynomials","text":"","category":"section"},{"location":"polymethods.html","page":"Dense polynomials","title":"Dense polynomials","text":"DensePolyManifold","category":"page"},{"location":"polymethods.html#FoliationsManifoldsAutoencoders.DensePolyManifold","page":"Dense polynomials","title":"FoliationsManifoldsAutoencoders.DensePolyManifold","text":"M = DensePolyManifold(ndim, n, order; min_order = 0, identity = false, field::AbstractNumbers=ℝ)\n\nCreates a manifold structure of a dense polynomial with ndim input variables and n output dimension of maximum order order.  One can set the smallest represented order by min_order. The parameter field can be set to either real ℝ or complex ℂ.\n\n\n\n\n\n","category":"type"},{"location":"polymethods.html","page":"Dense polynomials","title":"Dense polynomials","text":"zero(M::DensePolyManifold)","category":"page"},{"location":"polymethods.html#Base.zero-Tuple{DensePolyManifold}","page":"Dense polynomials","title":"Base.zero","text":"X = zero(M::DensePolyManifold)\n\nCreate a representation of a zero polynomial with manifold structure M.\n\n\n\n\n\n","category":"method"},{"location":"polymethods.html","page":"Dense polynomials","title":"Dense polynomials","text":"fromFunction","category":"page"},{"location":"polymethods.html#FoliationsManifoldsAutoencoders.fromFunction","page":"Dense polynomials","title":"FoliationsManifoldsAutoencoders.fromFunction","text":"X = fromFunction(M::DensePolyManifold, fun)\n\nTaylor expands Julia function fun to a polynomial, whose strcuture is prescribed by M.\n\n\n\n\n\n","category":"function"},{"location":"foliationexample.html#An-example-for-calculating-an-invariant-foliation","page":"Example","title":"An example for calculating an invariant foliation","text":"","category":"section"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"The following code calculates an invariant foliation of a 4-dimensional map and compares it to the analytic calculation by iManifoldMAP.","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"First we bring the required packeges into scope.","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"using FoliationsManifoldsAutoencoders\nusing Manifolds\nusing Plots","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"Then we define a discrete-time map, which we use to calculate invariant foliations and invariant manifolds for.","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"function trivial(z)\n    om1 = 0.12\n    om2 = 1\n    dm1 = 0.002\n    dm2 = 0.003\n    return [(exp(dm1)*z[4]*(z[1]^3*z[2] + z[4]^2 + z[4]^4) + 2*z[1]*cos(om1) - 2*z[2]*sin(om1))/(2.0*exp(dm1)),\n            -0.5*z[2]^3 - z[1]^2*z[2]^2*z[4] - (3*z[2]*z[3]*z[4])/4. + (z[2]*cos(om1) + z[1]*sin(om1))/exp(dm1),\n            (-3*z[1]*z[2]^3*z[3])/4.0 + z[3]^5 - (3*z[1]^3*z[2]*z[4])/4.0 + (z[3]*cos(om2) - z[4]*sin(om2))/exp(dm2),\n            (z[2]*z[3]^4)/4.0 + (z[2]*z[3]^3*z[4])/2.0 - z[4]^5 + (z[4]*cos(om2) + z[3]*sin(om2))/exp(dm2)]\nend","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"We define a function that creates the required data. This function iterates trivial with random initial conditions.","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"function generate(amp = 1.0)\n    ndim = 4\n    nruns = 16000\n    npoints = 1\n    xs = zeros(ndim, nruns*npoints)\n    ys = zeros(ndim, nruns*npoints)\n    aa = rand(ndim,nruns) .- ones(ndim,nruns)/2\n    ics =  aa ./ sqrt.(sum(aa.^2,dims=1)) .* (2*rand(1,nruns) .- 1)\n    \n    for j=1:nruns\n        u0 = ics[:,j] * amp\n        for k=1:npoints\n            u1 = trivial(u0)\n            xs[:,k+(j-1)*npoints] .= u0\n            ys[:,k+(j-1)*npoints] .= u1\n            u0 .= u1\n        end\n    end\n    return xs, ys\nend","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"We convert map trivial into a polynomial form.","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"MF = DensePolyManifold(4, 4, 5)\nXF = fromFunction(MF, trivial)","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"We calculate the invariant manifold corresponding to the 3rd and 4th eigenvalues of the Jacobian of trivial, which form a complex conjugate pair.","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"MWt, XWt, MS, XS = iManifoldMAP(MF, XF, [3, 4], [])","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"We calculate the corrected instantaneous frequencies and damping ratios for the invariant manifold, we have just calculated.","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"Dr = 0.0001\nr = range(0,1,step=Dr)\nopscal = ones(1,4)/4\nfrequency, damping, amplitude = MAPManifoldFrequencyDamping(MWt, XWt, MS, XS, r, 1.0; output = opscal)","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"We generate data.","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"# create data\ndataIN, dataOUT = generate(0.5)","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"We identify the invariant foliation, locally invriant foliation, extract the invariant manifold and calculate instantaneous frequencies, damping ratios. This is near the natural frequency 012, if the time step is assumed to be 10.","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"# for which frequency is the foliation calculated for\nfreq = 0.12/2/pi\n# we did not specify sampling frequency, so we keep it as unit time-step\nTstep = 1.0\n\nfrequencyD, dampingD, amplitudeD = FoliationIdentify(dataIN, dataOUT, Tstep, opscal, \"trivial\", freq; orders = (P=7,Q=1,U=5,W=5), iterations = (f=200, l=20))","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"Finally we plot the result.","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"pl = plot([frequency[2:end], frequencyD[2:end]], [amplitude[2:end], amplitudeD[2:end]], xlims=[0.11, 0.125], ylims=[0, 0.15], xlabel=\"frequency [rad/s]\", ylabel=\"amplitude\", label=[\"MAP\" \"DATA\"])\ndisplay(pl)","category":"page"},{"location":"foliationexample.html","page":"Example","title":"Example","text":"note: Note\nThe number of iterations are not high enough the produce accurate results, they are set that the example runs quickly enough. Similarly the maximum amplitude within the generated data is quite low, which influences accuracy at high amplitudes.","category":"page"},{"location":"index.html#Invariant-Foliations,-Manifolds-and-Autoencoders","page":"Home","title":"Invariant Foliations, Manifolds and Autoencoders","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"A Julia package to demonstrate various techniques for data oriented reduced order modelling","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"warning: Warning\nThis package and its documentation is experimental. The code includes constants that should be configurable, function names require re-thinking. Generally there is still a lot of work to be done.  The package was being made available so that the results in paper [paper] can be reproduced.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"This package collects various computational methods employed in paper [paper]. ","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"The package also includes methods to calculate invariant foliations and invariant manifolds from low-dimenional vector fields as assymptotic polynomials.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"We assume that the data is produced by an underlying map boldsymbolF, which maps from vector space X to itself. Our data comes in pairs","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"left(boldsymboly_k boldsymbolx_kright)k=1ldotsN","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"where","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"boldsymboly_k=boldsymbolFleft(boldsymbolx_kright)+boldsymbolxi_k","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"and boldsymbolxi_kin X is a small noise sampled from a distribution with zero mean. Further, we assume that boldsymbolFleft(boldsymbol0right)=boldsymbol0.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"There are four ways to connect a low-order model boldsymbolS to boldsymbolF. The figure below shows the four combinations. Only invariant foliations and invariant manifolds produce meaningful reduced order models. Only invariant foliations and autoencoders can be fitted to data. The intersection is the invariant foliation.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"(Image: alternative text)","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"An invariant foliation is fitted to data by solving","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"argmin_boldsymbolSboldsymbolUsum_k=1^NleftVert boldsymbolx_krightVert ^-2leftVert boldsymbolSleft(boldsymbolUleft(boldsymbolx_kright)right)-boldsymbolUleft(boldsymboly_kright)rightVert ^2","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"where boldsymbolU is an encoder mapping from X to a lower dimensional space Z and boldsymbolSZto Z is a low-dimensional map.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"An autoencoder is fitted to data by solving","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"argmin_boldsymbolSboldsymbolUboldsymbolWsum_k=1^NleftVert boldsymbolx_krightVert ^-2leftVert boldsymbolWleft(boldsymbolSleft(boldsymbolUleft(boldsymbolx_kright)right)right)-boldsymboly_krightVert ^2","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"where boldsymbolUXto Z is the encoder, boldsymbolWZto X is the decoder and boldsymbolSZto Z is a low-dimensional map.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"[paper]: R. Szalai, Data-driven reduced order models using invariant foliations, manifolds and auto-encoders, 2022, preprint","category":"page"}]
}
