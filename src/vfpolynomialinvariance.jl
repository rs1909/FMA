
# -------------------------------------------------------
# THE SSM FUNCTIONS
# -------------------------------------------------------

# calculates the adjoint spectral submanifold for a map given by F0
# the eigenvalues are selected by 'intvar', hence many dimensions can be selected
# takes into account the near internal resonances
@doc raw"""
    MUr, XUr, MSr, XSr, MU, XU, MS, XS = iFoliationVF(MF::DensePolyManifold, XF, vars, par; order = PolyOrder(MF))
    
Calculates the invariant foliation of nonlinear vector field ``\boldsymbol{F}`` with respect to
selected eigenvectors of the Jacobian ``D\boldsymbol{F}(\boldsymbol{0})``.

We solve the invariance equation
```math
D \boldsymbol{U} \boldsymbol{F} = \boldsymbol{S} \circ \boldsymbol{U}
```
for ``\boldsymbol{U}`` and ``\boldsymbol{S}`` using polynomial expansion. 

The input are 
  * `MF`, `XF` represent the dense polynomial expansion of function `\boldsymbol{F}``. 
  * `vars` is a vector of integers, selecting the eigenvectors of ``D\boldsymbol{F}(\boldsymbol{0})``.
  * `par` are the indices of input variable of ``\boldsymbol{F}``, which do not have dynamics and can be treated as parameters.
  * `order` is the polynomial order for which the calculation is carried out. The default is the same order as the order of `MF`, `XF`, but it can be greater for increased accuracy.
  
The output is
  * `MUr`, `XUr` represent ``\boldsymbol{U}`` in real coordinates
  * `MSr`, `XSr` represent ``\boldsymbol{S}`` in real coordinates
  * `MU`, `XU` represent ``\boldsymbol{U}`` in complex coordinates
  * `MS`, `XS` represent ``\boldsymbol{S}`` in complex coordinates
  
Note that it calculates the complex result first, which has the minimal number of parameters, then it transforms the result into real form.
"""
function iFoliationVF(MF0::DensePolyManifold, XF0, vars, par; order = PolyOrder(MF0))
    intvar = [vars; par]
    ndim = size(XF0, 1)
    F01 = getLinearPart(MF0, XF0)
    
    eigval, eigvec = eigen(F01)
    # rescale the parameter so that it remains unity
    if !isempty(par)
        eigvec[:,par] /= eigvec[par,par]
    end

    # makes the linear part diagonal, so that the rest of the calculation is easy
    MF, XF0c = toComplex(MF0, XF0)
    XF = zero(MF)
    DensePolyLinearTransform!(MF, XF, MF, XF0c, Complex.(eigvec))

    # define the indices of external variables
    extvar = setdiff(1:ndim, intvar)
    
    # it can possibly be done for higher dimansional SSMs as well
    zdim = length(intvar)
    
    # from here the operations are scalar, because of the diagonal matrices
    F1 = Diagonal(eigval)
    R1 = Diagonal(eigval[intvar])
    W1 = zeros(eltype(XF), zdim, ndim)
    W1[:, intvar] = one(R1)
    
    # put it into the polynomials
    modelorder = order
    
    MW = DensePolyManifold(ndim, zdim, modelorder; field = ManifoldsBase.ComplexNumbers())
    XW = zero(MW)
    @show size(XW), size(W1)
    MR = DensePolyManifold(zdim, zdim, modelorder; field = ManifoldsBase.ComplexNumbers())
    XR = zero(MR)
    @show size(XR), size(R1)
    # set the linear parts R1 -> R, W1 -> W
    setLinearPart!(MR, XR, R1)
    setLinearPart!(MW, XW, W1)

    # @time multabDWF = mulTable(W.mexp, W.mexp, W.mexp)
    # mulTable(out.mexp, out.mexp, in2.mexp)
    deritabDW = DensePolyDeriMatrices(MW.mexpALL, MW.mexpALL)
    multabDWoF = DensePolySubstituteTab!(MW, XW, MW, XW, MF, XF)
    multabRoW = DensePolySubstituteTab!(MW, XW, MR, XR, MW, XW)

    # recursively do the transformation by order
    for ord = 2:PolyOrder(MW)
        id = PolyOrderIndices(MW, ord)

        # the inhomogeneity: B = DW \dot F - R \circ W
        res0 = zero(MW)
        DensePolyDeriMul!(MW, res0, MW, XW, MF, XF, multabDWoF, deritabDW) # res0 = DW . F
        res1 = zero(MW)
        DensePolySubstitute!(MW, res1, MR, XR, MW, XW, multabRoW) # res1 = R o W
        B = res0 - res1
        
        # calculate for each monomial
        for x in id
            # the order of external variables
            extord = sum(MW.mexp[extvar,x])
            if extord == 0
                # rx is the index of this monomial 'x' in R
                rx = PolyFindIndex(MR.mexp, MW.mexp[intvar,x]) # this is calculated only once per monomial, hence no need to optimise out
            else
                rx = 0
            end   
            # now calculate for each dimension, which is the number of interval variables
            for j=1:length(intvar)
                # k is the index of the internal variable (we need double indexing)
                k = intvar[j]
                # if there are no external variables involved, we can take the resonances into account
                if extord == 0
                    # internal monomials
                    # SOLVE: sum(eigval.*W.mexp[:,x]) * W.W[k,x] - eigval[k] * W.W[k,x] = R.W[j,rx] - B[k,x]
                    den = sum(eigval.*MW.mexp[:,x]) - eigval[k]
                    # the threshold of 0.1 is arbitrary
                    # we probably should use fixed near resonances
                    if abs(den) < 0.1
                        XR[j,rx] = B[j,x]
                        XW[j,x] = 0
                        # println("Internal resonance: ", abs(den), " at dim=", k, " exp=", W.mexp[:,x])
                    else
                        XR[j,rx] = 0
                        XW[j,x] = -B[j,x]/den
                    end
                # here, external variables are involved, we cannot take the resonances into account
                else
                    # external and mixed monomials
                    # SOLVE: prod(eigval.^W.mexp[:,x])*W.W[k,x] - eigval[k]*W.W[k,x] = B[k,x]
                    den = sum(eigval.*MW.mexp[:,x]) - eigval[k]
                    if abs(den) > 1e-12
                        XW[j,x] = -B[j,x]/den
                    else
                        println("Fatal cross resonance, not calculating term: ", abs(den), " at dim=", k, " exp=", MW.mexp[:,x])
                        XW[j,x] = 0.0
                    end
                    if abs(den) < 0.1
                        println("Warning: near cross resonance: ", abs(den), " at dim=", k, " exp=", MW.mexp[:,x])
                    end
                end
            end
        end
    end
    # transform result back
    # transform result back
    MWr, XWr, MRr, XRr = iFoliationTransform(MW, XW, MR, XR, eigvec)

    res0 = zero(MWr)
    DensePolyDeriMul!(MWr, res0, MWr, XWr, MF0, XF0) # res0 = DW . F
    res1 = zero(MWr)
    DensePolySubstitute!(MWr, res1, MRr, XRr, MWr, XWr) # res1 = R o W
    B = res0 - res1
    if maximum(abs.(B)) > 1e-10
        println("High error in ISF calculation (REAL): ", maximum(abs.(B)))
    end

    res0 = zero(MW)
    DensePolyDeriMul!(MW, res0, MW, XW, MF, XF, multabDWoF, deritabDW) # res0 = DW . F
    res1 = zero(MW)
    DensePolySubstitute!(MW, res1, MR, XR, MW, XW, multabRoW) # res1 = R o W
    B = res0 - res1
    if maximum(abs.(B)) > 1e-10
        println("High error in iFoliationMAP calculation (CPLX): ", maximum(abs.(B)))
        @show MW.mexp[:,findall((abs.(B) .> 1e-12)[1,:])]
        @show abs.(B[1,findall((abs.(B) .> 1e-12)[1,:])])
    end
    
    return MWr, XWr, MRr, XRr, MW, XW, MR, XR
end

# calculates the spectral submanifold for a map given by F0
# the eigenvalues are selected by 'intvar', hence many dimensions can be selected
# takes into account the near internal resonances
@doc raw"""
    MWr, XWr, MRr, XRr, MW, XW, MR, XR = iManifoldVF(MF0::DensePolyManifold, XF0, vars, par; order = PolyOrder(MF0))
    
Calculates the invariant manifold of nonlinear vector field ``\boldsymbol{F}`` with respect to
selected eigenvectors of the Jacobian ``D\boldsymbol{F}(\boldsymbol{0})``.

We solve the invariance equation
```math
D \boldsymbol{W} \boldsymbol{R} = \boldsymbol{F} \circ \boldsymbol{W}
```
for ``\boldsymbol{U}`` and ``\boldsymbol{S}`` using polynomial expansion. 

The input are 
  * `MF`, `XF` represent the dense polynomial expansion of function `\boldsymbol{F}``. 
  * `vars` is a vector of integers, selecting the eigenvectors of ``D\boldsymbol{F}(\boldsymbol{0})``.
  * `par` are the indices of input variable of ``\boldsymbol{F}``, which do not have dynamics and can be treated as parameters.
  * `order` is the polynomial order for which the calculation is carried out. The default is the same order as the order of `MF`, `XF`, but it can be greater for increased accuracy.
  
The output is
  * `MWr`, `XUr` represent ``\boldsymbol{W}`` in real coordinates
  * `MWr`, `XSr` represent ``\boldsymbol{R}`` in real coordinates
  * `MW`, `XW` represent ``\boldsymbol{W}`` in complex coordinates
  * `MR`, `XR` represent ``\boldsymbol{R}`` in complex coordinates
  
Note that it calculates the complex result first, which has the minimal number of parameters, then it transforms the result into real form.
"""
function iManifoldVF(MF0::DensePolyManifold, XF0, vars, par; order = PolyOrder(MF0))
    intvar = [vars; par]
    ndim = size(XF0, 1)
    F01 = getLinearPart(MF0, XF0)
    
    eigval, eigvec = eigen(F01)
    # rescale the parameter so that it remains unity
    if !isempty(par)
        eigvec[:,par] /= eigvec[par,par]
    end

    # makes the linear part diagonal, so that the rest of the calculation is easy
    MF, XF0c = toComplex(MF0, XF0)
    XF = zero(MF)
    DensePolyLinearTransform!(MF, XF, MF, XF0c, Complex.(eigvec))

    # define the indices of external variables
    extvar = setdiff(1:ndim, intvar)
    
    # it can possibly be done for higher dimensional SSMs as well
    zdim = length(intvar)

    # from here the operations are scalar, because of the diagonal matrices
    F1 = Diagonal(eigval)
    R1 = Diagonal(eigval[intvar])
    W1 = zeros(eltype(XF), ndim, zdim)
    W1[intvar, :] = one(R1)
    
    # put it into the polynomials
    modelorder = order
    
    MW = DensePolyManifold(zdim, ndim, modelorder; field = ManifoldsBase.ComplexNumbers())
    XW = zero(MW)
    @show size(XW), size(W1)
    MR = DensePolyManifold(zdim, zdim, modelorder; field = ManifoldsBase.ComplexNumbers())
    XR = zero(MR)
    @show size(XR), size(R1)
    # set the linear parts R1 -> R, W1 -> W
    setLinearPart!(MR, XR, R1)
    setLinearPart!(MW, XW, W1)

    # DW R - F o W
    deritabDW = DensePolyDeriMatrices(MW.mexpALL, MW.mexpALL)
    multabDWoR = DensePolySubstituteTab!(MW, XW, MW, XW, MR, XR)
    multabFoW = DensePolySubstituteTab!(MW, XW, MF, XF, MW, XW)

    # recursively do the transformation by order
    for ord = 2:PolyOrder(MW)
        id = PolyOrderIndices(MW, ord)

        # the inhomogeneity: B = DW \dot R - F \circ W
        res0 = zero(MW)
        DensePolyDeriMul!(MW, res0, MW, XW, MR, XR, multabDWoR, deritabDW) # res0 = DW . R
        res1 = zero(MW)
        DensePolySubstitute!(MW, res1, MF, XF, MW, XW, multabFoW) # res1 = F o W
        B = res0 - res1
        
        for x in id
            # going through the internal diemsions
            for j=1:length(intvar)
                k = intvar[j]
                # SOLVE: eigval[k]*W.W[k,x] - prod(eigval.^W.mexp[:,id])*W.W[k,x] = R.W[sel,x] + B[k,x]
                den = eigval[k] - sum(eigval[intvar].*MW.mexp[:,x])
                if abs(den) < 0.2
                    XR[j,x] = -B[k,x]
                    XW[k,x] = 0
                    println("Internal resonance: ", abs(den), " at dim=", k, " exp=", MW.mexp[:,x])
                else
                    XR[j,x] = 0
                    XW[k,x] = B[k,x]/den
                end
            end
            # going through the external dimensions
            for k in extvar
                # SOLVE: eigval[k]*W.W[k,x] - prod(eigval.^W.mexp[:,id])*W.W[k,x] = B[k,x]
                den = eigval[k] - sum(eigval[intvar].*MW.mexp[:,x])
                XW[k,x] = B[k,x]/den
                if abs(den) < 0.1
                    println("Warning: near cross resonance: ", abs(den), " at dim=", k, " exp=", MW.mexp[:,x])
                end
            end
        end
    end
    # back to the original coordinate system
    XWc = Complex.(eigvec) * XW
    # Check the result
    res0 = zero(MW)
    DensePolyDeriMul!(MW, res0, MW, XWc, MR, XR, multabDWoR, deritabDW) # res0 = DW . R
    res1 = zero(MW)
    DensePolySubstitute!(MW, res1, MF, XF0c, MW, XWc, multabFoW) # res1 = F o W
    B = res0 - res1
    if maximum(abs.(B)) > 1e-10
        println("High error in iManifoldVF calculation (CPLX): ", maximum(abs.(B)))
    end

    MWr, XWr, MRr, XRr = iManifoldTransform(MW, XW, MR, XR, eigvec)
    # Check the transformed result
    res0 = zero(MWr)
    DensePolyDeriMul!(MWr, res0, MWr, XWr, MRr, XRr) # res0 = DW . R
    res1 = zero(MWr)
    DensePolySubstitute!(MWr, res1, MF0, XF0, MWr, XWr) # res1 = F o W
    B = res0 - res1
    if maximum(abs.(B)) > 1e-10
        println("High error in iManifoldVF calculation (REAL): ", maximum(abs.(B)))
    end

    return MWr, XWr, MRr, XRr, MW, XWc, MR, XR
end
