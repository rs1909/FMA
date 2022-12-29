
function ManifoldAmplitudeSquare(MWt, XWt)
    din = size(XWt,1)
    dout = size(MWt.mexp,1)
    order = PolyOrder(MWt)

    function CosP_SinQ(p,q)
        if iseven(p)*iseven(q)
            return gamma((p+1.0)/2)*gamma((q+1.0)/2) / factorial(div(p+q,2)) / pi / 2 
        else
            return 0.0
        end
    end
    # A00
    #   W^T . W
    #   = r^(2 p + 2 q) [ a1^2 * Cos^p * Sin^p
    #               +a2 * Cos^(p+1) * Sin^q
    M_WtWt = DensePolyManifold(dout, 1, 2*order)
    X_WtWt = zero(M_WtWt)
    DensePolySquared!(M_WtWt, X_WtWt, MWt, XWt)
    
    M_A = DensePolyManifold(1, 1, 4*order)
    X_A00 = zero(M_A)

    for k=1:size(M_WtWt.mexp,2)
        p = M_WtWt.mexp[1,k]
        q = M_WtWt.mexp[2,k]
        # r^{p+q} a Cos^p * Sin^q
        res = X_WtWt[1,k] * CosP_SinQ(p,q)
        ro = p+q
        roid = findfirst(isequal(ro), M_A.mexp[1,:])
        X_A00[roid] += res
    end
    return M_A, X_A00
end

function ManifoldGeometry(MWt::DensePolyManifold, XWt)
    #---------------------------------------------------------
    #
    # Calculate the geometry on the manifold
    #
    #---------------------------------------------------------
    # In 2 dims
    #   z = [ r Cos(t), r Sin(t) ]
    #   D1 W(r, t) = DW( T(r,t) ) [ Cos(t), Sin(t) ]
    #   D2 W(r, t) = DW( T(r,t) ) [ -r Sin(t), r Cos(t) ]
    # We need
    #   A12 = Int < D1 W(r,t), D2 W(r,t) > dt / (2 Pi) = r Int [ Cos(t), Sin(t) ] . DW^T * DW . [ -Sin(t), Cos(t) ] dt
    #   A22 = Int < D2 W(r,t), D2 W(r,t) > dt / (2 Pi) = r^2 Int [ -Sin(t), Cos(t) ] . DW^T * DW . [ -Sin(t), Cos(t) ] dt
    #   A10 = Int < D1 W(r,t),    W(r,t) > dt / (2 Pi) = Int [ Cos(t), Sin(t) ] . DW^T . W dt
    #   A20 = Int < D2 W(r,t),    W(r,t) > dt / (2 Pi) = r Int [ -Sin(t), Cos(t) ] . DW^T . W dt
    # Let 
    #   DW^T * DW = ( a11 a12 )
    #               ( a21 a22 )
    #   with r^(p+q) Cos^p * Sin^q coefficient, i.e., -> (p,q)
    # Then
    # 1. A12
    #   [ Cos(t), Sin(t) ] . DW^T * DW . [ -Sin(t), Cos(t) ] = 
    #   = r^(p+q+1) [ (a22 - a11) Cos^(p+1) * Sin^(q+1)
    #              + a12 * Cos^(p+2) * Sin^(q)
    #              - a21 * Cos^(p) * Sin^(q+2) ]
    # 2. A22
    #   [ -Sin(t), Cos(t) ] . DW^T * DW . [ -Sin(t), Cos(t) ] = 
    #   = r^(p+q+2) [ -(a12 + a21) Cos^(p+1) * Sin^(q+1)
    #              + a22 * Cos^(p+2) * Sin^(q)
    #              + a11 * Cos^(p) * Sin^(q+2) ]
    # Let 
    #   DW^T * W = ( a1 a2 )
    #   with r^(p+q) Cos^p * Sin^q coefficient, i.e., -> (p,q)
    # 3. A10
    #   [ Cos(t), Sin(t) ] . DW^T . W
    #   = r^(p+q) [a1 * Cos^(p+1) * Sin^q
    #             +a2 * Cos^p * Sin^(q+1)
    # 4. A20
    #   [ -Sin(t), Cos(t) ] . DW^T . W
    #   = r^(p+q) [ -a1 * Cos^p * Sin^(q+1)
    #               +a2 * Cos^(p+1) * Sin^q
    # For the integrals, we need to use the Gamma special function an factorial
    #   Int Cos^{2n}(t) Sin^{2m}(t) dt = Gamma(n+1/2) Gamma(m+1/2) / pi / (m+n)!
    # Solve
    #   delta' = - A12/A22
    #   kappa' = A10 + A20 * delta'
    #---------------------------------------------------------
    din = size(XWt,1)
    dout = size(MWt.mexp,1)
    order = PolyOrder(MWt)
#     @show din, dout
    function CosP_SinQ(p,q)
        if iseven(p)*iseven(q)
            return gamma((p+1.0)/2)*gamma((q+1.0)/2) / factorial(div(p+q,2)) / pi / 2 
        else
            return 0.0
        end
    end
    # DW^T . W
    M_DWtr_W = DensePolyManifold(dout, dout, 2*order)
    X_DWtr_W = zero(M_DWtr_W)
    DensePolyDeriTransposeMul!(M_DWtr_W, X_DWtr_W, MWt, XWt)
    
    # DW^T . DW
    M_DWtr_DW = DensePolyManifold(dout, dout, 2*order)
    X_DWtr_DW = zeroJacobianSquared(M_DWtr_DW)
    DensePolyJabobianSquared!(M_DWtr_DW, X_DWtr_DW, MWt, XWt)
    
    M_A = DensePolyManifold(1, 1, 4*order)
    X_A12 = zero(M_A)
    X_A22 = zero(M_A)
    X_A10 = zero(M_A)
    X_A20 = zero(M_A)
    
    for k=1:size(M_DWtr_DW.mexp,2)
        p = M_DWtr_DW.mexp[1,k]
        q = M_DWtr_DW.mexp[2,k]
        # A12
        a11 = X_DWtr_DW[1,1,k]
        a12 = X_DWtr_DW[1,2,k]
        a21 = X_DWtr_DW[2,1,k]
        a22 = X_DWtr_DW[2,2,k]
        res = (a22 - a11) * CosP_SinQ(p+1,q+1) + a12 * CosP_SinQ(p+2,q) - a21 * CosP_SinQ(p,q+2)
        ro = p+q+1
        roid = findfirst(isequal(ro), M_A.mexp[1,:])
        X_A12[roid] += res

        # A22
        res = -(a12 + a21) * CosP_SinQ(p+1,q+1) + a22 * CosP_SinQ(p+2,q) + a11 * CosP_SinQ(p,q+2)
        ro = p+q+2
        roid = findfirst(isequal(ro), M_A.mexp[1,:])
        X_A22[roid] += res
        
        p = M_DWtr_W.mexp[1,k]
        q = M_DWtr_W.mexp[2,k]
        # A10
        a1 = X_DWtr_W[1,k]
        a2 = X_DWtr_W[2,k]
        res = a1 * CosP_SinQ(p+1,q) + a2 * CosP_SinQ(p,q+1)
        ro = p+q
        roid = findfirst(isequal(ro), M_A.mexp[1,:])
        X_A10[roid] += res

        # A20
        res = -a1 * CosP_SinQ(p,q+1) + a2 * CosP_SinQ(p+1,q)
        ro = p+q+1
        roid = findfirst(isequal(ro), M_A.mexp[1,:])
        X_A20[roid] += res
    end

    # normalise A12, A22, divide both by r^2
    Wn = zero(X_A12)
    for k=1:size(M_A.mexp,2)
        id = findfirst(isequal(M_A.mexp[1,k]-2), M_A.mexp[1,:])
        if id != nothing
            Wn[1,id] = X_A12[1,k]
        end
    end
    X_A12 .= Wn
    #
    Wn = zero(X_A22)
    for k=1:size(M_A.mexp,2)
        id = findfirst(isequal(M_A.mexp[1,k]-2), M_A.mexp[1,:])
        if id != nothing
            Wn[1,id] = X_A22[1,k]
        end
    end
    X_A22 .= Wn
    return M_A, X_A12, X_A22, X_A10, X_A20
end

function ScalingFunctions(M_A, X_A12, X_A22, X_A10, X_A20, r0)
    #---------------------------------------------------------
    #
    # Calculate the transformations
    #
    #---------------------------------------------------------    
    # Solve
    #   delta' = - A12/A22
    #   kappa' = A10 + A20 * delta'
    #---------------------------------------------------------
    A12 = t -> Eval(M_A, X_A12,[t])[1]
    A22 = t -> Eval(M_A, X_A22,[t])[1]
    A10 = t -> Eval(M_A, X_A10,[t])[1]
    A20 = t -> Eval(M_A, X_A20,[t])[1]
    # delta derivative
    deltap = -[A12(t) for t in r0]./[A22(t) for t in r0]
    # integrate deltap: trapezoid with zero initial condition
    delta = [0; cumsum((deltap[1:end-1] + deltap[2:end])/2 .* (r0[2:end] - r0[1:end-1]))]
    # kappa derivative
    kappap = [A10(t) for t in r0] .+ [A20(t) for t in r0].*deltap
    # integrate kappap: trapezoid with zero initial condition
    kappa = [0; cumsum((kappap[1:end-1] + kappap[2:end])/2 .* (r0[2:end] - r0[1:end-1]))]

    # inverting kappa : t = kappa^{-1}(r^2/2)
    t = copy(r0) # = kappa^{-1}(r^2/2)
    r = sqrt.(2*kappa)
    return deltap, delta, kappap, kappa, r, t
end

# output is a linear map (matrix) from the full phase space to some output variables
@doc raw"""
    freq, damp, r = ODEManifoldFrequencyDamping(W, R, r0; output=nothing)

Calculates the instantaneous frequencies and damping ratios of the ROM ``\boldsymbol{W}, \boldsymbol{R}``, where ``\boldsymbol{R}`` is in the real normal form
```math
\begin{pmatrix}\dot{z}_{1}\\
\dot{z}_{2}
\end{pmatrix} = \begin{pmatrix}z_{1}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)-z_{2}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)\\
z_{1}f_{i}\left(z_{1}^{2}+z_{2}^{2}\right)+z_{2}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)
\end{pmatrix}.
```

The input parameters are
  * `W` is ``\boldsymbol{W}``: decoder or manifold immersion
  * `S` is ``\boldsymbol{S}``: nonlinear map
  * `r0`: a vector of amplitudes that result is calculated at. This must be a reasonably fine mesh, because a finite difference is taken with respect to this mesh.
  * `output`: an optional linear map, that 
"""
function ODEManifoldFrequencyDamping(MWt, XWt, MS, XS, r0; output=nothing)
    if output == nothing
        M_A, X_A12, X_A22, X_A10, X_A20 = ManifoldGeometry(MWt, XWt)
    else
        MWtnew = DensePolyManifold(size(MWt.mexp,1), size(output,1), PolyOrder(MWt))
        XWtnew = output*XWt
        M_A, X_A12, X_A22, X_A10, X_A20 = ManifoldGeometry(MWtnew, XWtnew)
    end
    deltap_t, delta_t, kappap_t, kappa_t, r, t = ScalingFunctions(M_A, X_A12, X_A22, X_A10, X_A20, r0)
    
    function VF_S_T_R(S, r)
        T_r = zero(r)
        R_r = zero(r)
        for k=1:length(r)
            Sr = real(S([r[k], 0.0]))
            T_r[k] = abs(Sr[2]/r[k])
            R_r[k] = real(Sr[1])
        end
        return T_r, R_r
    end
    Sout = x -> Eval(MS, XS,x)
    # ODE
    T_t, R_t = VF_S_T_R(Sout, t)

    #ODE
    # needs kappap_t, kappap_t, R_t, T_t, r
    R_hat = R_t .* kappap_t ./ (r.^2)
    T_hat = T_t - R_t .* deltap_t

    return T_hat, -R_hat./T_hat, r
end    

@doc raw"""
    freq, damp, r, freq_old, damp_old, r_old = MAPManifoldFrequencyDamping(W, S, r0, Tstep; output=nothing)

Calculates the instantaneous frequencies and damping ratios of the ROM ``\boldsymbol{W}, \boldsymbol{S}``, where ``\boldsymbol{S}`` is in the real normal form
```math
\begin{pmatrix}z_{1}\\
z_{2}
\end{pmatrix} \mapsto \begin{pmatrix}z_{1}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)-z_{2}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)\\
z_{1}f_{i}\left(z_{1}^{2}+z_{2}^{2}\right)+z_{2}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)
\end{pmatrix}.
```

The input parameters are
  * `W` is ``\boldsymbol{W}``: decoder or manifold immersion
  * `S` is ``\boldsymbol{S}``: nonlinear map
  * `Tstep` is the time step that one application of ``\boldsymbol{S}`` represents
  * `r0`: a vector of amplitudes that result is calculated at. This must be a reasonably fine mesh, because a finite difference is taken with respect to this mesh.
  * `output`: an optional linear map, that 
"""
function MAPManifoldFrequencyDamping(MWt, XWt, MS, XS, r0, Tstep; output=nothing)
# function MAPManifoldFrequencyDamping(Wt, Sout, r0, Tstep; output=nothing)
    if output == nothing
        M_A, X_A12, X_A22, X_A10, X_A20 = ManifoldGeometry(MWt, XWt)
    else
        MWtnew = DensePolyManifold(size(MWt.mexp,1), size(output,1), PolyOrder(MWt))
        XWtnew = output*XWt
        M_A, X_A12, X_A22, X_A10, X_A20 = ManifoldGeometry(MWtnew, XWtnew)
    end
    deltap_t, delta_t, kappap_t, kappa_t, r, t = ScalingFunctions(M_A, X_A12, X_A22, X_A10, X_A20, r0)

    A12 = t -> Eval(M_A, X_A12,[t])[1]
    A22 = t -> Eval(M_A, X_A22,[t])[1]
    A10 = t -> Eval(M_A, X_A10,[t])[1]
    A20 = t -> Eval(M_A, X_A20,[t])[1]
    Sout = x -> Eval(MS, XS, x)
    
    function S_Real_Imag(S, r)
        Sr = S([r, 0.0])
        return Sr[1], -Sr[2]
    end
    function S_T_R(S, r)
        T_r = zero(r)
        R_r = zero(r)
        for k=1:length(r)
            fr, fi = S_Real_Imag(S, r[k])
            T_r[k] = abs(angle(fr+1im*fi))
            R_r[k] = sqrt(fr^2 + fi^2)
        end
        return T_r, R_r
    end
    # MAP
    T_t, R_t = S_T_R(Sout, t)

    deltap_R_t = -[A12(t_) for t_ in R_t]./[A22(t_) for t_ in R_t]
    delta_R_t = [0; cumsum((deltap_R_t[1:end-1] + deltap_R_t[2:end])/2 .* (R_t[2:end] - R_t[1:end-1]))]
    kappap_R_t = [A10(t_) for t_ in R_t] .+ [A20(t_) for t_ in R_t].*deltap_R_t
    kappa_R_t = [0; cumsum((kappap_R_t[1:end-1] + kappap_R_t[2:end])/2 .* (R_t[2:end] - R_t[1:end-1]))]
    
    # MAP
    # needs T_t, delta_t, kappa_R_t, delta_R_t
    if ~isempty(findall(kappa_R_t .< 0))
        println("WOULD HAVE BEEN DOMAIN ERROR")
    end
    R_hat = sqrt.(abs.(2*kappa_R_t))
    T_hat = T_t + delta_t - delta_R_t
    freq = T_hat/Tstep
    damp = - log.(R_hat ./ r) ./ T_hat
    
    freq_old = T_t / Tstep 
    damp_old = - log.(R_t ./ t) ./ T_t
    r_old = t
        
    return freq, damp, r, freq_old, damp_old, r_old
end
