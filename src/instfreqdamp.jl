function ScalingFunctions(MW::DensePolyManifold{ndim, n, max_polyorder, false, ℂ}, XW, amp_max; output = ones(1,n)) where {ndim, n, max_polyorder}
    setConstantPart!(MW, XW, zero(size(XW,1)))
    XWres = output * XW
    # we have
    #   W_ipk W_iql r^(p1+p2+q1+q2) d_(p1-p2+q1-q2) d_(k+l)
    # so let l = -k
#     QPFourierPolynomial(dim_out, dim_in, fourier_order, min_polyorder, max_polyorder)
#            QPPolynomial(dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, field::AbstractNumbers=ℝ)
    Mres = DensePolyManifold(1, 1, 2*max_polyorder)
    # kappa = sqrt( r^2 alpha(r) )
    X_alpha = zero(Mres)
    X_g_g = zero(Mres)
    X_g_r = zero(Mres)
    for pid = 1:size(MW.mexp,2), qid = 1:size(MW.mexp,2)
        p1 = MW.mexp[1,pid]
        p2 = MW.mexp[2,pid]
        q1 = MW.mexp[1,qid]
        q2 = MW.mexp[2,qid]
        ord = p1 + p2 + q1 + q2
        fourier_ord = p2 - p1 + (q1 - q2)
        if (fourier_ord == 0) && (ord > 0)
            # here ord is even, because p1 = p2 + q1 - q2
            # hence ord = p1 + p2 + q1 + q2 = 2*p2 + 2*q1
#             ord_half = div(ord, 2)
#             oid = PolyFindIndex(Mres.mexp, ord)
#             ohid = PolyFindIndex(Mres.mexp, ord_half)
#             oidm1 = PolyFindIndex(Mres.mexp, ord-1)
            oidm2 = PolyFindIndex(Mres.mexp, ord-2)
            oidm3 = PolyFindIndex(Mres.mexp, ord-3)
            #
            XW_sq = sum(conj(XWres[:,pid]) .* XWres[:,qid])
            X_alpha[1,oidm2] += real(XW_sq)
            X_g_g[1,oidm2]   += real((p1-p2) * (q1-q2) * XW_sq)
            if oidm3 != nothing
                X_g_r[1,oidm3] += real((p1+p2) * 1im * (q1-q2) * XW_sq)
            else
                @show real((p1+p2) * 1im * (q1-q2) * XW_sq)
            end
        end
    end
    # now fit Chebyshev polynomials to the required quantities
    # We need to do this because Taylor expansion of polynomials divided is not a good idea
    r_max = InverseScalar(Mres, X_alpha, amp_max, zero(amp_max))
    # kappa
    S_kappa = Chebyshev(-0.02*r_max .. 1.02*r_max)
    kappa   = Fun(a -> a*sqrt(EvalScalar(Mres, X_alpha, a)), S_kappa)
    kappa_r = Fun(a -> sqrt(EvalScalar(Mres, X_alpha, a)), S_kappa)
    D_kappa = Derivative(S_kappa) * kappa
    
    # rho
    k_max = 0.98 * maximum(kappa)
    k_min = 0.98 * minimum(kappa)
    S_rho = Chebyshev(k_min .. k_max)
    rho = Fun(a -> InverseScalar(Mres, X_alpha, a, zero(a)), S_rho)
    rho_r = Fun(a -> InverseScalar(Mres, X_alpha, a, zero(a))/a, S_rho)

    # now phi_deri
    D_gamma = Fun(a -> EvalScalar(Mres, X_g_r, a) / EvalScalar(Mres, X_g_g, a), S_kappa)
    Iop = Integral(S_kappa)
    gamma = Iop * D_gamma
    gamma = gamma - gamma(0.0)
        
    return kappa, kappa_r, D_kappa, rho, rho_r, gamma, D_gamma, r_max
end

function ScalingFunctionsBrute(MW::DensePolyManifold{ndim, n, max_polyorder, false, ℂ}, XW, r_max; output = ones(1,n)) where {ndim, n, max_polyorder}
    pts = 25
    # do it by brute force
    W = (r,q) -> output * Eval(MW, XW, [r*exp(1im*q); r*exp(-1im*q)])
    S_kappa = Chebyshev(-0.1*r_max .. 1.1*r_max)
    Srq = TensorSpace(S_kappa, Fourier())
    Wf = Fun(W, Srq, pts^2)
    
    tq = range(0,2*pi,length=101)[1:end-1]
    kappa_fun = r -> sign(r) * sqrt(sum(norm.(W.(r,tq)) .^ 2) / length(tq))
    kappa = Fun(kappa_fun, S_kappa, pts)
    kappa_r = Fun(r -> kappa_fun(r) / r, S_kappa, pts)
    D_kappa = Derivative(S_kappa) * kappa
    
    k_max = 0.9 * maximum(kappa)
    k_min = 0.9 * minimum(kappa)
    S_rho = Chebyshev(k_min .. k_max)

    rho_fun = r -> findroot(kappa_fun, D_kappa, r, 0.0)
    rho = Fun(rho_fun, S_rho)
    rho_r = Fun(r -> rho_fun(r) / r, S_rho)
    
    Wf_q = Derivative(Srq,[0,1]) * Wf
    Wf_r = Derivative(Srq,[1,0]) * Wf
    Wf_qq = r -> real(sum(conj.(Wf_q.(r,tq)) .* Wf_q.(r,tq)) / length(tq))
    Wf_qr = r -> real(sum(conj.(Wf_q.(r,tq)) .* Wf_r.(r,tq)) / length(tq))

    D_gamma = Fun(a -> Wf_qr(a) / Wf_qq(a), S_kappa)
    Iop = Integral(S_kappa)
    gamma = Iop * D_gamma
    gamma = gamma - gamma(0.0)
    return kappa, kappa_r, D_kappa, rho, rho_r, gamma, D_gamma
end

function findroot(fun, Dfun, v, v0; maxit = 1000)
    xk = v0
    x = v0
    k = 0
    while true
        xk = x
        x = xk - (fun(xk) - v) / Dfun(xk)
        if k > maxit
            @show k, abs(fun(x) - v), Dfun(x)
            break
        end
        if abs(x-xk) < 20*eps(1.0)
            break
        end
        k += 1
    end
    return x
end

function MAPtoPolar(MS::DensePolyManifold{2, 2, max_polyorder, false, ℂ}, XS, r_max) where {max_polyorder}
    ords = dropdims(sum(MS.mexp, dims=1), dims=1)
    oid = findall(ords .> 0)
    f_p_r = r -> sum(XS[1,oid] .* (r .^ (ords[oid] .- 1)))
    f_p = r -> sum(XS[1,:] .* (r .^ ords))
    S_fun = Chebyshev(-0.02*r_max .. 1.02*r_max)
    R = Fun(r -> sign(r) * abs(f_p(r)), S_fun)
    R_r = Fun(r -> abs(f_p_r(r)), S_fun)
    T = Fun(r -> angle(f_p_r(r)), S_fun)
    return R, R_r, T
end

function ODEtoPolar(MS::DensePolyManifold{2, 2, max_polyorder, false, ℂ}, XS, r_max) where {max_polyorder}
    ords = dropdims(sum(MS.mexp, dims=1), dims=1)
    oid = findall(ords .> 0)
    f_p_r = r -> sum(XS[1,oid] .* (r .^ (ords[oid] .- 1)))
    f_p = r -> sum(XS[1,:] .* (r .^ ords))
    S_fun = Chebyshev(-0.02*r_max .. 1.02*r_max)
    R = Fun(r -> real(f_p(r)), S_fun)
    R_r = Fun(r -> real(f_p_r(r)), S_fun)
    T = Fun(r -> imag(f_p_r(r)), S_fun)
    return R, R_r, T
end

function MAPtoPolar(MS::DensePolyManifold{2, 2, max_polyorder, false, ℝ}, XS, r_max) where {max_polyorder}
    # the input is [r, 0]
    # the output is fr = x[1], fi = -x[2]
    # T(r) = angle(fr + 1im*fi)
    # R(r) = sqrt(fr^2 + fi^2)
    ords = MS.mexp[1,:]
    oid = findall((MS.mexp[2,:] .== 0) .&& (MS.mexp[1,:] .> 0))
    fr_p_r = r ->  sum(XS[1,oid] .* (r .^ (ords[oid] .- 1)))
    fi_p_r = r -> -sum(XS[2,oid] .* (r .^ (ords[oid] .- 1)))
    fr_p   = r ->  sum(XS[1,oid] .* (r .^ ords[oid]))
    fi_p   = r -> -sum(XS[2,oid] .* (r .^ ords[oid]))
    S_fun = Chebyshev(-0.02*r_max .. 1.02*r_max)
    R   = Fun(r -> sign(r) * sqrt(fr_p(r)^2 + fi_p(r)^2), S_fun)
    R_r = Fun(r -> sqrt(fr_p_r(r)^2 + fi_p_r(r)^2), S_fun)
    T   = Fun(r -> angle(fr_p_r(r) + 1im*fi_p_r(r)), S_fun)
    return R, R_r, T
end

function ODEtoPolar(MS::DensePolyManifold{2, 2, max_polyorder, false, ℝ}, XS, r_max) where {max_polyorder}
    # the input is [r, 0]
    # the output is fr = x[1], fi = -x[2]
    # T(r) = angle(fr + 1im*fi)
    # R(r) = sqrt(fr^2 + fi^2)
    ords = MS.mexp[1,:]
    oid = findall((MS.mexp[2,:] .== 0) .&& (MS.mexp[1,:] .> 0))
    fr_p_r = r ->  sum(XS[1,oid] .* (r .^ (ords[oid] .- 1)))
    fi_p_r = r -> -sum(XS[2,oid] .* (r .^ (ords[oid] .- 1)))
    fr_p   = r ->  sum(XS[1,oid] .* (r .^ ords[oid]))
#     fi_p   = r -> -sum(XS[2,oid] .* (r .^ ords[oid]))
    S_fun = Chebyshev(-0.02*r_max .. 1.02*r_max)
    R   = Fun(r -> fr_p(r), S_fun)
    R_r = Fun(r -> fr_p_r(r), S_fun)
    T   = Fun(r -> fi_p_r(r), S_fun)
    return R, R_r, T
end

# should be renamed to MapPolarForm
@doc raw"""
    That, Rhat = MAPFrequencyDamping(MW::DensePolyManifold, XW, MS::DensePolyManifold, XS, amp_max::Number; output = I)

Creates a polar representation of the system, written as
```math
\begin{pmatrix}r\\
\theta
\end{pmatrix} \mapsto
\begin{pmatrix}r \hat{R}(r)\\
\theta + \hat{T}(r)
\end{pmatrix}
```

The independent variable ``r`` represent the root mean square amplitude of the vibration over one period. Also, the frequency represented by ``\hat{T}(r)`` is correct with respect to the Euclidean frame.

The input parameters are
* `MW`, `XW` is ``\boldsymbol{W}``: decoder or manifold immersion
* `MS`, `XS` is ``\boldsymbol{S}``: two-dimenional nonlinear map
* `amp_max`: maximum amplitude accurately represented by 
* `output`: an optional linear map, that transforms the output of the immersion ``\boldsymbol{W}``. By default, the identity map is used.
   
It is assumed the the deconder and the nonlinear map satisfy the invariance equation
```math
\boldsymbol{W} \circ \boldsymbol{S} = \boldsymbol{F} \circ \boldsymbol{W},
```
where ``\boldsymbol{F}`` represent a dynamical system ``\dot{\boldsymbol{x}} = \boldsymbol{F}(\boldsymbol{x})``.

The two-dimensional system ``\boldsymbol{S}`` has a focus-type equilibrium at the origin, either in the real normal form
```math
\begin{pmatrix}z_{1}\\
z_{2}
\end{pmatrix} \mapsto \begin{pmatrix}z_{1}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)-z_{2}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)\\
z_{1}f_{i}\left(z_{1}^{2}+z_{2}^{2}\right)+z_{2}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)
\end{pmatrix}.
```
or in the complex normal form
```math
\begin{pmatrix}z\\
\overline{z}
\end{pmatrix} \mapsto
\begin{pmatrix} z + z f(\vert z \vert^2) \\
\overline{z} + \overline{z} \overline{f}(\vert z \vert^2)
\end{pmatrix}
```

The output `That`, `Rhat` represent ``\hat{T}, \hat{R}: [0,r_1) \to \mathbb{R}``, which satisfy the invariance equation
```math
\hat{\boldsymbol{W}}\left(r \hat{R}(r), \theta + \hat{T}(r)\right) = \boldsymbol{F}( \hat{\boldsymbol{W}}\left(r, \theta\right) ).
```
"""
function MAPFrequencyDamping(MW::DensePolyManifold{ndim, n, Worder, false, field}, XW, MS::DensePolyManifold{2, 2, Sorder, false, field}, XS, amp_max; output = I) where {ndim, n, Worder, Sorder, field}
    kappa, kappa_r, D_kappa, rho, rho_r, gamma, D_gamma, r_max = ScalingFunctions(MW, XW, amp_max, output = output)
    R, R_r, T = MAPtoPolar(MS, XS, r_max)

    len = maximum(map(m -> length(m.coefficients), (kappa, kappa_r, D_kappa, rho, rho_r, gamma, D_gamma, R, R_r, T)))
    S_rho = space(rho)
    That = Fun(r -> T(rho(r)) + gamma(rho(r)) - gamma(R(rho(r))), S_rho)
    Rhat_r = Fun(r -> rho_r(r) * R_r(rho(r)) * kappa_r(R(rho(r))), S_rho)
    return That, Rhat_r, rho, gamma
end

# should be renamed OdePolarForm
@doc raw"""
    That, Rhat = ODEFrequencyDamping(MW::DensePolyManifold, XW, MS::DensePolyManifold, XS, amp_max::Number; output = I)

Creates a polar representation of the system, written as
```math
\begin{pmatrix}\dot{r}\\
\dot{\theta}
\end{pmatrix} = 
\begin{pmatrix}r \hat{R}(r)\\
\hat{T}(r)
\end{pmatrix}
```

The independent variable ``r`` represent the root mean square amplitude of the vibration over one period. Also, the frequency represented by ``\hat{T}(r)`` is correct with respect to the Euclidean frame.
    
The input parameters are
* `MW`, `XW` is ``\boldsymbol{W}``: decoder or manifold immersion
* `MS`, `XS` is ``\boldsymbol{S}``: two-dimenional nonlinear map
* `amp_max`: maximum amplitude accurately represented by 
* `output`: an optional linear map, that transforms the output of the immersion ``\boldsymbol{W}``. By default, the identity map is used.
   
It is assumed that the deconder and the nonlinear map satisfy the invariance equation
```math
\boldsymbol{W} \cdot D\boldsymbol{S} = \boldsymbol{F} \circ \boldsymbol{W},
```
where ``\boldsymbol{F}`` represent the right-hand side of a differential equation ``\dot{\boldsymbol{x}} = \boldsymbol{F}(\boldsymbol{x})``.

The two-dimensional system ``\boldsymbol{S}`` has a focus-type equilibrium at the origin, either in the real normal form
```math
\begin{pmatrix}\dot{z}_{1}\\
\dot{z}_{2}
\end{pmatrix} = \begin{pmatrix}z_{1}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)-z_{2}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)\\
z_{1}f_{i}\left(z_{1}^{2}+z_{2}^{2}\right)+z_{2}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)
\end{pmatrix}.
```
or in the complex normal form
```math
\begin{pmatrix}\dot{z}\\
\dot{\overline{z}}
\end{pmatrix} =
\begin{pmatrix} z + z f(\vert z \vert^2) \\
\overline{z} + \overline{z} \overline{f}(\vert z \vert^2)
\end{pmatrix}
```

The output `That`, `Rhat` represent ``\hat{T}, \hat{R}: [0,r_1) \to \mathbb{R}``, which satisfy the invariance equation
```math
D_1\hat{\boldsymbol{W}}\left(r, \theta\right)r \hat{R}(r) + D_2\hat{\boldsymbol{W}}\left(r, \theta\right)\hat{T}(r) = \boldsymbol{F}( \hat{\boldsymbol{W}}\left(r, \theta\right) ).
```
"""
function ODEFrequencyDamping(MW::DensePolyManifold{ndim, n, Worder, false, field}, XW, MS::DensePolyManifold{2, 2, Sorder, false, field}, XS, amp_max; output = I) where {ndim, n, Worder, Sorder, field}
    kappa, kappa_r, D_kappa, rho, rho_r, gamma, D_gamma, r_max = ScalingFunctions(MW, XW, amp_max, output = output)
    R, R_r, T = ODEtoPolar(MS, XS, r_max)

    len = maximum(map(m -> length(m.coefficients), (kappa, kappa_r, D_kappa, rho, rho_r, gamma, R, R_r, T)))
    S_rho = space(rho)
    That = Fun(r -> T(rho(r)) - D_gamma(rho(r))*R(rho(r)), S_rho, 5*len)
    Rhat_r = Fun(r -> R_r(rho(r)) * rho_r(r) * D_kappa(rho(r)), S_rho, 5*len)

    return That, Rhat_r
end

function EvalScalar(MS::DensePolyManifold{1, 1, max_polyorder, identity, field}, XS, r::Number) where {max_polyorder, identity, field}
    fr = XS * dropdims(prod(r .^ MS.mexp, dims=1), dims=1)
    return fr[1]
end

function EvalDeriScalar(MS::DensePolyManifold{1, 1, max_polyorder, identity, field}, XS, r::Number) where {max_polyorder, identity, field}
    mons = dropdims(prod(r .^ MS.mexp, dims=1), dims=1)
    fr = XS * MS.DM[1] * mons
    return fr[1]
end

function InverseScalar(MS::DensePolyManifold{1, 1, max_polyorder, identity, field}, XS, r::Number, r0::Number; maxit = 100) where {max_polyorder, identity, field}
    f  = z_ -> EvalScalar(MS, XS, z_)
    df = z_ -> EvalDeriScalar(MS, XS, z_)
    g  = z_ -> z_ * sqrt(f(z_))
    dg = z_ -> sqrt(f(z_)) + z_ * df(z_) / (2 * sqrt(f(z_)))
    
    xk = r0
    x = r0
    k = 1
    while true
        xk = x
        x = xk - (g(xk) - r) / dg(xk)
        if abs(x-xk) <= 5*eps(1.0)
            break
        end
        if k > maxit
            @show k, abs(x-xk)
            break
        end
        k += 1
    end
    return x
end

function ValDeri(MS::DensePolyManifold{n, n, max_polyorder, identity, field}, XS, r) where {n, max_polyorder, identity, field}
    mons = dropdims(prod(r .^ MS.mexp, dims=1), dims=1)
    val = XS * mons
    jac = zeros(eltype(val), n, n)
    for k=1:n
        jac[:,k] .= XS * MS.DM[k] * mons
    end
    return val, jac
end

function InverseMap(MS::DensePolyManifold{n, n, max_polyorder, identity, field}, XS, r::Vector; maxit = 200) where {n, max_polyorder, identity, field}
    xk = zero(r)
    x = copy(xk)
    k = 1
    while true
        xk .= x
        f_xk, df_xk = ValDeri(MS, XS, xk)
        x .= xk - df_xk \ (f_xk - r)
        if norm(x .- xk) < 5*n*eps()
            break
        end
        if k > maxit
            @show k, norm(x .- xk)
            break
        end
        k += 1
    end
    @show k, norm(x .- xk)
    return x
end

function CosP_SinQ(p,q)
    if iseven(p)*iseven(q)
        return gamma((p+1.0)/2)*gamma((q+1.0)/2) / factorial(div(p+q,2)) / pi
    else
        return 0.0
    end
end

function createCircularIntegral(M::DensePolyManifold, r)
    return DensePolyManifold(1, 1, PolyOrder(M) + r)
end

# Calculate the integral the g(r) = 1/(2pi) \int_0^{2\pi} f(r Cos(t), r Sin(t) ) * cos(t)^c * sin(t)^s dt
function circularIntegrate(MO, M::DensePolyManifold, X, c_, s_, r_)
    XO = zero(MO)
    XI = vec(X)
    for k=1:size(M.mexp,2)
        p = M.mexp[1,k]
        q = M.mexp[2,k]
        id = findfirst(isequal(p+q+r_), MO.mexp[1,:])
        if id != nothing
            XO[1,id] += CosP_SinQ(p+c_,q+s_) * XI[k]
        elseif abs(CosP_SinQ(p+c_,q+s_) * XI[k]) >= eps(XI[k])
            println("NOT FOUND p+c=", p+c_, " q+s=", q+s_, " p+q+r=", p+q+r_, " CosP_SinQ(p+c,q+s) * XI[k]=", CosP_SinQ(p+c_,q+s_) * XI[k])
        end
    end
    return XO
end

function ScalingFunctions(MWt::DensePolyManifold{ndim, n, max_polyorder, false, ℝ}, XW, amp_max; output = ones(1,n)) where {ndim, n, max_polyorder}
    setConstantPart!(MWt, XW, zero(size(XW,1)))
    XWt = output * XW
#     @show getLinearPart(MWt, XWt)
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
    #   A00 = Int < D2 W(r,t),    W(r,t) > dt / (2 Pi) = r Int W^T . W dt
    #   A12 = Int < D1 W(r,t), D2 W(r,t) > dt / (2 Pi) = r Int [ Cos(t), Sin(t) ] . DW^T * DW . [ -Sin(t), Cos(t) ] dt
    #   A22 = Int < D2 W(r,t), D2 W(r,t) > dt / (2 Pi) = r^2 Int [ -Sin(t), Cos(t) ] . DW^T * DW . [ -Sin(t), Cos(t) ] dt
    # Let 
    #   DW^T * DW = ( a11 a12 )
    #               ( a21 a22 )
    #   with r^(p+q) Cos^p * Sin^q coefficient, i.e., -> (p,q)
    # Then
    # 1. A12
    #   [ Cos(t), Sin(t) ] . DW^T * DW . [ -Sin(t), Cos(t) ] = 
    #   = - a11 r Cos[t] Sin[t] + a12 r Cos[t]^2 - a21 r Sin[t]^2 + a22 r Cos[t] Sin[t]
    #   = r^(p+q+1) [ (a22 - a11) Cos^(p+1) * Sin^(q+1)
    #              + a12 * Cos^(p+2) * Sin^(q)
    #              - a21 * Cos^(p) * Sin^(q+2) ]
    # 2. A22
    #   [ -Sin(t), Cos(t) ] . DW^T * DW . [ -Sin(t), Cos(t) ] = 
    #   = a11 r^2 Sin[t]^2 - a12 r^2 Cos[t] Sin[t] - a21 r^2 Cos[t] Sin[t] + a22 r^2 Cos[t]^2 
    #   = r^(p+q+2) [ -(a12 + a21) Cos^(p+1) * Sin^(q+1)
    #              + a22 * Cos^(p+2) * Sin^(q)
    #              + a11 * Cos^(p) * Sin^(q+2) ]
    # For the integrals, we need to use the Gamma special function an factorial
    #   Int Cos^{2n}(t) Sin^{2m}(t) dt = Gamma(n+1/2) Gamma(m+1/2) / pi / (m+n)!
    # Solve
    #   delta' = - A12/A22
    #   kappa' = A10 + A20 * delta'
    #---------------------------------------------------------
    din = size(XWt,1)
    dout = size(MWt.mexp,1)
    order = PolyOrder(MWt)

    # for A00
    M_WtWt = DensePolyManifold(dout, 1, 2*order)
    X_WtWt = zero(M_WtWt)
    DensePolySquared!(M_WtWt, X_WtWt, MWt, XWt)
    # DW^T . W
    M_DWtr_W = DensePolyManifold(dout, dout, 2*order-1)
    X_DWtr_W = zero(M_DWtr_W)
    DensePolyDeriTransposeMul!(M_DWtr_W, X_DWtr_W, MWt, XWt)

    # DW^T . DW
    M_DWtr_DW = DensePolyManifold(dout, dout, 2*order-1)
    X_DWtr_DW = zeroJacobianSquared(M_DWtr_DW)
    DensePolyJabobianSquared!(M_DWtr_DW, X_DWtr_DW, MWt, XWt)
    
    # 2 is sufficient, 3 is important, of we want to integrate kappa accurately
    MO = createCircularIntegral(M_DWtr_W, 3)
    X_00    = circularIntegrate(MO, M_WtWt,     X_WtWt[1,:], 0, 0, 0 - 2) # division by r^2
    X_12_11 = circularIntegrate(MO, M_DWtr_DW, -X_DWtr_DW[1,1,:], 1, 1, 1 - 2) # -1 is division by r
    X_12_12 = circularIntegrate(MO, M_DWtr_DW,  X_DWtr_DW[1,2,:], 2, 0, 1 - 2)
    X_12_21 = circularIntegrate(MO, M_DWtr_DW, -X_DWtr_DW[2,1,:], 0, 2, 1 - 2)
    X_12_22 = circularIntegrate(MO, M_DWtr_DW,  X_DWtr_DW[2,2,:], 1, 1, 1 - 2)
    X_12 = X_12_11 + X_12_12 + X_12_21 + X_12_22
        
    X_22_11 = circularIntegrate(MO, M_DWtr_DW,  X_DWtr_DW[1,1,:], 0, 2, 2 - 2)
    X_22_12 = circularIntegrate(MO, M_DWtr_DW, -X_DWtr_DW[1,2,:], 1, 1, 2 - 2)
    X_22_21 = circularIntegrate(MO, M_DWtr_DW, -X_DWtr_DW[2,1,:], 1, 1, 2 - 2)
    X_22_22 = circularIntegrate(MO, M_DWtr_DW,  X_DWtr_DW[2,2,:], 2, 0, 2 - 2)
    X_22 = X_22_11 + X_22_12 + X_22_21 + X_22_22
        
    # X_00 is kappa_r squared
    # - X_12 / X_22 is gamma_derivative
    A00 = t -> Eval(MO, X_00,[t])[1]
    A12 = t -> Eval(MO, X_12,[t])[1]
    A22 = t -> Eval(MO, X_22,[t])[1]

    r_max = InverseScalar(MO, X_00, amp_max, zero(amp_max))
    # now fit Chebyshev polynomials to the required quantities
    # We need to do this because Taylor expansion of polynomials divided is not a good idea
    # kappa
    S_kappa = Chebyshev(-0.02*r_max .. 1.02*r_max)
    kappa   = Fun(a -> a*sqrt(A00(a)), S_kappa)
    kappa_r = Fun(a -> sqrt(A00(a)), S_kappa)
    D_kappa = Derivative(S_kappa) * kappa
    
    # rho
    k_max = 0.98 * maximum(kappa)
    k_min = 0.98 * minimum(kappa)
    S_rho = Chebyshev(k_min .. k_max)
    rho = Fun(a -> InverseScalar(MO, X_00, a, zero(a)), S_rho)
    rho_r = Fun(a -> InverseScalar(MO, X_00, a, zero(a))/a, S_rho)

    # now phi_deri
    D_gamma = Fun(a -> -A12(a)/A22(a), S_kappa)
    Iop = Integral(S_kappa)
    gamma = Iop * D_gamma
    gamma = gamma - gamma(0.0)
    
    @show X_00
    @show X_12
    @show X_22
    return kappa, kappa_r, D_kappa, rho, rho_r, gamma, D_gamma, r_max
end

# # output is a linear map (matrix) from the full phase space to some output variables
# @doc raw"""
#     freq, damp, r = ODEManifoldFrequencyDamping(W, R, r0; output=nothing)
# 
# Calculates the instantaneous frequencies and damping ratios of the ROM ``\boldsymbol{W}, \boldsymbol{R}``, where ``\boldsymbol{R}`` is in the real normal form
# ```math
# \begin{pmatrix}\dot{z}_{1}\\
# \dot{z}_{2}
# \end{pmatrix} = \begin{pmatrix}z_{1}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)-z_{2}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)\\
# z_{1}f_{i}\left(z_{1}^{2}+z_{2}^{2}\right)+z_{2}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)
# \end{pmatrix}.
# ```
# 
# The input parameters are
#   * `W` is ``\boldsymbol{W}``: decoder or manifold immersion
#   * `S` is ``\boldsymbol{S}``: nonlinear map
#   * `r0`: a vector of amplitudes that result is calculated at. This must be a reasonably fine mesh, because a finite difference is taken with respect to this mesh.
#   * `output`: an optional linear map, that 
# """
# function ODEManifoldFrequencyDamping(MWt, XWt, MS, XS, r0; output=nothing)
#     if output == nothing
#         M_A, X_A12, X_A22, X_A10, X_A20 = ManifoldGeometry(MWt, XWt)
#     else
#         MWtnew = DensePolyManifold(size(MWt.mexp,1), size(output,1), PolyOrder(MWt))
#         XWtnew = output*XWt
#         M_A, X_A12, X_A22, X_A10, X_A20 = ManifoldGeometry(MWtnew, XWtnew)
#     end
#     deltap_t, delta_t, kappap_t, kappa_t, r, t = ScalingFunctions(M_A, X_A12, X_A22, X_A10, X_A20, r0)
#     
#     function VF_S_T_R(S, r)
#         T_r = zero(r)
#         R_r = zero(r)
#         for k=1:length(r)
#             Sr = real(S([r[k], 0.0]))
#             T_r[k] = abs(Sr[2]/r[k])
#             R_r[k] = real(Sr[1])
#         end
#         return T_r, R_r
#     end
#     Sout = x -> Eval(MS, XS,x)
#     # ODE
#     T_t, R_t = VF_S_T_R(Sout, t)
# 
#     #ODE
#     # needs kappap_t, kappap_t, R_t, T_t, r
#     R_hat = R_t .* kappap_t ./ (r.^2)
#     T_hat = T_t - R_t .* deltap_t
# 
#     return T_hat, -R_hat./T_hat, r
# end    

# @doc raw"""
#     freq, damp, r, freq_old, damp_old, r_old = MAPManifoldFrequencyDamping(W, S, r0, Tstep; output=nothing)
# 
# Calculates the instantaneous frequencies and damping ratios of the ROM ``\boldsymbol{W}, \boldsymbol{S}``, where ``\boldsymbol{S}`` is in the real normal form
# ```math
# \begin{pmatrix}z_{1}\\
# z_{2}
# \end{pmatrix} \mapsto \begin{pmatrix}z_{1}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)-z_{2}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)\\
# z_{1}f_{i}\left(z_{1}^{2}+z_{2}^{2}\right)+z_{2}f_{r}\left(z_{1}^{2}+z_{2}^{2}\right)
# \end{pmatrix}.
# ```
# 
# The input parameters are
#   * `W` is ``\boldsymbol{W}``: decoder or manifold immersion
#   * `S` is ``\boldsymbol{S}``: nonlinear map
#   * `Tstep` is the time step that one application of ``\boldsymbol{S}`` represents
#   * `r0`: a vector of amplitudes that result is calculated at. This must be a reasonably fine mesh, because a finite difference is taken with respect to this mesh.
#   * `output`: an optional linear map, that 
# """
# function MAPManifoldFrequencyDamping(MWt, XWt, MS, XS, r0, Tstep; output=nothing)
# # function MAPManifoldFrequencyDamping(Wt, Sout, r0, Tstep; output=nothing)
#     if output == nothing
#         M_A, X_A12, X_A22, X_A10, X_A20 = ManifoldGeometry(MWt, XWt)
#     else
#         MWtnew = DensePolyManifold(size(MWt.mexp,1), size(output,1), PolyOrder(MWt))
#         XWtnew = output*XWt
#         M_A, X_A12, X_A22, X_A10, X_A20 = ManifoldGeometry(MWtnew, XWtnew)
#     end
#     deltap_t, delta_t, kappap_t, kappa_t, r, t = ScalingFunctions(M_A, X_A12, X_A22, X_A10, X_A20, r0)
# 
#     A12 = t_ -> Eval(M_A, X_A12,[t_])[1]
#     A22 = t_ -> Eval(M_A, X_A22,[t_])[1]
#     A10 = t_ -> Eval(M_A, X_A10,[t_])[1]
#     A20 = t_ -> Eval(M_A, X_A20,[t_])[1]
#     Sout = x -> Eval(MS, XS, x)
#     
#     function S_Real_Imag(S, r)
#         Sr = S([r, 0.0])
#         return Sr[1], -Sr[2]
#     end
#     function S_T_R(S, r)
#         T_r = zero(r)
#         R_r = zero(r)
#         for k=1:length(r)
#             fr, fi = S_Real_Imag(S, r[k])
#             T_r[k] = abs(angle(fr+1im*fi))
#             R_r[k] = sqrt(fr^2 + fi^2)
#         end
#         return T_r, R_r
#     end
#     # MAP
#     T_t, R_t = S_T_R(Sout, t)
# 
#     deltap_R_t = -[A12(t_) for t_ in R_t]./[A22(t_) for t_ in R_t]
#     delta_R_t = [0; cumsum((deltap_R_t[1:end-1] + deltap_R_t[2:end])/2 .* (R_t[2:end] - R_t[1:end-1]))]
#     kappap_R_t = [A10(t_) for t_ in R_t] .+ [A20(t_) for t_ in R_t].*deltap_R_t
#     kappa_R_t = [0; cumsum((kappap_R_t[1:end-1] + kappap_R_t[2:end])/2 .* (R_t[2:end] - R_t[1:end-1]))]
#     
#     # MAP
#     # needs T_t, delta_t, kappa_R_t, delta_R_t
#     if ~isempty(findall(kappa_R_t .< 0))
#         println("WOULD HAVE BEEN DOMAIN ERROR")
#     end
#     R_hat = sqrt.(abs.(2*kappa_R_t))
#     T_hat = T_t + delta_t - delta_R_t
#     freq = T_hat/Tstep
#     damp = - log.(R_hat ./ r) ./ T_hat
#     
#     freq_old = T_t / Tstep 
#     damp_old = - log.(R_t ./ t) ./ T_t
#     r_old = t
#         
#     return freq, damp, r, freq_old, damp_old, r_old
# end
# 
# function testFREQ()
#     M_W = DensePolyManifold(2, 2, 7, min_order=0)
#     X_W = zero(M_W)
#     M_S = DensePolyManifold(2, 2, 7, min_order=0)
#     X_S = zero(M_S)
#     setLinearPart!(M_S, X_S, [cos(1.0) -sin(1.0); sin(1.0) cos(1.0)]./exp(1/50))
#     setLinearPart!(M_W, X_W, [1.0 0.0; 0.0 1.0])
#     id = PolyFindIndex(M_W.mexp, [3; 0])
#     @show M_W.mexp[:,id] X_W[:,id]
#     X_W[:,id] .= [-1/4.0, 1/2.0]
#     freq, damp, r, freq_old, damp_old, r_old = MAPManifoldFrequencyDamping(M_W, X_W, M_S, X_S, collect(range(0,1,length=1000)), 1.0, output = [1 0; 0 1])
#     return freq, damp, r, freq_old, damp_old, r_old
# end
