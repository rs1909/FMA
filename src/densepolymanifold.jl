# -------------------------------------------------------
# VECTOR VALUED MODELS AND HELPER FUNCTIONS
# -------------------------------------------------------

function DensePolyExponents(ndim::Integer, order::Integer; min_order = 0)
    @polyvar x[1:ndim]
    mx0 = monomials(x, min_order:order)
    return hcat([exponents(m) for m in mx0]...)
end

# defined in polymethods
# function PolyOrder(mexp::AbstractArray)
#     return maximum(sum(mexp, dims=1))
# end

# defined in polymethods
# find the indices of "order' order terms.
# function PolyOrderIndices(mexp::AbstractArray, order::Integer)
#     return findall(isequal(order), dropdims(sum(mexp,dims=1),dims=1))
# end

# ndim: input dimensionality
# n:    output dimensionality
struct DensePolyManifold{ndim, n, order, identity, 𝔽} <: AbstractManifold{𝔽}
    mexpALL
    mexp
    admissible
    M        :: AbstractManifold{𝔽} # Euclidean{Tuple{n, ndim}, 𝔽}
    R        :: AbstractRetractionMethod
    VT       :: AbstractVectorTransportMethod
end

struct NonlinearRetraction <: AbstractRetractionMethod
end

struct NonconstRetraction <: AbstractRetractionMethod
end

@doc raw"""
    M = DensePolyManifold(ndim, n, order; min_order = 0, identity = false, field::AbstractNumbers=ℝ)
    
Creates a manifold structure of a dense polynomial with `ndim` input variables and `n` output dimension of maximum order `order`. 
One can set the smallest represented order by `min_order`. The parameter `field` can be set to either real `ℝ` or complex `ℂ`.
"""
function DensePolyManifold(ndim, n, order; min_order = 0, identity = false, field::AbstractNumbers=ℝ)
    mexpALL = DensePolyExponents(ndim, order)
    admissible = findall(dropdims(sum(mexpALL, dims=1), dims=1) .>= min_order)
    return DensePolyManifold{ndim, n, order, identity, field}(mexpALL, mexpALL[:,admissible], admissible, Euclidean(n, length(admissible); field), ExponentialRetraction(), ParallelTransport())
end

function DenseNonlinearManifold(ndim, n, order; field::AbstractNumbers=ℝ)
    return DensePolyManifold(ndim, n, order; min_order = 2, field = field)
end

function DenseNearIdentityManifold(ndim, n, order; field::AbstractNumbers=ℝ)
    return DensePolyManifold(ndim, n, order; min_order = 2, identity = true, field = field)
end

function DenseNonconstManifold(ndim, n, order; field::AbstractNumbers=ℝ)
    return DensePolyManifold(ndim, n, order; min_order = 1, field = field)
end

function PolyOrder(M::DensePolyManifold{ndim, n, order, identity, field}) where {ndim, n, order, identity, field}
    return order
end

function PolyOrderIndices(M::DensePolyManifold{ndim, n, order, identity, field}, o::Integer) where {ndim, n, order, identity, field}
    return findall(isequal(o), dropdims(sum(M.mexp,dims=1),dims=1))
end

@doc raw"""
    X = zero(M::DensePolyManifold)

Create a representation of a zero polynomial with manifold structure `M`.
"""
function zero(M::DensePolyManifold{ndim, n, order, identity, ℝ}) where {ndim, n, order, identity, field}
    return zeros(n, length(M.admissible))
end

function zero(M::DensePolyManifold{ndim, n, order, identity, ℂ}) where {ndim, n, order, identity, field}
    return zeros(ComplexF64, n, length(M.admissible))
end

function zeroJacobianSquared(M::DensePolyManifold{ndim, n, order, identity, ℝ}) where {ndim, n, order, identity, field}
    return zeros(ndim, ndim, length(M.admissible))
end

function randn(M::DensePolyManifold{ndim, n, order, identity, ℝ}) where {ndim, n, order, identity, field}
    return randn(n, length(M.admissible))
end

function randn(M::DensePolyManifold{ndim, n, order, identity, ℂ}) where {ndim, n, order, identity, field}
    return randn(ComplexF64, n, length(M.admissible))
end

function toReal(M::DensePolyManifold{ndim, n, order, identity, ℂ}, X) where {ndim, n, order, identity}
    return DensePolyManifold{ndim, n, order, identity, ℝ}(M.mexpALL, M.mexp, M.admissible, Euclidean(n, length(M.admissible); field=ℝ), M.R, M.VT), real.(X)
end

function toComplex(M::DensePolyManifold{ndim, n, order, identity, ℝ}, X) where {ndim, n, order, identity}
    return DensePolyManifold{ndim, n, order, identity, ℂ}(M.mexpALL, M.mexp, M.admissible, Euclidean(n, length(M.admissible); field=ℂ), M.R, M.VT), Complex.(X)
end

function copySome!(Mo::DensePolyManifold, Xo, range, Mi::DensePolyManifold, Xi)
    # Mo and Mi must have the same number of variables
    @assert size(Mo.mexp,1) == size(Mi.mexp,1) "Mo and Mi must have the same number of variables!"
    for k=1:size(Mi.mexp,2)
        id = PolyFindIndex(Mo.mexp, Mi.mexp[:,k])
        Xo[range,id] .= Xi[:,k]
    end
    return nothing
end
@doc raw"""
    M, X = LinearDensePolynomial(A::AbstractArray{T,2})

Creates a linear polynomial, which is the same as matrix `A`.
"""
function LinearDensePolynomial(A::AbstractArray{Complex{T},2}) where T
    M = DensePolyManifold(size(A,2), size(A,1), 1, field = ℂ)
    X = zero(M)
    setLinearPart!(M, X, A)
    return M, X
end

function zero_tangent(M::DensePolyManifold{ndim, n, order, identity, ManifoldsBase.RealNumbers}) where {ndim, n, order, identity, field}
    return zeros(n, length(M.admissible))
end

function zero_tangent(M::DensePolyManifold{ndim, n, order, identity, ManifoldsBase.ComplexNumbers}) where {ndim, n, order, identity, field}
    return zeros(ComplexF64, n, length(M.admissible))
end

function zero_vector!(M::DensePolyManifold{ndim, n, order, identity, field}, X, p) where {ndim, n, order, identity, field}
    return zero_vector!(M.M, X, p)
end

function manifold_dimension(M::DensePolyManifold{ndim, n, order, identity, field}) where {ndim, n, order, identity, field}
    return n * length(M.admissible)
end

function inner(M::DensePolyManifold{ndim, n, order, identity, field}, p, X, Y) where {ndim, n, order, identity, field}
    return inner(M.M, p, X, Y)
end

function toFullDensePolynomial(M::DensePolyManifold{ndim, n, order, identity, field}, X) where {ndim, n, order, identity, field}
    Mnew = DensePolyManifold(ndim, n, order; min_order = 0, field = field)
    Xnew = zero(Mnew)
    for k=1:size(M.mexp,2)
        Xnew[:,PolyFindIndex(M.mexp, Mnew.mexp[:,k])] .= X[:,k]
    end
    return Mnew, Xnew
end

function setLinearPart!(M::DensePolyManifold{ndim, n, order, identity, field}, X, B) where {ndim, n, order, identity, field}
    linid = findall(isequal(1), dropdims(sum(M.mexp,dims=1),dims=1))
    for k=1:length(linid)
        @views id = findfirst(isequal(1), M.mexp[:,linid[k]])
        X[:,linid[k]] .= B[:,k]
    end
    return nothing
end

function getLinearPart(M::DensePolyManifold{ndim, n, order, identity, field}, X) where {ndim, n, order, identity, field}
    B = zeros(size(X,1), size(M.mexp,1))
    linid = findall(isequal(1), dropdims(sum(M.mexp,dims=1),dims=1))
    for k=1:length(linid)
        @views id = findfirst(isequal(1), M.mexp[:,linid[k]])
        B[:,k] .= X[:,linid[k]]
    end
    return B
end

function retract!(M::DensePolyManifold{ndim, n, order, identity, field}, q, p, X, method::ExponentialRetraction) where {ndim, n, order, identity, field}
    q .= p .+ X
    return q
end

function retract(M::DensePolyManifold{ndim, n, order, identity, field}, p, X, method::ExponentialRetraction) where {ndim, n, order, identity, field}
    return p .+ X
end

function project!(M::DensePolyManifold{ndim, n, order, identity, field}, Y, q, X) where {ndim, n, order, identity, field}
    Y .= X
    return Y
end

function vector_transport_to!(M::DensePolyManifold{ndim, n, order, identity, field}, Y, p, X, q, method::AbstractVectorTransportMethod) where {ndim, n, order, identity, field}
#     println("DensePoly VECTOR TRANSPORT 1")
    return vector_transport_to!(M.M, Y, p, X, q, method)
end

function vector_transport_to(M::DensePolyManifold{ndim, n, order, identity, field}, p, X, q, method::AbstractVectorTransportMethod) where {ndim, n, order, identity, field}
#     println("DensePoly VECTOR TRANSPORT 1")
    return vector_transport_to(M.M, p, X, q, method)
end

function HessFullProjection(M::DensePolyManifold{ndim, n, order, identity, field}, X, grad, hess) where {ndim, n, order, identity, field}
    return hess
end

function DenseMonomials(M::DensePolyManifold{ndim, n, order, identity, field}, X, data) where {ndim, n, order, identity, field}
#     @show size(reshape(M.mexp, ndim, :, 1))
#     @show size(data), ndim
#     @show size(reshape(data, ndim, 1, :))
    zp = reshape(data, ndim, 1, :) .^ reshape(M.mexp, ndim, :, 1)
    return dropdims(prod(zp, dims=1), dims=1)
end

function DenseMonomialsALL(M::DensePolyManifold{ndim, n, order, identity, field}, X, data) where {ndim, n, order, identity, field}
#     @show size(reshape(M.mexp, ndim, :, 1))
#     @show size(data), ndim
#     @show size(reshape(data, ndim, 1, :))
    zp = reshape(data, ndim, 1, :) .^ reshape(M.mexpALL, ndim, :, 1)
    return dropdims(prod(zp, dims=1), dims=1)
end

function Eval(M::DensePolyManifold{ndim, n, order, false, field}, X, data, L0 = nothing) where {ndim, n, order, field}
    # only use admissible monomials
    if L0 == nothing
        return X * DenseMonomials(M, X, data[1])
    else
        return sum(L0 .* (X * DenseMonomials(M, X, data[1])))
    end
    nothing
end

# with identity added
function Eval(M::DensePolyManifold{ndim, n, order, true, field}, X, data, L0 = nothing) where {ndim, n, order, field}
    # only use admissible monomials
    if L0 == nothing
        return X * DenseMonomials(M, X, data[1]) .+ data[1]
    else
        return sum(L0 .* (X * DenseMonomials(M, X, data[1]) .+ data[1]))
    end
    nothing
end

# returns the first index of mexp, which equals to iexp
# defined in polymethods.jl
function PolyFindIndex(mexp, iexp)
    return findfirst(dropdims(prod(mexp .== iexp, dims=1),dims=1))
end

#-------------------------------------------------
# FUNCTION POLYNOMIAL EXPANSION
#-------------------------------------------------

@doc raw"""
    X = fromFunction(M::DensePolyManifold, fun)

Taylor expands Julia function `fun` to a polynomial, whose strcuture is prescribed by `M`.
"""
function fromFunction(M::DensePolyManifold{ndim, n, order, identity, field}, fun) where {ndim, n, order, identity, field}
    W = zero(M)
	x = set_variables("x", numvars=ndim, order=order)
    y = fun(x)
	for k=1:ndim
		for i = 1:size(M.mexp,2)
	    	W[k,i] = getcoeff(y[k], M.mexp[:,i])
	     end
	end
    return W
end

#-------------------------------------------------
# POLYNOMIAL SUBSTITUTION
#-------------------------------------------------

@doc raw"""
    tab = mulTable(oexp, in1exp, in2exp)
    
Creates a list of triplets `[i1, i2, o]`, where `i1`, `i2` are the input monomial, which produce `o` output monomial.
"""
function mulTable(oexp, in1exp, in2exp)
    res = []
    od = maximum(sum(oexp,dims=1))
    p1 = sum(in1exp,dims=1)
    p2 = sum(in2exp,dims=1)
    pexp = zero(in1exp[:,1])
    for k1=1:size(in1exp,2)
        for k2=1:size(in2exp,2)
            if p1[k1]+p2[k2] <= od
                pexp[:] = in1exp[:,k1] + in2exp[:,k2]
                idx = PolyFindIndex(oexp, pexp)
                push!(res, [k1,k2,idx])
            end
        end
    end
    out = zeros(typeof(od), length(res), 3)
    for k = 1:length(res)
        out[k,:] = res[k]
    end
    return out
end

@doc raw"""
    PolyMul!(out, in1, in2, multab)
    
Multiplies two scalar valued polynomials `in1`, `in2` and adds the result to `out`. 
All inputs are one-dimensional arrays, `multab` is produced by `mulTable`.
"""
function PolyMul!(out::AbstractArray{T,1}, in1::AbstractArray{T,1}, in2::AbstractArray{T,1}, multab) where T
    for k = 1:size(multab,1)
        out[multab[k,3]] += in1[multab[k,1]]*in2[multab[k,2]]
    end
    return nothing
end

function DensePolySubstituteTab!(Mout::DensePolyManifold, Xout, M1::DensePolyManifold, X1, M2::DensePolyManifold, X2)
    return mulTable(Mout.mexp, Mout.mexp, M2.mexp)
end

function DensePolySubstitute!(Mout::DensePolyManifold, Xout, M1::DensePolyManifold, X1, M2::DensePolyManifold, X2, tab)
    to = zeros(eltype(Xout), size(Mout.mexp,2)) # temporary for d an k
    res = zeros(eltype(Xout), size(Mout.mexp,2)) # temporary for d an k
    # index of constant in the output
    Xout .= 0
    cfc = findfirst(dropdims(sum(Mout.mexp, dims=1),dims=1) .== 0)
    # substitute into all monomials
    for d = 1:size(X1,1) # all dimensions
        for k = 1:size(M1.mexp,2) # all monomials
            to .= 0
            to[cfc] = X1[d, k] # the constant coefficient
            # l select the variable in the monomial
            for l = 1:size(M1.mexp,1)
                # multiply the exponent times
                for p = 1:M1.mexp[l,k]
                    # should not add to the previous result
                    res .= 0
                    @views PolyMul!(res, to, X2[l,:], tab)
                    to[:] .= res
                end
            end
            Xout[d,:] .+= to
        end
    end
    return nothing
end

function DensePolySubstitute!(Mout::DensePolyManifold, Xout, M1::DensePolyManifold, X1, M2::DensePolyManifold, X2)
    tab = mulTable(Mout.mexp, Mout.mexp, M2.mexp)
    DensePolySubstitute!(Mout, Xout, M1, X1, M2, X2, tab)
    return nothing
end

#-------------------------------------------------
# POLYNOMIAL DIFFERENTIATION
#-------------------------------------------------

@doc raw"""
    DM = DensePolyDeriMatrices(to::DensePolyManifold, from::DensePolyManifold)

creates a set of sparse matrices, for each input variable, that maps the monimials of `from` to the monomials to `to`.
Differentiation with respect to the `k`-the variable is `Xto = Xfrom * DM[k]`, 
where `Xfrom` is the the polynomial to differentiate, and `Xto` is the derivative.
"""
function DensePolyDeriMatrices(to::DensePolyManifold, from::DensePolyManifold)
    MC = []
    for var = 1:size(from.mexpALL,1)
        I = Array{Int,1}(undef,0)
        J = Array{Int,1}(undef,0)
        V = Array{Int,1}(undef,0)
        for k = 1:size(from.mexpALL,2)
            id = copy(from.mexpALL[:,k]) # this is a copy not a view
            if id[var] > 0
                id[var] -= 1
                x = PolyFindIndex(to.mexpALL, id)
                if x != nothing
                    push!(I, k)
                    push!(J, x)
                    push!(V, from.mexpALL[var,k])
                end
            end
        end
        MV = sparse(I, J, V, size(from.mexpALL,2), size(to.mexpALL,2))
        push!(MC, MV)
    end
#     @show MC
    return MC
end

function DensePolyDeriMul!(Mout::DensePolyManifold, Xout, M1::DensePolyManifold, X1, M2::DensePolyManifold, X2, multab, DM)
    # this is a matrix-vector multiplication
    for k = 1:size(X1,1) # number of rows in input
        for l = 1:size(M1.mexpALL,1) # number of variables in input, hence columns of derivative
            # out[k] = sum deri[k,l]*in2[l]
            # deri is the derivative of in1[k] with respect to l
            @views deri = DM[l]' * X1[k,:]
            @views PolyMul!(Xout[k,:], vec(deri), X2[l,:], multab)
        end
    end
    return nothing
end

function DensePolyDeriMul!(Mout::DensePolyManifold, Xout, M1::DensePolyManifold, X1, M2::DensePolyManifold, X2)
    multab = mulTable(Mout.mexp, Mout.mexp, M2.mexp)
    DM = DensePolyDeriMatrices(Mout, M1)
    DensePolyDeriMul!(Mout, Xout, M1, X1, M2, X2, multab, DM)
    return nothing
end


@doc raw"""
    (P)^T P is a scalar
"""
function DensePolySquared!(Mout::DensePolyManifold, Xout, Min::DensePolyManifold, Xin)
    multab = mulTable(Mout.mexp, Min.mexp, Min.mexp)
    Xout .= 0
    # this is a matrix-vector multiplication
    for p = 1:size(Xin,1) # number of rows in input
        @views PolyMul!(Xout[1,:], Xin[p,:], Xin[p,:], multab)
    end
    return nothing
end

@doc raw"""
    (D_P)^T P is a vector
"""
function DensePolyDeriTransposeMul!(Mout::DensePolyManifold, Xout, Min::DensePolyManifold, Xin)
    multab = mulTable(Mout.mexp, Min.mexp, Min.mexp)
    DM = DensePolyDeriMatrices(Min, Min)
    Xout .= 0
    # this is a matrix-vector multiplication
    for k = 1:size(Min.mexpALL,1)
        for p = 1:size(Xin,1) # number of rows in input
            @views deri_k = DM[k]' * Xin[p,:]
            @views PolyMul!(Xout[k,:], vec(deri_k), Xin[p,:], multab)
        end
    end
    return nothing
end


@doc raw"""
    (D_P)^T D_P is a matrix
    
    make note that the output type as a manifold does not exist, so care needs to be taken.
    It is assumed that the dimensionality is the same as the number of variables of Mout.
"""
function DensePolyJabobianSquared!(Mout::DensePolyManifold, Xout, Min::DensePolyManifold, Xin)
    multab = mulTable(Mout.mexp, Min.mexp, Min.mexp)
    DM = DensePolyDeriMatrices(Min, Min)
    Xout .= 0
    # this is a matrix-matrix multiplication
    for k = 1:size(Min.mexpALL,1)
        for l = 1:size(Min.mexpALL,1) # number of variables in input, hence columns of derivative
            for p = 1:size(Xin,1) # number of rows in input
                @views deri_k = DM[k]' * Xin[p,:]
                @views deri_l = DM[l]' * Xin[p,:]
                @views PolyMul!(Xout[k,l,:], vec(deri_k), vec(deri_l), multab)
            end
        end
    end
    return nothing
end

#-------------------------------------------------
# POLYNOMIAL TRANSFORMATION
#-------------------------------------------------

# mexp: exponents
# inp: the polynomial
# T: the linear transformation

@doc raw"""
    ModelLinearTransform!(out::PolyModel{T}, m::PolyModel{T}, Tran::AbstractArray{T,2}) where T


"""
function DensePolyLinearTransform!(Mout::DensePolyManifold, Xout, Min::DensePolyManifold, Xin, Tran::AbstractArray{T,2}) where T
    Mtr, Xtr = LinearDensePolynomial(Tran)
    DensePolySubstitute!(Mout, Xout, Min, Xin, Mtr, Xtr)
    Xout .= Tran\Xout
    return nothing
end

function DensePolyLinearTransform!(Mout::DensePolyManifold, Xout, Min::DensePolyManifold, Xin, Tran::AbstractArray{T,2}, TranBack::AbstractArray{T,2}) where T
    Mtr, Xtr = LinearDensePolynomial(Tran)
    DensePolySubstitute!(Mout, Xout, Min, Xin, Mtr, Xtr)
    Xout .= TranBack*Xout
    return nothing
end

#-------------------------------------------------
# FOR OPTIMISATION
#-------------------------------------------------

function Jacobian(M::DensePolyManifold{ndim, n, order, false, field}, X, data) where {ndim, n, order, identity, field}
    DM = DensePolyDeriMatrices(M, M)
    mon = DenseMonomialsALL(M, X, data)
    res = zeros(size(X,1), length(DM), size(mon,2))
    for k=1:length(DM)
#         @show size(Array(DM[k])), size(mon)
        res[:,k,:] .= X * (DM[k] * mon)[M.admissible,:]
    end
    return res
end

# with identity added
function Jacobian(M::DensePolyManifold{ndim, n, order, true, field}, X, data) where {ndim, n, order, identity, field}
    DM = DensePolyDeriMatrices(M, M)
    mon = DenseMonomialsALL(M, X, data)
    res = zeros(size(X,1), length(DM), size(mon,2))
    for k=1:length(DM)
#         @show size(Array(DM[k])), size(mon)
        res[:,k,:] .= X * (DM[k] * mon)[M.admissible,:] 
    end
    for k=1:size(res,3)
        for l=1:size(res,2)
            res[l,l,k] += 1
        end
    end
    return res
end

function Hessian(M::DensePolyManifold{ndim, n, order, identity, field}, X, data) where {ndim, n, order, identity, field}
    hess = zeros(n, ndim, ndim, size(data,2))
    DM = DensePolyDeriMatrices(M, M)
    mon = DenseMonomialsALL(M, X, data)
    for k=1:length(DM), l=1:length(DM)
        hess[:,k,l,:] .= X * (DM[l]*DM[k] * mon)[M.admissible,:]
    end
    return hess
end

function L0_DF(M::DensePolyManifold{ndim, n, order, identity, field}, X, DV, data, L0, ii) where {ndim, n, order, identity, field}
    mon = DenseMonomials(M, X, data)
    return dropdims(sum(reshape(L0,size(L0,1),1,size(L0,2)) .* reshape(mon, 1, size(mon,1), size(mon,2)), dims=3), dims=3)
end

function XO_mon_mon!(XO, mon, scale)
    for q=1:size(XO,1), p1=1:size(XO,2), p2=1:size(XO,4), l=1:size(mon,2)
        @inbounds XO[q,p1,q,p2] += mon[p1,l] * mon[p2,l] / scale[l]
    end
    nothing
end

function DFoxT_DFox(M::DensePolyManifold{ndim, n, order, identity, field}, X, data, ii; scale = alwaysone()) where {ndim, n, order, identity, field}
    XO = zeros(n, size(M.mexp,2), n, size(M.mexp,2))
    mon = DenseMonomials(M, X, data)
#     @time for q=1:min(ndim,n), l=1:size(data,2), p1=1:size(M.mexp,2), p2=1:size(M.mexp,2)
    XO_mon_mon!(XO, mon, scale)
    return XO
end

function testDensePoly()
    din = 3
    dout = 2
    M = DensePolyManifold(din, dout, 4; min_order = 2) # out dim=2 in dim=3, order 4
    X = randn(M)
    dataIN = randn(din, 100)
    dataOUT = randn(dout, 100)
    
    # Jacobian
    println("DensePolyManifold Jacobian")
    res_orig = Jacobian(M, X, dataIN)
    eps = 1e-7
    res = deepcopy(res_orig)
    dataINp = deepcopy(dataIN)
    for k=1:size(dataIN,1)
        dataINp[k,:] .+= eps
        res[:,k,:] = (Eval(M, X, [dataINp]) - Eval(M, X, [dataIN])) / eps
        dataINp[k,:] = dataIN[k,:]
    end
    @show maximum(abs.(res_orig .- res))
    
    # Hessian
    println("DensePolyManifold Hessian")
    dataINp = deepcopy(dataIN)
    eps = 1e-6
    hessU = Hessian(M, X, dataIN)
    hessUp = deepcopy(hessU)
    for k=1:size(dataIN,1)
        dataINp[k,:] .+= eps
        hessUp[:,:,k,:] = (Jacobian(M, X, dataINp) .- Jacobian(M, X, dataIN)) / eps
        dataINp[k,:] .= dataIN[k,:]
    end
    @show maximum(abs.(hessU .- hessUp))
    
    grad = L0_DF(M, X, nothing, dataIN, dataOUT, nothing)
    xp = deepcopy(X)
    gradp = deepcopy(grad)
    eps = 1e-6
    flag = false
    for l=1:length(X)
        xp[l] += eps
        gradp[l] = sum(Eval(M, xp, [dataIN], dataOUT) - Eval(M, X, [dataIN], dataOUT)) / eps
        relErr = (gradp[l] - grad[l]) / grad[l]
        if abs(relErr) > 1e-4
            flag = true
        end
        xp[l] = X[l]
    end
    if flag
        println("DensePolyManifold L0_DF")
        @show diff = gradp - grad
    end
end
