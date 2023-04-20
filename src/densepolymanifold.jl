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
    DM
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
    DM = DensePolyDeriMatrices(mexpALL, mexpALL)
    admissible = findall(dropdims(sum(mexpALL, dims=1), dims=1) .>= min_order)
    return DensePolyManifold{ndim, n, order, identity, field}(mexpALL, DM, mexpALL[:,admissible], admissible, Euclidean(n, length(admissible); field), ExponentialRetraction(), ParallelTransport())
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
function zero(M::DensePolyManifold{ndim, n, order, identity, ℝ}) where {ndim, n, order, identity}
    return zeros(n, length(M.admissible))
end

function zero(M::DensePolyManifold{ndim, n, order, identity, ℂ}) where {ndim, n, order, identity}
    return zeros(ComplexF64, n, length(M.admissible))
end

# take into account mexpALL monomials
function zeroALL(M::DensePolyManifold{ndim, n, order, identity, ℝ}) where {ndim, n, order, identity}
    return zeros(n, size(M.mexpALL,2))
end

function zeroALL(M::DensePolyManifold{ndim, n, order, identity, ℂ}) where {ndim, n, order, identity}
    return zeros(ComplexF64, n, size(M.mexpALL,2))
end

function zeroJacobianSquared(M::DensePolyManifold{ndim, n, order, identity, ℝ}) where {ndim, n, order, identity}
    return zeros(ndim, ndim, length(M.admissible))
end

function randn(M::DensePolyManifold{ndim, n, order, identity, ℝ}) where {ndim, n, order, identity}
    return randn(n, length(M.admissible))
end

function randn(M::DensePolyManifold{ndim, n, order, identity, ℂ}) where {ndim, n, order, identity}
    return randn(ComplexF64, n, length(M.admissible))
end

function toReal(M::DensePolyManifold{ndim, n, order, identity, ℂ}, X) where {ndim, n, order, identity}
    return DensePolyManifold{ndim, n, order, identity, ℝ}(M.mexpALL, M.DM, M.mexp, M.admissible, Euclidean(n, length(M.admissible); field=ℝ), M.R, M.VT), real.(X)
end

function toComplex(M::DensePolyManifold{ndim, n, order, identity, ℝ}, X) where {ndim, n, order, identity}
    return DensePolyManifold{ndim, n, order, identity, ℂ}(M.mexpALL, M.DM, M.mexp, M.admissible, Euclidean(n, length(M.admissible); field=ℂ), M.R, M.VT), Complex.(X)
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

function zero_tangent(M::DensePolyManifold{ndim, n, order, identity, ManifoldsBase.RealNumbers}) where {ndim, n, order, identity}
    return zeros(n, length(M.admissible))
end

function zero_tangent(M::DensePolyManifold{ndim, n, order, identity, ManifoldsBase.ComplexNumbers}) where {ndim, n, order, identity}
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
    admissible = 1:size(M.mexpALL,2)
    Mnew = DensePolyManifold{ndim, n, order, false, field}(M.mexpALL, M.DM, M.mexpALL, admissible, Euclidean(n, length(admissible); field = field), ExponentialRetraction(), ParallelTransport())
    Xnew = zero(Mnew)
    for k=1:size(M.mexp,2)
        Xnew[:,PolyFindIndex(M.mexp, Mnew.mexp[:,k])] .= X[:,k]
    end
    # putting the identity back if necessary
    if identity
        linid = findall(isequal(1), dropdims(sum(M.mexp,dims=1),dims=1))
        for k=1:length(linid)
            @views id = findfirst(isequal(1), M.mexp[:,linid[k]])
            X[k,linid[k]] += one(eltype(Xnew))
        end
    end
    return Mnew, Xnew
end

function setConstantPart!(M::DensePolyManifold{ndim, n, order, identity, field}, X, C) where {ndim, n, order, identity, field}
    cid = findfirst(isequal(0), dropdims(sum(M.mexp,dims=1),dims=1))
    if cid != nothing
        X[:,cid] .= C
    end
    return nothing
end

function getConstantPart(M::DensePolyManifold{ndim, n, order, identity, field}, X) where {ndim, n, order, identity, field}
    cid = findfirst(isequal(0), dropdims(sum(M.mexp,dims=1),dims=1))
    if cid != nothing
        return X[:,cid]
    end
    return zero(X[:,1])
end

function setLinearPart!(M::DensePolyManifold{ndim, n, order, identity, field}, X, B) where {ndim, n, order, identity, field}
    linid = findall(isequal(1), dropdims(sum(M.mexp,dims=1),dims=1))
    for k=1:length(linid)
        @views id = findfirst(isequal(1), M.mexp[:,linid[k]])
        X[:,linid[k]] .= B[1:size(X,1),k]
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

function retract!(M::DensePolyManifold{ndim, n, order, identity, field}, q, p, X, t::Number, method::ExponentialRetraction) where {ndim, n, order, identity, field}
    q .= p .+ t .* X
    return q
end

function retract(M::DensePolyManifold{ndim, n, order, identity, field}, p, X, method::ExponentialRetraction) where {ndim, n, order, identity, field}
    return p .+ X
end

function retract(M::DensePolyManifold{ndim, n, order, identity, field}, p, X, t::Number, method::AbstractRetractionMethod) where {ndim, n, order, identity, field}
    return p .+ t .* X
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

function DenseMonomials(ndim, mexp, data)
    @assert ndim == size(data,1) "Incompatible dimensions: ndim=$(ndim) != size(data,1)=$(size(data,1))"
    zp = reshape(data, ndim, 1, :) .^ reshape(mexp, ndim, :, 1)
    return dropdims(prod(zp, dims=1), dims=1)
end

function DenseMonomials(M::DensePolyManifold{ndim, n, order, false, field}, data) where {ndim, n, order, field}
    return DenseMonomials(ndim, M.mexpALL, data)
end

function Eval_MON(M::DensePolyManifold{ndim, n, order, false, field}, X, MON) where {ndim, n, order, field}
    return X * MON
end

function Eval_MON(M::DensePolyManifold{ndim, n, order, true, field}, X, MON) where {ndim, n, order, field}
    return X * MON .+ data
end

function Eval(M::DensePolyManifold{ndim, n, order, false, field}, X, data; L0 = nothing, DV=nothing) where {ndim, n, order, field}
    # only use admissible monomials
    if L0 == nothing
        return X * DenseMonomials(ndim, M.mexp, data)
    else
        return sum(L0 .* (X * DenseMonomials(ndim, M.mexp, data)))
    end
    nothing
end

# with identity added
function Eval(M::DensePolyManifold{ndim, n, order, true, field}, X, data; L0 = nothing, DV=nothing) where {ndim, n, order, field}
    # only use admissible monomials
    if L0 == nothing
        return X * DenseMonomials(ndim, M.mexp, data) .+ data
    else
        return sum(L0 .* (X * DenseMonomials(ndim, M.mexp, data) .+ data))
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
	for k=1:n
		for i = 1:size(M.mexp,2)
	    	W[k,i] = getcoeff(y[k], M.mexp[:,i])
	     end
	end
    return W
end

function fromData(M::DensePolyManifold{ndim, n, order, identity, field}, dataIN, dataOUT) where {ndim, n, order, identity, field}
    phi = DenseMonomials(ndim, M.mexp, dataIN)
    G = phi*transpose(phi)
    H = dataOUT*transpose(phi)
    X = H/G
    return X
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
    DM = DensePolyDeriMatrices(to_mexpALL, from_mexpALL)

creates a set of sparse matrices, for each input variable, that maps the monimials of `from` to the monomials to `to`.
Differentiation with respect to the `k`-the variable is `Xto = Xfrom * DM[k]`, 
where `Xfrom` is the the polynomial to differentiate, and `Xto` is the derivative.
"""
function DensePolyDeriMatrices(to_mexpALL, from_mexpALL)
    MC = []
    for var = 1:size(from_mexpALL,1)
        I = Array{Int,1}(undef,0)
        J = Array{Int,1}(undef,0)
        V = Array{Int,1}(undef,0)
        for k = 1:size(from_mexpALL,2)
            id = copy(from_mexpALL[:,k]) # this is a copy not a view
            if id[var] > 0
                id[var] -= 1
                x = PolyFindIndex(to_mexpALL, id)
                if x != nothing
                    push!(I, k)
                    push!(J, x)
                    push!(V, from_mexpALL[var,k])
                end
            end
        end
        MV = sparse(I, J, V, size(from_mexpALL,2), size(to_mexpALL,2))
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
    DM = DensePolyDeriMatrices(Mout.mexpALL, M1.mexpALL)
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
    DM = DensePolyDeriMatrices(Min.mexpALL, Min.mexpALL)
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
    DM = DensePolyDeriMatrices(Min.mexpALL, Min.mexpALL)
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

function Jacobian_MON(M::DensePolyManifold{ndim, n, order, false, field}, X, mon) where {ndim, n, order, field}
    res = zeros(size(X,1), length(M.DM), size(mon,2))
    for k=1:length(M.DM)
#         @show size(Array(DM[k])), size(mon)
        res[:,k,:] .= X * (M.DM[k] * mon)[M.admissible,:]
    end
    return res
end

function Jacobian_DMON(M::DensePolyManifold{ndim, n, order, false, field}, X, DMON) where {ndim, n, order, field}
#     @tullio res[i,k,j] := X[i,p] * DMON[p,k,j]
    @tullio res[i,j,k] := X[i,p] * DMON[p,j,k]
    return res
end

function Jacobian(M::DensePolyManifold{ndim, n, order, false, field}, X, data) where {ndim, n, order, field}
    mon = DenseMonomials(ndim, M.mexpALL, data)
    res = zeros(size(X,1), length(M.DM), size(mon,2))
    for k=1:length(M.DM)
#         @show size(Array(DM[k])), size(mon)
        res[:,k,:] .= X * (M.DM[k] * mon)[M.admissible,:]
    end
    return res
end

# with identity added
function Jacobian(M::DensePolyManifold{ndim, n, order, true, field}, X, data) where {ndim, n, order, field}
    mon = DenseMonomials(ndim, M.mexpALL, data)
    res = zeros(size(X,1), length(M.DM), size(mon,2))
    for k=1:length(M.DM)
#         @show size(Array(DM[k])), size(mon)
        res[:,k,:] .= X * (M.DM[k] * mon)[M.admissible,:] 
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
    mon = DenseMonomials(ndim, M.mexpALL, data)
    for k=1:length(M.DM), l=1:length(M.DM)
        hess[:,k,l,:] .= X * (M.DM[l]*M.DM[k] * mon)[M.admissible,:]
    end
    return hess
end

# derivatives of the combination
# L0  = P1 o Up o x - P2 o Up o y
# L   = L0^T L0
# L   = (P1 o Up o x)^T (P1 o Up o x) + (P2 o Up o y)^T (P2 o Up o y) - 2*(P1 o Up o x)^T (P2 o Up o y)
# dL  = 2 (P1 o Up o x)^T (DP1 o Up o x) dUp o x + 2 (P2 o Up o y)^T (DP2 o Up o y) dUp o y - 2 (P1 o Up o x)^T (DP2 o Up o y) dUp o y - 2 (P2 o Up o y)^T (DP1 o Up o x) dUp o x
# dL  = 2 [(P1 o Up o x)^T (DP1 o Up o x) - (P2 o Up o y)^T (DP1 o Up o x)] dUp o x
#      +2 [(P2 o Up o y)^T (DP2 o Up o y) - (P1 o Up o x)^T (DP2 o Up o y)] dUp o y
# d2L = 2 [(P1 o Up o x)^T (D2P1 o Up o x) dUp o x + (DP1 o Up o x . dUp o x)^T (DP1 o Up o x) - (P2 o Up o y)^T (D2P1 o Up o x) dUp o x - (DP2 o Up o y . dUp o y)^T (DP1 o Up o x)] dUp o x
#      +2 [(P2 o Up o y)^T (D2P2 o Up o y) dUp o y + (DP2 o Up o y . dUp o y)^T (DP2 o Up o y) - (P1 o Up o x)^T (D2P2 o Up o y) dUp o y - (DP1 o Up o x . dUp o x)^T (DP2 o Up o y)] dUp o y
# d2L = 2 [(P1 o Up o x)^T (D2P1 o Up o x) + (DP1 o Up o x)^T (DP1 o Up o x) - (P2 o Up o y)^T (D2P1 o Up o x)]{dUp o x, dUp o x}
#      +2 [(P2 o Up o y)^T (D2P2 o Up o y) + (DP2 o Up o y)^T (DP2 o Up o y) - (P1 o Up o x)^T (D2P2 o Up o y)]{dUp o y, dUp o y}
#      -4 [(DP1 o Up o x)^T (DP2 o Up o y)] {dUp o x, dUp o y}
# what to return:
# [(P1 o Up o x)^T (D2P1 o Up o x) + (DP1 o Up o x)^T (DP1 o Up o x) - (P2 o Up o y)^T (D2P1 o Up o x)] = F1F1_D2D2 + F1F1_D1D2 - F2F1_D2D2 
#                                                                                                       = F1F1_D2D2 + F1F1_D1D2 - F1F2_D1D1
# [(P2 o Up o y)^T (D2P2 o Up o y) + (DP2 o Up o y)^T (DP2 o Up o y) - (P1 o Up o x)^T (D2P2 o Up o y)] = F2F2_D2D2 + F2F2_D1D2 - F1F2_D2D2
# - 2 [(DP1 o Up o x)^T (DP2 o Up o y)]                                                                 = -2 F1F2_D1D2
function HessianCombination(M1::DensePolyManifold{ndim, n, order1, identity1, field}, X1, M2::DensePolyManifold{ndim, n, order2, identity2, field}, X2, data1, data2) where {ndim, n, order1, identity1, order2, identity2, field}
    M1f, X1f = toFullDensePolynomial(M1, X1)
    M2f, X2f = toFullDensePolynomial(M2, X2)
    
    X1X1 = transpose(X1f) * X1f
    X2X2 = transpose(X2f) * X2f
    X1X2 = transpose(X1f) * X2f

    DM1 = DensePolyDeriMatrices(M1f.mexpALL, M1f.mexpALL)
    DM2 = DensePolyDeriMatrices(M2f.mexpALL, M2f.mexpALL)
    mon1 = DenseMonomials(ndim, M1.mexpALL, data1)
    mon2 = DenseMonomials(ndim, M2.mexpALL, data2)
    H11 = zeros(eltype(X1X1), ndim, ndim, size(data1,2))
    H22 = zeros(eltype(X2X2), ndim, ndim, size(data2,2))
    H12 = zeros(eltype(X2X2), ndim, ndim, size(data1,2))
    @show size(mon1), size(mon2), size(X1X1), size(X2X2), size(X1X2)
    for k=1:length(DM1), l=1:length(DM1)
#         F1F1_D2D2[l,k,:] = transpose(mon1) * X1X1 * (DM1[l]*DM1[k] * mon1)
#         F1F1_D1D2[l,k,:] = transpose(DM1[l] * mon1) * X1X1 * (DM1[k] * mon1)
#         F1F2_D1D1[l,k,:] = transpose(DM1[l]*DM1[k] * mon1) * X1X2 * mon2
        H11[k,l,:] .= dropdims(sum(mon1 .* (X1X1 * (DM1[l]*DM1[k] * mon1)), dims=1) .+ sum((DM1[l] * mon1) .* (X1X1 * (DM1[k] * mon1)),dims=1) .- sum((DM1[l]*DM1[k] * mon1) .* (X1X2 * mon2),dims=1),dims=1)
    end
    for k=1:length(DM2), l=1:length(DM2)
#         F2F2_D2D2[l,k,:] = transpose(mon2) * X2X2 * (DM2[l]*DM2[k] * mon2)
#         F2F2_D1D2[l,k,:] = transpose(DM2[l] * mon2) * X2X2 * (DM2[k] * mon2)
#         F1F2_D2D2[l,k,:] = transpose(mon1) * X1X2 * (DM2[l]*DM2[k] * mon2)
        H22[k,l,:] .= dropdims(sum(mon2 .* (X2X2 * (DM2[l]*DM2[k] * mon2)),dims=1) .+ sum((DM2[l] * mon2) .* (X2X2 * (DM2[k] * mon2)),dims=1) .- sum(mon1 .* (X1X2 * (DM2[l]*DM2[k] * mon2)),dims=1),dims=1)
    end
    for k=1:length(DM1), l=1:length(DM2)
#         F1F2_D1D2[l,k,:] = transpose(DM1[l] * mon1) * X1X2 * (DM2[k] * mon2)
        H12[k,l,:] .= -2 * dropdims(sum((DM1[k] * mon1) .* (X1X2 * (DM2[l] * mon2)),dims=1),dims=1)
    end
    return H11, H22, H12
end

# With respect to the parameters
function L0_DF_MON(M::DensePolyManifold{ndim, n, order, identity, field}, X, MON; L0, ii = 0, DV = nothing) where {ndim, n, order, identity, field}
#     @tullio res[i,p] := L0[i,k] * MON[p,k]
    return L0 * transpose(MON)
#     return dropdims(sum(reshape(L0,size(L0,1),1,size(L0,2)) .* reshape(mon, 1, size(mon,1), size(mon,2)), dims=3), dims=3)
end

function L0_DF(M::DensePolyManifold{ndim, n, order, identity, field}, X, data; L0, ii = 0, DV = nothing) where {ndim, n, order, identity, field}
    mon = DenseMonomials(ndim, M.mexp, data)
    return dropdims(sum(reshape(L0,size(L0,1),1,size(L0,2)) .* reshape(mon, 1, size(mon,1), size(mon,2)), dims=3), dims=3)
end

function L0_JF(M::DensePolyManifold{ndim, n, order, false, field}, X, data; L0) where {ndim, n, order, field}
#     DM = DensePolyDeriMatrices(M.mexpALL, M.mexpALL) # reshuffles the monomials and multiplies with the right constant
    mon = DenseMonomials(ndim, M.mexpALL, data)
    res = zeros(length(M.DM), size(mon,2))
    L0X = transpose(X) * L0
    for k=1:length(M.DM)
#         @show size(res[k,:]), size(L0X), size((DM[k] * mon)[M.admissible,:])
        @views res[k,:] .+= dropdims(sum(L0X .* (M.DM[k] * mon)[M.admissible,:],dims=1),dims=1)
    end
    return res
end

function JF_dx(M::DensePolyManifold{ndim, n, order, false, field}, X, data, dx) where {ndim, n, order, field}
#     DM = DensePolyDeriMatrices(M.mexpALL, M.mexpALL) # reshuffles the monomials and multiplies with the right constant
    mon = DenseMonomials(ndim, M.mexpALL, data)
    res = zeros(size(X,1), size(mon,2))
    for k=1:length(M.DM)
        res .+= (X * (M.DM[k] * mon)[M.admissible,:]) .* reshape(dx[k,:],1,:)
    end
    return res
end

# with identity added
function JF_dx(M::DensePolyManifold{ndim, n, order, true, field}, X, data, dx) where {ndim, n, order, field}
#     DM = DensePolyDeriMatrices(M.mexpALL, M.mexpALL)
    mon = DenseMonomials(ndim, M.mexpALL, data)
    res = zeros(size(X,1), size(mon,2))
    for k=1:length(M.DM)
#         @show size(X * (DM[k] * mon)[M.admissible,:]), size(data), size(mon), size(dx)
        res .+= (X * (M.DM[k] * mon)[M.admissible,:]) .* reshape(dx[k,:],1,:)
    end
    res .+= dx
    return res
end

function L0_HF_dx(M::DensePolyManifold{ndim, n, order, identity, field}, X, data, L0, dx) where {ndim, n, order, identity, field}
#     DM = DensePolyDeriMatrices(M.mexpALL, M.mexpALL)
    mon = DenseMonomials(ndim, M.mexpALL, data)
#     @tullio hess[p,k] := L0[j,k] * X[j,l] * (DM[p]*DM[q] * mon)[M.admissible[l],k] * dx[q,k]
    # or
    hess = zeros(ndim, size(data,2))
    for p=1:length(M.DM), q=1:length(M.DM)
#         @show size((X * (DM[p]*DM[q] * mon)[M.admissible,:])), size(dx)
        hess[p,:] .+= dropdims(sum(L0 .* (X * (M.DM[p]*M.DM[q] * mon)[M.admissible,:]) .* reshape(dx[q,:],1,:),dims=1),dims=1)
    end
    return hess
end

function XO_mon_mon!(XO, mon, scale)
    for q=1:size(XO,1), p1=1:size(XO,2), p2=1:size(XO,4), l=1:size(mon,2)
        @inbounds XO[q,p1,q,p2] += mon[p1,l] * mon[p2,l] / scale[l]
    end
    nothing
end

function DFoxT_DFox(M::DensePolyManifold{ndim, n, order, identity, field}, X, data; scale = alwaysone()) where {ndim, n, order, identity, field}
    XO = zeros(n, size(M.mexp,2), n, size(M.mexp,2))
    mon = DenseMonomials(ndim, M.mexp, data)
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

# -------------------------------------------------------
# MATRIX VALUED MODELS AND HELPER FUNCTIONS
# -------------------------------------------------------

# This is a matrix valued polynomial whose constant part is Stiefel
# the representation is index-3
# ndim : number of variables
# n: rows, m: columns
# transpose: if true the retraction works on the 
# orthogonal: if true the constant part is a Stiefel manifold. The row vectors are 

# the retraction method is PolarRetraction, 
# the vector transport method is DifferentiatedRetractionVectorTransport{PolarRetraction}(PolarRetraction())

struct DenseMatrixPolynomial{ndim, n, m, order, 𝔽} <: AbstractManifold{𝔽}
    mexpALL
    DM
    mexp
    admissible
    M        :: AbstractManifold{𝔽} # Euclidean{Tuple{n, ndim}, 𝔽}
    R        :: AbstractRetractionMethod
    VT       :: AbstractVectorTransportMethod
end

function DenseMatrixPolynomial(ndim, n, m, order; min_order = 0, field::AbstractNumbers=ℝ)
    mexpALL = DensePolyExponents(ndim, order)
    DM = DensePolyDeriMatrices(mexpALL, mexpALL)
    admissible = findall(dropdims(sum(mexpALL, dims=1), dims=1) .>= min_order)
    return DenseMatrixPolynomial{ndim, n, m, order, field}(mexpALL, DM, mexpALL[:,admissible], admissible, Euclidean(n, m, length(admissible); field), ExponentialRetraction(), ParallelTransport())
end

function zero(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}) where {ndim, n, m, order}
    return zeros(n, m, length(M.admissible))
end

function randn(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}) where {ndim, n, m, order}
    return randn(n, m, length(M.admissible))
end

function zero_vector!(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}, X, p) where {ndim, n, m, order}
    return zero_vector!(M.M, X, p)
end

function manifold_dimension(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}) where {ndim, n, m, order}
    return manifold_dimension(M.M)
end

function inner(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}, p, X, Y) where {ndim, n, m, order}
    return inner(M.M, p, X, Y)
end

function retract!(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}, q, p, X, method::AbstractRetractionMethod) where {ndim, n, m, order}
    return retract!(M.M, q, p, X, method)
end

function retract!(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}, q, p, X, t::Number, method::AbstractRetractionMethod) where {ndim, n, m, order}
    return retract!(M.M, q, p, X, t, method)
end

function retract(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}, p, X, method::AbstractRetractionMethod) where {ndim, n, m, order}
    return retract(M.M, p, X, method)
end

function retract(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}, p, X, t::Number, method::AbstractRetractionMethod) where {ndim, n, m, order}
    return retract(M.M, p, X, t, method)
end

function vector_transport_to!(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}, Y, p, X, q, method::AbstractVectorTransportMethod) where {ndim, n, m, order}
    return vector_transport_to!(M.M, Y, p, X, q, method)
end

function vector_transport_to(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}, p, X, q, method::AbstractVectorTransportMethod) where {ndim, n, m, order}
#     println("DensePoly VECTOR TRANSPORT 1")
    return vector_transport_to(M.M, p, X, q, method)
end

function project!(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}, Y, q, X) where {ndim, n, m, order}
    return project!(M.M, Y, q, X)
end

# substitution
function DensePolySubstitute!(Mout::DenseMatrixPolynomial, Xout, M1::DenseMatrixPolynomial, X1, M2::DensePolyManifold, X2, tab)
    to = zeros(eltype(Xout), size(Mout.mexp,2)) # temporary for d an k
    res = zeros(eltype(Xout), size(Mout.mexp,2)) # temporary for d an k
    # index of constant in the output
    Xout .= zero(eltype(Xout))
    cfc = findfirst(dropdims(sum(Mout.mexp, dims=1),dims=1) .== 0)
    # substitute into all monomials
    for d1 = 1:size(X1,1), d2 = 1:size(X1,2) # all dimensions
        for k = 1:size(M1.mexp,2) # all monomials
            to .= 0
            to[cfc] = X1[d1, d2, k] # the constant coefficient
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
            Xout[d1, d2, :] .+= to
        end
    end
    return nothing
end

function DensePolySubstitute!(Mout::DenseMatrixPolynomial, Xout, M1::DenseMatrixPolynomial, X1, M2::DensePolyManifold, X2)
    tab = mulTable(Mout.mexp, Mout.mexp, M2.mexp)
    DensePolySubstitute!(Mout, Xout, M1, X1, M2, X2, tab)
    return nothing
end

function DensePolyMultiply!(Mout::DensePolyManifold, Xout, M1::DenseMatrixPolynomial, X1, M2::DensePolyManifold, X2, tab)
    Xout .= zero(eltype(Xout))
    for d1 = 1:size(X1,1), d2 = 1:size(X1,2) # all dimensions
        @views PolyMul!(Xout[d1,:], X1[d1,d2,:], X2[d2,:], tab)
    end
    return nothing
end

function DensePolyMultiply!(Mout::DensePolyManifold, Xout, M1::DenseMatrixPolynomial, X1, M2::DensePolyManifold, X2)
    tab = mulTable(Mout.mexp, M1.mexp, M2.mexp)
    DensePolyMultiply!(Mout, Xout, M1, X1, M2, X2, tab)
    return nothing
end

# Eval

function DenseMonomials(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}, dataIN) where {ndim, n, m, order}
    return DenseMonomials(ndim, M.mexpALL, dataIN)
end

function DenseDMonomials(M::DensePolyManifold{ndim, n, order, identity, field}, dataIN) where {ndim, n, order, identity, field}
    MON = DenseMonomials(ndim, M.mexpALL, dataIN)
    DMON = zeros(size(MON,1), ndim, size(MON,2))
    for k=1:length(M.DM)
        DMON[:,k,:] .= M.DM[k] * MON
    end
    return MON, DMON
end

function DenseDMonomials(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}, dataIN) where {ndim, n, m, order}
    MON = DenseMonomials(ndim, M.mexpALL, dataIN)
    DMON = zeros(size(MON,1), ndim, size(MON,2))
    for k=1:length(M.DM)
        DMON[:,k,:] .= M.DM[k] * MON
    end
    return MON, DMON
end

function Eval_MON(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}, X, MON, dataMUL) where {ndim, n, m, order}
    @tullio res[i,k] := X[i,j,p] * MON[p,k] * dataMUL[j,k]
    return res
end

function Eval(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}, X, dataIN, dataMUL) where {ndim, n, m, order}
    MON = DenseMonomials(ndim, M.mexp, dataIN)
    return Eval_MON(M, X, MON, dataMUL)
end

# L0_DF

function L0_DF_MON(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}, X, MON, dataMUL; L0) where {ndim, n, m, order}
    @tullio res[i,j,p] := L0[i,k] * MON[p,k] * dataMUL[j,k]
    return res
end

function L0_DF(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}, X, dataIN, dataMUL; L0) where {ndim, n, m, order}
    MON = DenseMonomials(ndim, M.mexp, dataIN)
    return L0_DF_MON(M, X, MON, dataMUL; L0 = L0)
end

function Jacobian_DMON(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}, X, DMON, dataMUL) where {ndim, n, m, order}
    @tullio res[i,k,j] := X[i,q,p] * dataMUL[q,j] * DMON[p,k,j]
    return res
end

function Jacobian(M::DenseMatrixPolynomial{ndim, n, m, order, ℝ}, X, dataIN, dataMUL) where {ndim, n, m, order}
    MON = DenseMonomials(ndim, M.mexpALL, dataIN)
    DMON = zeros(length(M.DM), length(M.DM), size(MON,2))
    for k=1:length(DM)
        DMON[:,k,:] .= M.DM[k] * MON
    end
    return Jacobian_DMON(M, X, DMON[M.admissible,:], dataMUL)
end

function testMatrixJacobian()
    M = DenseMatrixPolynomial(2, 2, 10, 3, min_order = 1)
    X = randn(M)
    dataZ = randn(2,10)
    dataX = randn(10,10)
    J = Jacobian(M, X, dataZ, dataX)
    dataZp = copy(dataZ)
    Jp = copy(J)
    epsilon = 2^(-20)
    for q=1:size(J,2)
        dataZp[q,:] .+= epsilon
        Jp[:,q,:] .= (Eval(M, X, dataZp, dataX) .- Eval(M, X, dataZ, dataX)) ./ epsilon
        dataZp[q,:] .= dataZ[q,:]
    end
    display( J )
    display( J - Jp )
end
