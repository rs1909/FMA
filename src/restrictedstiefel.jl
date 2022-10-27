# n : rows    of X
# k : columns of X
# m : columns of B
# the definition is that 
# X^T . X = I
# B^T . X = 0
struct RestrictedStiefel{n,k,m,𝔽} <: AbstractManifold{𝔽} #AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType} 
    B :: Array{Float64,2}
end

function manifold_dimension(M::RestrictedStiefel{n,k,m,field}) where {n, k, m, field}
    return Integer(n*k - k*(k+1)/2)
end

function RestrictedStiefel(n, k, B, field::AbstractNumbers=ℝ)
    return RestrictedStiefel{n, k, size(B,2), field}(B)
end

function vector_transport_to!(M::RestrictedStiefel{n,k,m,field}, Y::Matrix{Float64}, p::Matrix{Float64}, X::Matrix{Float64}, q::Matrix{Float64}, method::DifferentiatedRetractionVectorTransport{PolarRetraction}) where {n, k, m, field}
    vector_transport_to!(Stiefel(n,k), Y, p, X, q, method)
end

function vector_transport_to(M::RestrictedStiefel{n,k,m,field}, p, X, q, method::AbstractVectorTransportMethod) where {n, k, m, field}
#     println("ISF VECTOR TRANSPORT 1")
    return vector_transport_to(Stiefel(n,k), p, X, q, method)
end


function inner(M::RestrictedStiefel{n,k,m,field}, p, X, Y) where {n, k, m, field}
    return tr(transpose(X)*Y)
end

function retract!(M::RestrictedStiefel, q, p, X, ::PolarRetraction)
#     print("R.")
    s = svd(p + X)
#     mul!(q, (I - M.B * M.B')*s.U, s.Vt)
    # same as the Stiefel retraction if we assume that X is in the tangent space
    mul!(q, s.U, s.Vt)
    return q
end

function retract(M::RestrictedStiefel, p, X, m::AbstractRetractionMethod = default_retraction_method(M))
    q = allocate_result(M, retract, p, X)
    return retract!(M, q, p, X, m)
end

function project!(M::RestrictedStiefel, Y, p, X)
#     print("T.")
# mul!(C, A, B, α, β) -> C
# A B α + C β

    A = p' * X
    T = eltype(Y)
    mul!(Y, (I - M.B * M.B'), X, T(1.0), T(0.0))
    mul!(Y, p, A + A', T(-0.5), T(1.0))
    return Y
end

function project!(M::RestrictedStiefel{n,k,m,field}, q, p) where {n, k, m, field}
#     print("P.")
    r = qr(hcat(M.B, p))
    q .= r.Q[:,m+1:m+k]
    return q
end

function zero(M::RestrictedStiefel{n,k,m,field}) where {n, k, m, field}
    r = qr(M.B)
    return r.Q[:,m+1:m+k]
end

function randn(M::RestrictedStiefel{n,k,m,field}) where {n, k, m, field}
    r = qr(hcat(M.B, randn(n,k)))
    return r.Q[:,m+1:m+k]
end

function zero_vector!(M::RestrictedStiefel, X, p)
    X .= zero(p)
    return X
end

function HessProjection(M::RestrictedStiefel, X, grad, HessV, V)
    HessR = zero(HessV)
    project!(M, HessR, X, HessV - V*SymPart(transpose(X)*grad))
    return HessR
end

function HessFullProjection(M::RestrictedStiefel{n,k,m,field}, X, grad, hess) where {n, k, m, field}
    # H (DR_X, DR_X) + G D^2R_X 
#     print("RStiefel -> H.")
    GTX = grad' * X
    XGT = X * grad'
    H = deepcopy(hess)
    # RestrictedStiefel calculated
    for p1=1:n, q1=1:k, p2=1:n, q2=1:k
        H[p1,q1,p2,q2] -= (GTX[q1,q2]*I[p1,p2] + GTX[q2,q1]*I[p1,p2])/2
    end
    # In Stiefel
#     for p1=1:n, q1=1:k, p2=1:n, q2=1:k
#         H[p1,q1,p2,q2] -= (grad[p2,q1]*X[p1,q2] + XGT[p1,p2]*I[q1,q2] + GTX[q1,q2]*I[p1,p2] + GTX[q2,q1]*I[p1,p2])/2
#     end

    XXT = X*transpose(X)
    # ---------------------------------------------------
    # ORIGINAL RESTRICTED
    #             H2[p1,q1,p2,q2] += (H[i1,j1,i2,j2]*( beta[i1,p1]*I[j1,q1] - (XXT[i1,p1]*I[j1,q1] + X[i1,q1]*X[p1,j1])/2 ) * ( beta[i2,p2]*I[j2,q2] - (XXT[i2,p2]*I[j2,q2] + X[i2,q2]*X[p2,j2])/2 ) )
    #             H2[p1,q1,p2,q2] += ( H[i1,q1,i2,j2]*beta[i1,p1] - H[i1,j1,i2,j2]*(XXT[i1,p1]*I[j1,q1] + X[i1,q1]*X[p1,j1])/2 ) * ( beta[i2,p2]*I[j2,q2] - (XXT[i2,p2]*I[j2,q2] + X[i2,q2]*X[p2,j2])/2 )
    #             H2[p1,q1,p2,q2] += ( H[i1,q1,i2,j2]*beta[i1,p1] - (H[i1,q1,i2,j2]*XXT[i1,p1] + H[i1,j1,i2,j2]*X[i1,q1]*X[p1,j1])/2 ) * ( beta[i2,p2]*I[j2,q2] - (XXT[i2,p2]*I[j2,q2] + X[i2,q2]*X[p2,j2])/2 )
    #             H2[p1,q1,p2,q2] += ( H[i1,q1,i2,q2]*beta[i1,p1] - (H[i1,q1,i2,q2]*XXT[i1,p1] + H[i1,j1,i2,q2]*X[i1,q1]*X[p1,j1])/2 ) * beta[i2,p2]
    #                              - ( H[i1,q1,i2,j2]*beta[i1,p1] - (H[i1,q1,i2,j2]*XXT[i1,p1] + H[i1,j1,i2,j2]*X[i1,q1]*X[p1,j1])/2 ) * (XXT[i2,p2]*I[j2,q2] + X[i2,q2]*X[p2,j2])/2
    #             H2[p1,q1,p2,q2] += ( H[i1,q1,i2,q2]*beta[i1,p1] - (H[i1,q1,i2,q2]*XXT[i1,p1] + H[i1,j1,i2,q2]*X[i1,q1]*X[p1,j1])/2 ) * beta[i2,p2]
    #                              - ( H[i1,q1,i2,q2]*beta[i1,p1]/2 - (H[i1,q1,i2,q2]*XXT[i1,p1] + H[i1,j1,i2,q2]*X[i1,q1]*X[p1,j1])/4 ) * XXT[i2,p2]
    #                              - ( H[i1,q1,i2,j2]*beta[i1,p1]/2 - (H[i1,q1,i2,j2]*XXT[i1,p1] + H[i1,j1,i2,j2]*X[i1,q1]*X[p1,j1])/4 ) * X[i2,q2]*X[p2,j2]
    beta = I - M.B*transpose(M.B)
    @tensoropt H3[p1,q1,p2,q2] := (( H[i1,q1,i2,q2]*beta[i1,p1] - (H[i1,q1,i2,q2]*XXT[i1,p1] + H[i1,j1,i2,q2]*X[i1,q1]*X[p1,j1])/2 ) * beta[i2,p2]
                                  - ( H[i1,q1,i2,q2]*beta[i1,p1]/2 - (H[i1,q1,i2,q2]*XXT[i1,p1] + H[i1,j1,i2,q2]*X[i1,q1]*X[p1,j1])/4 ) * XXT[i2,p2]
                                  - ( H[i1,q1,i2,j2]*beta[i1,p1]/2 - (H[i1,q1,i2,j2]*XXT[i1,p1] + H[i1,j1,i2,j2]*X[i1,q1]*X[p1,j1])/4 ) * X[i2,q2]*X[p2,j2] )
#     H2 = zero(hess)
#     for p1=1:n, q1=1:k, p2=1:n, q2=1:k
#         for i1=1:n, j1=1:k, i2=1:n, j2=1:k
#             H2[p1,q1,p2,q2] += (H[i1,j1,i2,j2]*( beta[i1,p1]*I[j1,q1] - (XXT[i1,p1]*I[j1,q1] + X[i1,q1]*X[p1,j1])/2 ) * 
#                                                ( beta[i2,p2]*I[j2,q2] - (XXT[i2,p2]*I[j2,q2] + X[i2,q2]*X[p2,j2])/2 ) )
#         end
#     end
#     println("HessProjTime-R ERROR = ", maximum(abs.(H2 - H3)))
    return H3
end

function RSloss(M, X)
    return sum(X' * sin.(X) * X')
end

function RSgrad(M, X)
    G = sum(sin.(X) * X',dims=2) .* ones(1,size(X,2)) 
    G .+= sum(X' * sin.(X), dims=1) .* ones(size(X,1),1) 
    G .+= sum(X,dims=2) .* cos.(X) .* sum(X,dims=1)
    return G
end

function RSHessian(M, X)
    H = zeros(size(X,1), size(X,2), size(X,1), size(X,2))
    for p1=1:size(X,1), q1=1:size(X,2), p2=1:size(X,1), q2=1:size(X,2)
        for j=1:size(X,2), l=1:size(X,1)
            H[p1,q1,p2,q2] += ( 
                  I[j,q1]*cos(X[p1,q2])*I[p1,p2]*X[l,q2]
                + I[j,q1]*sin(X[p1,q2])*I[l,p2]
                + I[p1,p2]*I[j,q2]*cos(X[p1,q1])*X[l,q1]
                - X[p1,j]*sin(X[p1,q1])*I[p1,p2]*I[q1,q2]*X[l,q1]
                + X[p1,j]*cos(X[p1,q1])*I[l,p2]*I[q1,q2]
                + I[j,q2]*sin(X[p2,q1])*I[l,p1]
                + X[p2,j]*cos(X[p2,q1])*I[q1,q2]*I[l,p1] )
        end
    end
    return H
end

function testRStiefel()
    s = svd(randn(10, 2))
    B = s.U * s.Vt
#     M = RestrictedStiefel(10, 4, B)
    M = Stiefel(10, 4)
    X = randn(M)
    G = RSgrad(M, X)
    H = RSHessian(M, X)
    GR = project(M, X, G)
    HR = HessFullProjection(M, X, G, H)
    
    # plain derivative
    eps =  1e-6
    Xp = deepcopy(X)
    Gp = deepcopy(G)
    for k=1:length(X)
        Xp[k] += eps
        Gp[k] = (RSloss(M, Xp) - RSloss(M, X))/eps
        Xp[k] = X[k]
        @show k, Gp[k] - G[k], G[k]
    end
    # Riemannian derivative
    eps =  1e-6
    dX = zero(X)
    Gp = deepcopy(G)
    for k=1:length(X)
        dX[k] = eps
        Xpp = project(M, X, dX)
        Xp = retract(M, X, Xpp, PolarRetraction())
        Gp[k] = (RSloss(M, Xp) - RSloss(M, X))/eps
        dX[k] = 0
        @show k, Gp[k] - GR[k], GR[k]
    end
    # plain Hessian
    eps =  1e-6
    Xp = deepcopy(X)
    Hp = deepcopy(H)
    for p1=1:size(X,1), q1=1:size(X,2)
        Xp[p1,q1] += eps
        Hp[p1,q1,:,:] = (RSgrad(M, Xp) - RSgrad(M, X))/eps
        Xp[p1,q1] = X[p1,q1]
        @show maximum(abs.(Hp[p1,q1,:,:] - H[p1,q1,:,:])), maximum(abs.(H[p1,q1,:,:]))
#         display(Hp[p1,q1,:,:] - H[p1,q1,:,:])
    end
    # Riemannian Hessian
    # just to make it accurate use the finite difference version of Hessian
    HR = HessFullProjection(M, X, G, H)
    
    HR2 = reshape(HR, size(HR,1)*size(HR,2), :)
    
    # assymetric part
#     @show maximum(abs.(HR2 .- transpose(HR2)))
    
    eps =  1e-9
    dX = zero(X)
    HRp = zero(HR)
#     HRp2 = zero(HR)
    for p1=1:size(X,1), q1=1:size(X,2)
        dX[p1,q1] = eps
        Xpp = project(M, X, dX)
        Xp = retract(M, X, Xpp, PolarRetraction())
        HRp[p1,q1,:,:] .= project(M, X, (project(M, Xp, RSgrad(M, Xp)) - project(M, X, RSgrad(M, X)))/eps)
#         dX[p1,q1] = 1.0
#         HRp2[p1,q1,:,:] .= HessProjection(M, X, G, H[:,:,p1,q1], dX)
        dX[p1,q1] = 0
        @show maximum(abs.(HRp[p1,q1,:,:] .- HR[p1,q1,:,:])), maximum(abs.(HR[p1,q1,:,:]))
#         display(abs.(HRp[p1,q1,:,:] .- HR[p1,q1,:,:]))
#         display(HRp[p1,q1,:,:] - 0*HR[p1,q1,:,:])
#         display(HR[p1,q1,:,:])
    end
    HRp2 = reshape(HRp, size(HRp,1)*size(HRp,2), :)
    
    # assymetric part
#     @show display(abs.(HRp2 .- transpose(HRp2)) )
    
    @show maximum(abs.(HRp .- HR))
#     @show maximum(abs.(HRp2 .- HR))
    println("H - analytic")
#     display(reshape(H, size(X,1) * size(X,2), size(X,1) * size(X,2)))
    FH = eigen(reshape(H, size(X,1) * size(X,2), size(X,1) * size(X,2)))
    @show real.(FH.values)
    display(FH.vectors)
    println("HR - analytic")
#     display(reshape(HR, size(X,1) * size(X,2), size(X,1) * size(X,2)))
    FHR = eigen(reshape(HR, size(X,1) * size(X,2), size(X,1) * size(X,2)))
    @show real.(FHR.values)
    display(FHR.vectors)
    println("HRp - proj fin diff")
#     display(reshape(HRp, size(X,1) * size(X,2), size(X,1) * size(X,2)))
    FHRp = eigen(reshape(HRp, size(X,1) * size(X,2), size(X,1) * size(X,2)))
    @show real.(FHRp.values)
    display(FHRp.vectors)
#     println("HRp2 - non-proj fin diff")
#     display(reshape(HRp2, size(X,1) * size(X,2), size(X,1) * size(X,2)))
#     @show real.(eigvals(reshape(HRp2, size(X,1) * size(X,2), size(X,1) * size(X,2))))
#     display(abs.(HRp2 .- HR2) .> 1e-4)
end
