function zero(M::Euclidean{Tuple{m, n}, field}) where {m, n, field}
    return zeros(m, n)
end

function randn(M::Euclidean{Tuple{m, n}, field}) where {m, n, field}
    return randn(m, n)
end

function zero(M::Stiefel{m, n, field}) where {m, n, field}
    return project(M, randn(m, n))
end

function randn(M::Stiefel{m, n, field}) where {m, n, field}
    return project(M, randn(m, n))
end

function *(x::AbstractArray{T,3}, y::AbstractArray{T,1}) where T
    return dropdims(sum(x .* reshape(y, 1, 1, size(y,1)), dims=3), dims=3)
end

function *(x::AbstractArray{T,3}, y::AbstractArray{T,2}) where T
    return dropdims(sum(reshape(x, size(x,1), size(x,2), size(x,3), 1) .* reshape(y, 1, 1, size(y,1), size(y,2)), dims=3), dims=3)
end

# extending the Stiefel manifold with projecting the Hessian
function SymPart(V)
    return (transpose(V) + V)/2.0
end

function HessProjection(M::Stiefel{n,k,field}, X, grad, HessV, V) where {n, k, field}
    return project(M, X, HessV - V*SymPart(transpose(X)*grad))
end

function HessProjection(M::Euclidean{Tuple{n,k},field}, X, grad, HessV, V) where {n, k, field}
    return HessV
end

function HessProjection(M::ProductManifold, X, grad, HessV, V)
    return ProductRepr(map(HessProjection, M.manifolds, X.parts, grad.parts, HessV.parts, V.parts))
end

function HessFullProjection(M::Stiefel{n,k,field}, X, grad, hess) where {n, k, field}
    H = deepcopy(hess)
    GTX = grad' * X
    XGT = X * grad'
    XXT = X*transpose(X)

    for p1=1:n, q1=1:k, p2=1:n, q2=1:k
        @inbounds H[p1,q1,p2,q2] -= (GTX[q1,q2]*I[p1,p2] + GTX[q2,q1]*I[p1,p2])/2
    end

    # H2[p1,q1,p2,q2] += H[i1,j1,i2,j2]*( I[i1,p1]*I[j1,q1] - (XXT[i1,p1]*I[j1,q1] + X[i1,q1]*X[p1,j1])/2 )*(I[i2,p2]*I[j2,q2] - (XXT[i2,p2]*I[j2,q2] + X[i2,q2]*X[p2,j2])/2)
    # H2[p1,q1,p2,q2] += H[i1,j1,i2,j2]*I[i1,p1]*I[j1,q1]*I[i2,p2]*I[j2,q2] - H[i1,j1,i2,j2]*I[i1,p1]*I[j1,q1]*(XXT[i2,p2]*I[j2,q2] + X[i2,q2]*X[p2,j2])/2 
    #                  - H[i1,j1,i2,j2]*(XXT[i1,p1]*I[j1,q1] + X[i1,q1]*X[p1,j1]) *(I[i2,p2]*I[j2,q2] - (XXT[i2,p2]*I[j2,q2] + X[i2,q2]*X[p2,j2])/2)/2
    # H2[p1,q1,p2,q2] += H[p1,q1,p2,q2] - H[p1,q1,i2,j2]*(XXT[i2,p2]*I[j2,q2] + X[i2,q2]*X[p2,j2])/2 
    #                  - (H[i1,j1,i2,j2]*XXT[i1,p1]*I[j1,q1] + H[i1,j1,i2,j2]*X[i1,q1]*X[p1,j1])*(I[i2,p2]*I[j2,q2] - (XXT[i2,p2]*I[j2,q2] + X[i2,q2]*X[p2,j2])/2)/2
    # H2[p1,q1,p2,q2] += H[p1,q1,p2,q2] 
    #                  - (H[p1,q1,i2,q2]*XXT[i2,p2] + H[p1,q1,i2,j2]*X[p2,j2]*X[i2,q2])/2  # P 
    #                  - (H[i1,q1,p2,q2]*XXT[i1,p1] + H[i1,j1,p2,q2]*X[i1,q1]*X[p1,j1])/2  # P^T
    #                  + H[i1,q1,i2,q2]*XXT[i1,p1]*XXT[i2,p2]/4                            # Q
    #                  + H[i1,j1,i2,q2]*X[i1,q1]*X[p1,j1]*XXT[i2,p2]/4                     # R
    #                  + H[i1,q1,i2,j2]*XXT[i1,p1]*X[i2,q2]*X[p2,j2]/4                     # R^T
    #                  + H[i1,j1,i2,j2]*X[i1,q1]*X[p1,j1]*X[i2,q2]*X[p2,j2]/4              # S
    @tensoropt H3[p1,q1,p2,q2] := (H[p1,q1,p2,q2] 
                     - (H[p1,q1,i2,q2]*XXT[i2,p2] + H[p1,q1,i2,j2]*X[p2,j2]*X[i2,q2])/2  # P 
                     - (H[i1,q1,p2,q2]*XXT[i1,p1] + H[i1,j1,p2,q2]*X[i1,q1]*X[p1,j1])/2  # P^T
                     + H[i1,q1,i2,q2]*XXT[i1,p1]*XXT[i2,p2]/4                            # Q
                     + H[i1,j1,i2,q2]*X[i1,q1]*X[p1,j1]*XXT[i2,p2]/4                     # R
                     + H[i1,q1,i2,j2]*XXT[i1,p1]*X[i2,q2]*X[p2,j2]/4                     # R^T
                     + H[i1,j1,i2,j2]*X[i1,q1]*X[p1,j1]*X[i2,q2]*X[p2,j2]/4)             # S
    # Check if the result is correct in another way...
#     H2 = zero(hess)
#     @time for p1=1:n, q1=1:k, p2=1:n, q2=1:k
#         for i1=1:n, j1=1:k, i2=1:n, j2=1:k
#             @inbounds H2[p1,q1,p2,q2] += (H[i1,j1,i2,j2]*( I[i1,p1]*I[j1,q1] - (XXT[i1,p1]*I[j1,q1] + X[i1,q1]*X[p1,j1])/2 ) * 
#                                                ( I[i2,p2]*I[j2,q2] - (XXT[i2,p2]*I[j2,q2] + X[i2,q2]*X[p2,j2])/2 ) )
#         end
#     end
#     println("HessProjTime ERROR = ", maximum(abs.(H2 - H3)))
    return H3
end

function HessFullProjection(M::Euclidean{Tuple{n,k},field}, X, grad, hess) where {n, k, field}
    return hess
end
