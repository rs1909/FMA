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
    return project(M, X, HessV) - V*SymPart(transpose(X)*grad)
end

function HessProjection(M::Euclidean{Tuple{n,k},field}, X, grad, HessV, V) where {n, k, field}
    return HessV
end

function HessProjection(M::ProductManifold, X, grad, HessV, V)
    return ProductRepr(map(HessProjection, M.manifolds, X.parts, grad.parts, HessV.parts, V.parts))
end

# function HessFullProjection(M::Stiefel{n,k,field}, X, grad, hess) where {n, k, field}
#     B = SymPart(transpose(X)*grad)
#     H = zero(hess)
#     for k1=1:size(H,3), k2=1:size(H,4)
#         @views project!(M, H[:,:,k1,k2], X, hess[:,:,k1,k2])
#         for j1=1:size(H,1)
#             H[j1,:,k1,k2] .-= I[j1,k1]*B[k2,:]
#         end
#     end
#     return H
# end

function HessFullProjection(M::Stiefel{n,k,field}, X, grad, hess) where {n, k, field}
#     H = zero(hess)
#     XXT = X*transpose(X)
#     for r1=1:n, s1=1:k, r2=1:n, s2=1:k
#         for p1=1:n, q1=1:k, p2=1:n, q2=1:k
#             H[r1,s1,r2,s2] += (hess[p1,q1,p2,q2]*(I[p1,r1]*I[q1,s1] - (XXT[p1,r1] + X[p1,s1]*X[r1,q1])/2)
#                 *(I[p2,r2]*I[q2,s2] - (XXT[p2,r2] + X[p2,s2]*X[r2,q2])/2))
#         end
#     end
    H = deepcopy(hess)
#     for i=1:n, p=1:n, q1=1:k, q2=1:k
#         H[p,q1,p,q2] -= (1/2)*(grad[i,q2]*X[i,q1] + grad[i,q1]*X[i,q2])
#     end
    GTX = grad' * X
    XGT = X * grad'
#     for p1=1:n, q1=1:k, p2=1:n, q2=1:k
#         H[p1,q1,p2,q2] -= (grad[p2,q1]*X[p1,q2] + XGT[p1,p2]*I[q1,q2] + GTX[q1,q2]*I[p1,p2] + GTX[q2,q1]*I[p1,p2])/2
#     end
    for p1=1:n, q1=1:k, p2=1:n, q2=1:k
        H[p1,q1,p2,q2] -= (GTX[q1,q2]*I[p1,p2] + GTX[q2,q1]*I[p1,p2])/2
    end
    H2 = zero(hess)
    XXT = X*transpose(X)
#     for r1=1:n, s1=1:k, r2=1:n, s2=1:k
#         for p1=1:n, q1=1:k, p2=1:n, q2=1:k
#             H2[r1,s1,r2,s2] += (H[p1,q1,p2,q2]*(I[p1,r1]*I[q1,s1] - (XXT[p1,r1]*I[q1,s1] + X[p1,s1]*X[r1,q1])/2)
#                 *(I[p2,r2]*I[q2,s2] - (XXT[p2,r2]*I[q2,s2] + X[p2,s2]*X[r2,q2])/2))
#         end
#     end
    for p1=1:n, q1=1:k, p2=1:n, q2=1:k
        for i1=1:n, j1=1:k, i2=1:n, j2=1:k
            H2[p1,q1,p2,q2] += (H[i1,j1,i2,j2]*( I[i1,p1]*I[j1,q1] - (XXT[i1,p1]*I[j1,q1] + X[i1,q1]*X[p1,j1])/2 ) * 
                                               ( I[i2,p2]*I[j2,q2] - (XXT[i2,p2]*I[j2,q2] + X[i2,q2]*X[p2,j2])/2 ) )
        end
    end
    return H2
end

function HessFullProjection(M::Euclidean{Tuple{n,k},field}, X, grad, hess) where {n, k, field}
    return hess
end
