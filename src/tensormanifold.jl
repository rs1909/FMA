
## ---------------------------------------------------------------------------------------
## TensorManifold
## 
## ---------------------------------------------------------------------------------------

struct TensorManifold{𝔽} <: AbstractManifold{𝔽}
    ranks    :: Array{T,1} where T <: Integer
    children :: Array{T,2} where T <: Integer
    dim2ind  :: Array{T,1} where T <: Integer
    M        :: ProductManifold 
    R        :: ProductRetraction
    VT       :: ProductVectorTransport
end

function getRetraction(M::TensorManifold{𝔽}) where 𝔽
    return M.R
end

# internal
"""
    number of nodes of a HT tensor
"""
function nr_nodes(children::Array{T,2}) where T <: Integer
    return size(children, 1)
end

function nr_nodes(hten::TensorManifold{𝔽}) where 𝔽
    return nr_nodes(hten.children)
end

"""
    check if a node is a leaf
"""
function is_leaf(children::Array{T,2}, ii) where T <: Integer
    return prod(children[ii,:] .== 0)
end

function is_leaf(hten::TensorManifold{𝔽}, ii) where 𝔽
    return is_leaf(hten.children, ii)
end

# import Base.size
# 
# function size(M::TensorManifold{𝔽}) where T
#     return Tuple(size(X.parts[k],1) for k in M.dim2ind)
# end

"""
    define_tree(d, tree_type = :balanced)
    creates a tree structure from dimensions 'd'
"""
function define_tree(d, tree_type = :balanced)
    children = zeros(typeof(d), 2*d-1, 2)
    dims = [collect(1:d)]
    
    nr_nodes = 1
    ii = 1
    while ii <= nr_nodes
        if length(dims[ii]) == 1
            children[ii,:] = [0, 0]
        else
            ii_left = nr_nodes + 1
            ii_right = nr_nodes + 2
            nr_nodes = nr_nodes + 2
            push!(dims, [])
            push!(dims, [])
            
            children[ii,:] = [ii_left, ii_right]
            if tree_type == :first_separate && ii == 1
                dims[ii_left]  = [dims[ii][1]]
                dims[ii_right] = dims[ii][2:end]
            elseif tree_type == :first_pair_separate && ii == 1 && d > 2
                dims[ii_left]  = dims[ii][1:2]
                dims[ii_right] = dims[ii][3:end]
            elseif tree_type == :TT
                dims[ii_left]  = [dims[ii][1]]
                dims[ii_right] = dims[ii][2:end]
            else
                dims[ii_left]  = dims[ii][1:div(end,2)]
                dims[ii_right] = dims[ii][div(end,2)+1:end]
            end
        end
        ii += 1
    end
    ind_leaves = findall(children[:,1] .== 0)
    pivot = [dims[k][1] for k in ind_leaves]
    dim2ind = zero(pivot)
    dim2ind[pivot] = ind_leaves
    return children, dim2ind
end

# a complicated constructor
function TensorManifold(dims::Array{T,1}, topdim::T, ranks::Array{T,1}, children, dim2ind, B=nothing, tree_type = :balanced; field::AbstractNumbers=ℝ) where T <: Integer
    M = []
    R = []
    VT = []
    for ii = 1:nr_nodes(children)
        if is_leaf(children, ii)
            dim_id = findfirst(isequal(ii), dim2ind)
            n_ii = dims[dim_id]
            if B == nothing
                push!(M, Stiefel(n_ii, ranks[ii]))
            else
#                 if dim_id == 1
                    push!(M, RestrictedStiefel(n_ii, ranks[ii], B))
#                 else
#                     push!(M, Stiefel(n_ii, ranks[ii]))
#                 end
            end
            push!(R, PolarRetraction())
            push!(VT, DifferentiatedRetractionVectorTransport(PolarRetraction()))
        else
            ii_left = children[ii,1]
            ii_right = children[ii,2]
            if ii == 1
                push!(M, Euclidean(ranks[ii_left] * ranks[ii_right], ranks[ii]))
                push!(R, ExponentialRetraction())
                push!(VT, ParallelTransport())
            else
                push!(M, Stiefel(ranks[ii_left] * ranks[ii_right], ranks[ii]))
                push!(R, PolarRetraction())
                push!(VT, DifferentiatedRetractionVectorTransport(PolarRetraction()))
            end
        end
    end
    return TensorManifold{field}(ranks, children, dim2ind, ProductManifold(M...), ProductRetraction(R...), ProductVectorTransport(VT...))
end

function prune_ranks!(dims::Array{T,1}, topdim::T, ranks::Array{T,1}, children, dim2ind, B=nothing) where T <: Integer
    if B == nothing
        ranksub = 0
    else
        ranksub = size(B,2)
    end
    ranks[1] = topdim
    for ii = nr_nodes(children):-1:2
        if is_leaf(children, ii)
            n_ii = dims[findfirst(isequal(ii), dim2ind)] - ranksub
            if ranks[ii] > n_ii
                ranks[ii] = n_ii
            end
        else
            ii_left = children[ii,1]
            ii_right = children[ii,2]
            if ranks[ii] > ranks[ii_left] * ranks[ii_right]
                ranks[ii] = ranks[ii_left] * ranks[ii_right]
            end
        end
    end
    return nothing
end

function MinimalTensorManifold(dims::Array{T,1}, topdim::T = 1, B=nothing, tree_type = :balanced) where T <: Integer
    children, dim2ind = define_tree(length(dims), tree_type)
    nodes = nr_nodes(children)
    # create ranks at each node
    ranks = ones(Int, nodes)
    # the root node is singular
    prune_ranks!(dims, topdim, ranks, children, dim2ind, B)
    return TensorManifold(dims, topdim, ranks, children, dim2ind, B, tree_type)
end

"""
    Create a tensor manifold with rank 6 at each node. It is not random.
    This was supposed to be a temporary measure, but became permanent due to having worked well
    A better analysis on how to select ranks is needed
"""
function RandomTensorManifold(dims::Array{T,1}, topdim::T = 1, B=nothing, tree_type = :balanced) where T <: Integer
    children, dim2ind = define_tree(length(dims), tree_type)
    nodes = nr_nodes(children)
    # create ranks at each node
    ranks = [rand(6:6) for k=1:nodes]
    # the root node is singular
    prune_ranks!(dims, topdim, ranks, children, dim2ind, B)
    return TensorManifold(dims, topdim, ranks, children, dim2ind, B, tree_type)
end

function project!(M::TensorManifold{field}, Y, p, X) where field
    return project!(M.M, Y, p, X)
end

function randn(M::TensorManifold{field}) where field
    return ProductRepr(map(randn, M.M.manifolds))
end

function zero(M::TensorManifold{field}) where field
    return ProductRepr(map(zero, M.M.manifolds))
end

function zero_vector!(M::TensorManifold{field}, X, p) where field
    return zero_vector!(M.M, X, p)
end

function getel(hten::TensorManifold{field}, X, idx) where field
    # this goes from the top level to the lowest level
    vecs = [zeros(Float64, 0) for k = 1:size(hten.children, 1)]
    for ii = size(hten.children,1):-1:1
        if is_leaf(hten, ii)
            sel = idx[findfirst(isequal(ii), hten.dim2ind)]
            vecs[ii] = X.parts[ii][sel,:]
        else
            ii_left = hten.children[ii,1]
            ii_right = hten.children[ii,2]
            s_l = size(X.parts[ii_left],2)
            s_r = size(X.parts[ii_right],2)
            vecs[ii] = [transpose(vecs[ii_left])*reshape(X.parts[ii][:,k], s_l, s_r)*vecs[ii_right] for k=1:size(X.parts[ii],2)]
        end
    end
    return vecs[1][idx[end]]
end

"""
    Calculates the Gramian matrices of the HT tensor
"""
function gramians(hten::TensorManifold{field}, X) where field
    gr = Array{Array{Float64,2},1}(undef, size(hten.children,1))
    gr[1] = transpose(X.parts[1]) * X.parts[1]
    for ii = 1:size(hten.children,1)
        if !is_leaf(hten, ii)
            # Child nodes
            ii_left  = hten.children[ii, 1]
            ii_right = hten.children[ii, 2]
            s_l = size(X.parts[ii_left],2)
            s_r = size(X.parts[ii_right],2)
  
#             % Calculate contractions < B{ii}, G{ii} o_1 B{ii} >_(1, 2) and _(1, 3)
#             B_mod = ttm(conj(x.B{ii}), G{ii}, 3);
#             @show size(X.parts[ii]), size(gr[ii])
#             @show size(reshape(X.parts[ii], s_l, s_r,:)), size(reshape(gr[ii], size(gr[ii],1), size(gr[ii],2), 1))
            B_mod = reshape(X.parts[ii], s_l, s_r,:) * gr[ii]
#   G{ii_left } = ttt(conj(x.B{ii}), B_mod, [2 3], [2 3], 1, 1);
#   G{ii_right} = ttt(conj(x.B{ii}), B_mod, [1 3], [1 3], 2, 2);
            gr[ii_left] = dropdims(sum(reshape(X.parts[ii], s_l, 1, s_r, :) .* reshape(B_mod, 1, s_l, s_r, :), dims=(3,4)), dims=(3,4))
            gr[ii_right] = dropdims(sum(reshape(X.parts[ii], s_l, 1, s_r, :) .* reshape(B_mod, s_l, s_r, 1, :), dims=(1,4)), dims=(1,4))           
        end
    end
    return gr
end

# normally we don't need this
# orthogonalise the non-root nodes
"""
    This orthogonalises the non-root nodes of the HT tensor, by leaving the value unchanged.
    Normally, this method is not needed, because our algorithms keep these matrices orthogonal
"""
function orthog!(M::TensorManifold{field}, X) where field
    for ii = nr_nodes(M):-1:2
        if is_leaf(M, ii)
            F = qr(X.parts[ii])
            X.parts[ii] .= Array(F.Q)
            R = F.R
        else
            F = qr(X.parts[ii])
            X.parts[ii] .= Array(F.Q)
            R = F.R
        end
        left_par = findfirst(isequal(ii), M.children[:,1])
        right_par = findfirst(isequal(ii), M.children[:,2])
        if left_par != nothing
            for k=1:size(X.parts[left_par], 2)
                X.parts[left_par][:,k] .= vec(R*reshape(X.parts[left_par][:,k],size(R,2),:))
            end
        else
            for k=1:size(X.parts[right_par], 2)
                X.parts[right_par][:,k] .= vec(reshape(X.parts[right_par][:,k],:,size(R,2))*transpose(R))
            end
        end
    end
    return nothing
end

# -------------------------------------------------------------------------------------
#
# In this section we calculate the hmtensor derivative of X with respect to X.U, X.B
# The data is an array for multiple evaluations at the same time
# we need to allow for a missing index, so that a vector is output
# therefore we need to propagate a matrix through
#
# -------------------------------------------------------------------------------------

# this acts as a cache for the tensor evaluations and gradients
struct diadicVectors{T}
    valid_vecs  :: Array{Bool,1}
    valid_bvecs :: Array{Bool,1}
    vecs        :: Array{Array{T,3},1}
    bvecs       :: Array{Array{T,3},1}
end

function invalidateVecs(DV::diadicVectors)
    DV.valid_vecs .= false
    nothing
end

function invalidateBVecs(DV::diadicVectors)
    DV.valid_bvecs .= false
    nothing
end

function invalidateAll(DV::diadicVectors)
    DV.valid_vecs .= false
    DV.valid_bvecs .= false
    nothing
end

function diadicVectors(T, nodes)
    return diadicVectors{T}(zeros(Bool, nodes), zeros(Bool, nodes), Array{Array{T,3},1}(undef, nodes), Array{Array{T,3},1}(undef, nodes))
end

function LmulBmulR!(vecs_ii, vecs_left, B, vecs_right)
    for k=1:size(vecs_right,3)
        for jr = 1:size(vecs_right, 2)
            for jl = 1:size(vecs_left, 2)
                for jb=1:size(B,2)
                    for q=1:size(vecs_right, 1)
                        for p=1:size(vecs_left, 1)
                            vecs_ii[jb, jr*jl, k] += vecs_left[p,jl,k] * B[p + (q-1)*size(vecs_left, 1), jb] .* vecs_right[q,jr,k]
                        end
                    end
                end
            end
        end
    end
    return nothing
end

function BmulData!(vecs_ii, B, data_wdim)
    for k=1:size(data_wdim,2)
        for p=1:size(B,2)
            for q=1:size(B,1)
                vecs_ii[p,1,k] += B[q,p] * data_wdim[q,k]
            end
        end
    end
    return nothing
end

# contracts data[i] with index[i] of the tensor X
# if i > length(data) data[end] is contracted with index[i]
# if third != 0, index[third] is not contracted
# output:
#   vecs is a matrix for each data point
# input:
#   X: is the tensor
#   data: is a vector of two dimensional arrays. 
#       The first dimension is the vector, the second dimension is the data index
#   ii: the node to contract at
#   second: if non-zero data[2] is contracted with this index, data[1] with the rest except 'third'
#   third: if non-zero, the index not to contract
#   dt: to replace X at index rep
#   rep: where to replace X with dt
function tensorVecsRecursive!(DV::diadicVectors{T}, M::TensorManifold{field}, X, data::Vector{Matrix{T}}, ii; second::Integer = 0, third::Integer = 0, dt::ProductRepr = ProductRepr(), rep::Integer = 0) where {field, T}
    if DV.valid_vecs[ii]
        return nothing
    end
    if ii != rep
        B = X.parts[ii]
    else
        B = dt.parts[ii]
    end
#     @show ii, is_leaf(M, ii), size(X.parts[ii]), M.children[ii,:]
    if is_leaf(M, ii)
        # it is a leaf
        dim = findfirst(isequal(ii), M.dim2ind)
        if second == dim
            wdim = 2
        elseif second > 0
            wdim = 1
        elseif dim > length(data)
            wdim = length(data)
        else
            wdim = dim
        end
        if dim == third
            contract = false
        else
            contract = true
        end
#         @show wdim, contract, length(data), third
        if contract
#             tt = zeros(T, size(B,2), 1, size(data[1],2) )
            DV.vecs[ii] = zeros(T, size(B,2), 1, size(data[1],2) ) # Array{T}(undef, (size(B,2), 1, size(data[1],2)))
#             @show size(DV.vecs[ii]), size(B), size(data[wdim])
            BmulData!(DV.vecs[ii], B, data[wdim])
        else
            DV.vecs[ii] = zeros(T, size(B,2), size(B,1), size(data[1],2) )
            for k=1:size(data[1],2)
                @views DV.vecs[ii][:,:,k] .= transpose(B)
            end
        end
    else
        # it is a node
        ii_left = M.children[ii,1]
        ii_right = M.children[ii,2]
        tensorVecsRecursive!(DV, M, X, data, ii_left; second = second, third = third, dt = dt, rep = rep)
        tensorVecsRecursive!(DV, M, X, data, ii_right; second = second, third = third, dt = dt, rep = rep)
        vs_l = size(DV.vecs[ii_left], 2)
        vs_r = size(DV.vecs[ii_right], 2)
        vs = max(vs_l,vs_r)
        DV.vecs[ii] = zeros( size(B,2), vs, size(data[1],2) )
        
        LmulBmulR!(DV.vecs[ii], DV.vecs[ii_left], B, DV.vecs[ii_right])
    end
    # This is an error. It must return something
#     @show ii, vecs[ii]
    DV.valid_vecs[ii] = true
    return nothing
end

# invalidates vecs that are dependent on node "ii"
function tensorVecsInvalidate(DV::diadicVectors{T}, M::TensorManifold{field}, ii) where {field, T}
    DV.valid_vecs[ii] = false
    # not the root node
    if ii != 1
        left = findfirst(isequal(ii), M.children[:,1])
        right = findfirst(isequal(ii), M.children[:,2])
        if left != nothing
            tensorVecsInvalidate(DV, M, left)
        end
        if right != nothing
            tensorVecsInvalidate(DV, M, right)
        end
    end
    return nothing
end

function tensorHessianRecursive(M::TensorManifold{field}, X, data, ii, d1, d2) where field
    B = X.parts[ii]
    if is_leaf(M, ii)
        if ii == d1
            hess = zeros(size(B,2), 1, size(B,1), size(data,2))
            for k=1:size(data,2)
                hess[:,1,:,k] .= transpose(B)
            end
        elseif ii == d2
            hess = zeros(size(B,2), size(B,1), 1, size(data,2))
            for k=1:size(data,2)
                hess[:,:,1,k] .= transpose(B)
            end
        else
            hess = sum(reshape(transpose(B), size(B,2), size(B,1), 1, 1) .* reshape(data, 1, size(data,1), 1, size(data,2)), dims=2)
        end
    else
        ii_left  = M.children[ii,1]
        ii_right = M.children[ii,2]
        v_l = tensorHessianRecursive(M, X, data, ii_left, d1, d2)
        v_r = tensorHessianRecursive(M, X, data, ii_right, d1, d2)
        vs_l = size(v_l, 1)
        vs_r = size(v_r, 1)
        hess = zeros(size(B,2), size(v_l,2)*size(v_r,2), size(v_l,3)*size(v_r,3), size(data,2))
#         @show ii, size(hess), size(v_l), size(v_r)
        for j1 = 1:vs_r, j2 = 1:vs_l
            for l = 1:size(data,2), q_l = 1:size(v_l,3), q_r = 1:size(v_r,3), p_l = 1:size(v_l,2), p_r = 1:size(v_r,2)
                for r=1:size(B,2)
                    hess[r, p_l*p_r, q_l*q_r, l] += B[j2 + (j1-1)*vs_l, r] * v_l[j2, p_l, q_l, l] .* v_r[j1, p_r, q_r, l]
                end
            end
        end
    end
    return hess
end

function Hessian(M::TensorManifold{field}, X, data) where field
    hess = zeros(size(X.parts[1],2), size(data,1), size(data,1), size(data,2))
    for d1 in M.dim2ind
        for d2 in M.dim2ind
            if d1 != d2
#                 println("H ", d1, " ", d2)
                hessDelta = tensorHessianRecursive(M, X, data, 1, d1, d2)
#                 @show size(hess), size(hessDelta)  
                hess .+= hessDelta
            end
        end
    end
    return hess
end

# create partial results at the end of each node or leaf when multiplied with 'data'
function tensorVecs(M::TensorManifold{field}, X, data::Vector{Matrix{T}}; second::Integer = 0, third::Integer = 0, dt::ProductRepr = ProductRepr(), rep::Integer = 0) where {field, T}
    DV = diadicVectors(T, size(M.children, 1))
    tensorVecsRecursive!(DV, M, X, data, 1; second = second, third = third, dt = dt, rep = rep)
    return DV
end

# topdata is a multiplication from the left
function Eval(M::TensorManifold{field}, X, data, topdata = nothing; DV::diadicVectors = tensorVecs(M, X, data)) where field
    if topdata == nothing
#         @show size(vecs[1])
        return dropdims(DV.vecs[1], dims=2)
    else
#         @show size(topdata), size(vecs[1])
        return dropdims(sum(DV.vecs[1] .* reshape(topdata, size(topdata,1), 1, size(topdata,2)), dims=1), dims=(1,2))
    end
end


function BVpmulBmulV!(bvecs_ii::AbstractArray{T,3}, B::AbstractArray{T,2}, vecs_sibling::AbstractArray{T,3}, bvecs_parent::AbstractArray{T,3}) where T
    for k=1:size(bvecs_ii,3)
        for r=1:size(B,2)
            for q=1:size(vecs_sibling,1)
                for p=1:size(bvecs_ii,2)
                    for l=1:size(bvecs_ii,1)
                        bvecs_ii[l,p,k] += B[p + (q-1)*size(bvecs_ii,2),r] * bvecs_parent[l,r,k] * vecs_sibling[q,1,k]
                    end
                end
            end
        end
    end
    return nothing
end

function VmulBmulBVp!(bvecs_ii::AbstractArray{T,3}, vecs_sibling::AbstractArray{T,3}, B::AbstractArray{T,2}, bvecs_parent::AbstractArray{T,3}) where T
    for k=1:size(bvecs_ii,3)
        for r=1:size(B,2)
            for q=1:size(bvecs_ii,2)
                for p=1:size(vecs_sibling,1)
                    for l=1:size(bvecs_ii,1)
                        bvecs_ii[l,q,k] += B[p + (q-1)*size(vecs_sibling,1),r] * bvecs_parent[l,r,k] * vecs_sibling[p,1,k]
                    end
                end
            end
        end
    end
    return nothing
end

# Only supports full contraction
# dt is used for second derivatives. dt replaces X at node [rep].
# This is the same as taking the derivative w.r.t. node [rep] and multiplying by dt[rep]
# topdata is used to contract with tensor output
function tensorBVecsIndexed!(DV::diadicVectors{T}, M::TensorManifold{field}, X, topdata, ii; dt::ProductRepr = ProductRepr(), rep::Integer = 0) where {field, T}
    # find the parent and multiply with the vecs from the other brancs and the becs fron the bottom
    if DV.valid_bvecs[ii]
        return nothing
    end
    datalen = size(DV.vecs[1],3)
    if ii==1
        if topdata == nothing
            DV.bvecs[ii] = zeros(T, size(X.parts[ii],2), size(X.parts[ii],2), datalen)
            for k=1:size(X.parts[ii],2)
                DV.bvecs[ii][k,k,:] .= 1
            end
        else
            DV.bvecs[ii] = zeros(T, 1, size(X.parts[ii],2), datalen)
            DV.bvecs[ii][1,:,:] .= topdata
        end
    else
#         @show "problem line\n"
        # find parent. Everything has a parent!
        left = findfirst(isequal(ii), M.children[:,1])
        right = findfirst(isequal(ii), M.children[:,2])
        if left != nothing
            # B[left,right,parent] <- parent = BVecs[parent], right = vecs[sibling]
            parent = left
            if parent != rep
                B = X.parts[parent]
            else
                B = dt.parts[parent]
            end
            s_l = size(X.parts[M.children[parent,1]],2)
            s_r = size(X.parts[M.children[parent,2]],2)
            sibling = M.children[left,2] # right sibling
            tensorBVecsIndexed!(DV, M, X, topdata, parent; dt = dt, rep = rep)
#             @show size(X.B[parent]), size(DV.bvecs[parent]), size(vec(vecs[sibling]))
            DV.bvecs[ii] = zeros(T, size(DV.bvecs[parent],1), s_l, datalen)
#             @show ii, parent, size(B), size(vecs[sibling]), size(DV.bvecs[parent])
            BVpmulBmulV!(DV.bvecs[ii], B, DV.vecs[sibling], DV.bvecs[parent])
#             for r=1:size(B,2)
#                 for q=1:s_r
#                     for p=1:s_l
#                         @views DV.bvecs[ii][k, p,:] .+= B[p + (q-1)*s_l,r] .* DV.bvecs[parent][k, r,:] .* DV.vecs[sibling][q,1,:]
#                     end
#                 end
#             end
        end
        if right != nothing
            # B[left,right,parent] <- parent = BVecs[parent], right = vecs[sibling]
            parent = right
            if parent != rep
                B = X.parts[parent]
            else
                B = dt.parts[parent]
            end
            s_l = size(X.parts[M.children[parent,1]],2)
            s_r = size(X.parts[M.children[parent,2]],2)

            sibling = M.children[right,1] # right sibling
            tensorBVecsIndexed!(DV, M, X, topdata, parent; dt = dt, rep = rep)
#             @show size(X.B[parent]), size(DV.bvecs[parent]), size(vec(vecs[sibling]))
            DV.bvecs[ii] = zeros(T, size(DV.bvecs[parent],1), s_r, datalen)
            VmulBmulBVp!(DV.bvecs[ii], DV.vecs[sibling], B, DV.bvecs[parent])
#             for r=1:size(B,2)
#                 for q=1:s_r
#                     for p=1:s_l
#                         @views DV.bvecs[ii][k,q,:] .+= B[p + (q-1)*s_l,r] .* DV.bvecs[parent][k,r,:] .* DV.vecs[sibling][p,1,:]
#                     end
#                 end
#             end
        end
    end
    DV.valid_bvecs[ii] = true
    return nothing
end

# if ii == 0, topdata is invalid.
# if ii > 0  -> all children nodes get invalidate
function tensorBVecsInvalidate(DV::diadicVectors{T}, M::TensorManifold{field}, ii) where {field, T}
    if ii == 0
        DV.valid_bvecs[1] = false
        tensorBVecsInvalidate(DV, M, 1)
    else
        if is_leaf(M, ii)
            # do nothing as it has no children
        else
            ii_left = M.children[ii,1]
            ii_right = M.children[ii,2]
            DV.valid_bvecs[ii_left] = false
            DV.valid_bvecs[ii_right] = false
            tensorBVecsInvalidate(DV, M, ii_left)
            tensorBVecsInvalidate(DV, M, ii_right)
        end
    end
    return nothing
end

function tensorBVecs!(DV::diadicVectors, M::TensorManifold{field}, X, topdata = nothing; dt::ProductRepr = ProductRepr(), rep::Integer = 0) where field
    if size(DV.vecs[1],2) != 1
        println("only supports full contraction")
        return nothing
    end
    datalen = size(DV.vecs[1],3)
    for ii=1:nr_nodes(M)
#         println("bvecs ", ii)
        tensorBVecsIndexed!(DV, M, X, topdata, ii; dt = dt, rep = rep)
    end
    return nothing
end

function makeCache(M::TensorManifold, X, data::Vector{Matrix{T}}, topdata = nothing) where T
    DV = diadicVectors(T, size(M.children, 1))
    tensorVecsRecursive!(DV, M, X, data, 1)
    tensorBVecs!(DV, M, X, topdata)
    return DV
end

# update the content which is invalidated
function updateCache!(DV::diadicVectors, M::TensorManifold, X, data::Vector{Matrix{T}}, topdata = nothing) where T
    tensorVecsRecursive!(DV, M, X, data, 1)
    tensorBVecs!(DV, M, X, topdata)
    return nothing
end

# So the gradient of
# L0 = QoUoy - PoUox
# loss = 1/2 L0^T . L0

function tensorLoss(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT)
    Uoy = Eval(MU, XU, [dataOUT])
    Uox = Eval(MU, XU, [dataIN])
    QoUoy = Eval(MQ, XQ, [Uoy])
    PoUox = Eval(MP, XP, [Uox])
    L0 = QoUoy .- PoUox
    return sum(L0 .^ 2)/2
end

# grad_P = - L0^T . DPoUox
# grad_Q = L0^T DQoUoy
# grad_U = L0^T (JQoUoy x DUoy - JPoUox x DUox)

function tensorGradientP(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii)
    Uoy = Eval(MU, XU, [dataOUT])
    Uox = Eval(MU, XU, [dataIN])
    QoUoy = Eval(MQ, XQ, [Uoy])
    PoUox = Eval(MP, XP, [Uox])
    L0 = QoUoy .- PoUox

    DVP = tensorVecs(MP, XP, [Uox])
    tensorBVecs!(DVP, MP, XP)
    return L0_DF(MP, XP, DVP, Uox, -1.0*L0, ii)
end

function tensorGradientQ(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii)
    Uoy = Eval(MU, XU, [dataOUT])
    Uox = Eval(MU, XU, [dataIN])
    QoUoy = Eval(MQ, XQ, [Uoy])
    PoUox = Eval(MP, XP, [Uox])
    L0 = QoUoy .- PoUox

    DVQ = tensorVecs(MQ, XQ, [Uoy])
    tensorBVecs!(DVQ, MQ, XQ)
    return L0_DF(MQ, XQ, DVQ, Uoy, L0, ii)
end

function tensorGradientU(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii)
    Uoy = Eval(MU, XU, [dataOUT])
    Uox = Eval(MU, XU, [dataIN])
    QoUoy = Eval(MQ, XQ, [Uoy])
    PoUox = Eval(MP, XP, [Uox])
    L0 = QoUoy .- PoUox
    
    # the Jacobians call tensorVecs for each leaf once, so they are expensive
    JQoUoy = Jacobian(MQ, XQ, Uoy)
    JPoUox = Jacobian(MP, XP, Uox)
    L0_JQoUoy = dropdims(sum(JQoUoy .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    L0_JPoUox = dropdims(sum(JPoUox .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    DVUoy = tensorVecs(MU, XU, [dataOUT])
    tensorBVecs!(DVUoy, MU, XU)
    DVUox = tensorVecs(MU, XU, [dataIN])
    tensorBVecs!(DVUox, MU, XU)
    return L0_DF(MU, XU, DVUoy, dataOUT, L0_JQoUoy, ii) .- L0_DF(MU, XU, DVUox, dataIN, L0_JPoUox, ii)
end

# hess_P = DPoUox^T x DPoUox
# hess_Q = DQoUoy^T x DQoUoy
# hess_U = 
#     + DUox^T x JPoUox^T x JPoUox x DUox   -> 1
#     + DUoy^T x JQoUoy^T x JQoUoy x DUoy   -> 2
#     - DUox^T x JPoUox^T x JQoUoy x DUoy   -> 3
#     - DUoy^T x JQoUoy^T x JPoUox x DUox   -> 3^T          
#     - DUox^T x L0J2PoUox x DUox
#     + DUoy^T x L0J2QoUoy x DUoy

function tensorGradientHessianP(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii)
    Uoy = Eval(MU, XU, [dataOUT])
    Uox = Eval(MU, XU, [dataIN])
    QoUoy = Eval(MQ, XQ, [Uoy])
    PoUox = Eval(MP, XP, [Uox])
    L0 = QoUoy .- PoUox

    DVP = tensorVecs(MP, XP, [Uox])
    tensorBVecs!(DVP, MP, XP)
    
    grad = L0_DF(MP, XP, DVP, Uox, -1.0*L0, ii)
    hess = DFoxT_DFox(MP, DVP, Uox, ii)
    return grad, hess
end

function tensorGradientHessianQ(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii)
    Uoy = Eval(MU, XU, [dataOUT])
    Uox = Eval(MU, XU, [dataIN])
    QoUoy = Eval(MQ, XQ, [Uoy])
    PoUox = Eval(MP, XP, [Uox])
    L0 = QoUoy .- PoUox

    DVQ = tensorVecs(MQ, XQ, [Uoy])
    tensorBVecs!(DVQ, MQ, XQ)
    
    grad = L0_DF(MQ, XQ, DVQ, Uoy, L0, ii)
    hess = DFoxT_DFox(MQ, DVQ, Uoy, ii)

    return grad, hess
end

function tensorGradientHessianU(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii)
#       DUox^T x JPoUox^T x JPoUox x DUox   -> 1
#     + DUoy^T x JQoUoy^T x JQoUoy x DUoy   -> 2
#     - DUox^T x JPoUox^T x JQoUoy x DUoy   -> 3
#     - DUoy^T x JQoUoy^T x JPoUox x DUox   -> 3^T
#     - DUox^T x L0J2PoUox x DUox
#     + DUoy^T x L0J2QoUoy x DUoy
    Uoy = Eval(MU, XU, [dataOUT])
    Uox = Eval(MU, XU, [dataIN])
    QoUoy = Eval(MQ, XQ, [Uoy])
    PoUox = Eval(MP, XP, [Uox])
    L0 = QoUoy .- PoUox

    JQoUoy = Jacobian(MQ, XQ, Uoy)
    JPoUox = Jacobian(MP, XP, Uox)
    
    # the hessians or P and Q
#     println("Hessian")
    J2QoUoy = Hessian(MQ, XQ, Uoy)
    J2PoUox = Hessian(MP, XP, Uox)
    L0J2QoUoy = dropdims(sum(J2QoUoy .* reshape(L0, size(L0,1), 1, 1, size(L0,2)), dims=1), dims=1)
    L0J2PoUox = dropdims(sum(J2PoUox .* reshape(L0, size(L0,1), 1, 1, size(L0,2)), dims=1), dims=1)
    
    DVUoy = tensorVecs(MU, XU, [dataOUT])
    tensorBVecs!(DVUoy, MU, XU)
    DVUox = tensorVecs(MU, XU, [dataIN])
    tensorBVecs!(DVUox, MU, XU)

    hess = DFT_JFT_JF_DF(MU, DVUox, DVUoy, JPoUox, JQoUoy, L0J2PoUox, L0J2QoUoy, dataIN, dataOUT, ii)

    # FOR THE GRADIENT
    L0_JQoUoy = dropdims(sum(JQoUoy .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    L0_JPoUox = dropdims(sum(JPoUox .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=1)
    # grad_U = L0^T (JQoUoy x DUoy - JPoUox x DUox)
    grad = L0_DF(MU, XU, DVUoy, dataOUT, L0_JQoUoy, ii) .- L0_DF(MU, XU, DVUox, dataIN, L0_JPoUox, ii)
    return grad, hess
end

# same as wDF, except that it only applies to node ii
function L0_DF(M::TensorManifold{field}, X, DV, data, L0, ii) where field
    if is_leaf(M, ii)
        XO = zeros(size(data,1), size(DV.bvecs[ii],2))
        for q=1:size(XO,2)
            for p=1:size(XO,2)
                for l=1:size(DV.bvecs[ii],1)
                    @inbounds XO[p,q] += sum(L0[l,:] .* data[p,:] .* DV.bvecs[ii][l,q,:])
                end
            end
        end
    else
        # it is a node
        s_l = size(X.parts[M.children[ii,1]],2)
        s_r = size(X.parts[M.children[ii,2]],2)
        XO = zeros(s_l*s_r, size(DV.bvecs[ii],2))
        for r=1:size(XO,2)
            for q=1:s_r
                for p=1:s_l
                    for l=1:size(DV.bvecs[ii],1)
                        @inbounds XO[p + (q-1)*s_l,r] += sum(L0[l,:] .* DV.vecs[M.children[ii,1]][p,1,:] .* DV.vecs[M.children[ii,2]][q,1,:] .* DV.bvecs[ii][l,r,:])
                    end
                end
            end
        end
    end
    return XO
end

# Hessian with respect to parameters in P and Q
# DF^T x DF
# this is a 4 index quantity
function DFoxT_DFox(M::TensorManifold{field}, DV, data, ii; scale = alwaysone()) where field
    if is_leaf(M, ii)
        XO = zeros(size(data,1), size(DV.bvecs[ii],2), size(data,1), size(DV.bvecs[ii],2))
        for l = 1:size(data,2)
            for p2 = 1:size(XO,3), q2 = 1:size(XO,4), q1 = 1:size(XO,2), p1 = 1:size(XO,1)
                for k = 1:size(DV.bvecs[ii],1)
                    @inbounds XO[p1,q1,p2,q2] += data[p1,l] * DV.bvecs[ii][k,q1,l] * DV.bvecs[ii][k,q2,l] * data[p2,l] / scale[l]
                end
            end
        end
    else
        # it is a node
        ii_left = M.children[ii,1]
        ii_right = M.children[ii,2]
        s_l = size(DV.vecs[ii_left],1)
        s_r = size(DV.vecs[ii_right],1)
        XO = zeros(s_l*s_r, size(DV.bvecs[ii],2), s_l*s_r, size(DV.bvecs[ii],2))
        for l = 1:size(data,2)
            for p2 = 1:s_l, r2 = 1:s_r, q2 = 1:size(XO,4), q1 = 1:size(XO,2), r1 = 1:s_r, p1 = 1:s_l
                for k = 1:size(DV.bvecs[ii],1)
                    @inbounds XO[p1 + (r1-1)*s_l,q1,p2 + (r2-1)*s_l,q2] += DV.vecs[ii_left][p1,1,l] * DV.vecs[ii_right][r1,1,l] * (DV.bvecs[ii][k,q1,l] * DV.bvecs[ii][k,q2,l]) * DV.vecs[ii_left][p2,1,l] * DV.vecs[ii_right][r2,1,l] / scale[l]
                end
            end
        end
    end
    return XO
end

function node_XO!(XO::AbstractArray{T,4}, coreX::AbstractArray{T,3}, coreY::AbstractArray{T,3}, coreXY::AbstractArray{T,3}, 
                  Xvecs_l::AbstractArray{T,3}, Yvecs_l::AbstractArray{T,3}, Xvecs_r::AbstractArray{T,3}, Yvecs_r::AbstractArray{T,3}) where T
    s_l = size(Xvecs_l,1)
    s_r = size(Xvecs_r,1)
    for l=1:size(coreX,3), p2 = 1:s_l, r2 = 1:s_r, q2 = 1:size(XO,4), q1 = 1:size(XO,2), r1 = 1:s_r, p1 = 1:s_l
        @inbounds XO[p1 + (r1-1)*s_l,q1,p2 + (r2-1)*s_l,q2] += (
            Xvecs_l[p1,1,l] * Xvecs_r[r1,1,l] * coreX[q1,q2,l] * Xvecs_l[p2,1,l] * Xvecs_r[r2,1,l]
            + Yvecs_l[p1,1,l] * Yvecs_r[r1,1,l] * coreY[q1,q2,l] * Yvecs_l[p2,1,l] * Yvecs_r[r2,1,l]
            - Xvecs_l[p1,1,l] * Xvecs_r[r1,1,l] * coreXY[q1,q2,l] * Yvecs_l[p2,1,l] * Yvecs_r[r2,1,l]
            - Xvecs_l[p2,1,l] * Xvecs_r[r2,1,l] * coreXY[q2,q1,l] * Yvecs_l[p1,1,l] * Yvecs_r[r1,1,l])
    end
    nothing
end

# Hessian with respect to parameters in U
# needs JPoUox, JQoUoy, DUox, DUoy
# then  L0J2PoUox, L0J2QoUoy
# results in 
#       DUox^T x JPoUox^T x JPoUox x DUox   -> 1
#     + DUoy^T x JQoUoy^T x JQoUoy x DUoy   -> 2
#     - DUox^T x JPoUox^T x JQoUoy x DUoy   -> 3
#     - DUoy^T x JQoUoy^T x JPoUox x DUox   -> 3^T
#     - DUox^T x L0J2PoUox x DUox
#     + DUoy^T x L0J2QoUoy x DUoy
# The Jacobians JX, JY are evaluated on the whole polynomial!
#        DFT_JFT_JF_DF(MU,                       DVUox,                 DVUoy,                 JPoUox,                 JQoUoy, 
function DFT_JFT_JF_DF(M::TensorManifold{field}, DVX::diadicVectors{T}, DVY::diadicVectors{T}, JX::AbstractArray{T,3}, JY::AbstractArray{T,3},
#                      L0J2PoUox,                 L0J2QoUoy,                 dataIN,                     dataOUT,                   ii)
                       L0J2X::AbstractArray{T,3}, L0J2Y::AbstractArray{T,3}, dataX::AbstractArray{T,2},  dataY::AbstractArray{T,2}, ii; scale = alwaysone()) where {field, T}
    # DF is left_vec x right_vec x bottom_vec
    # bottom_vec is the output, and it is a matrix, so it needs to be multiplied with JF
    coreX = zeros(T, size(DVX.bvecs[ii],2), size(DVX.bvecs[ii],2), size(dataX,2))
    coreY = zeros(T, size(DVY.bvecs[ii],2), size(DVY.bvecs[ii],2), size(dataY,2))
    coreXY = zeros(T, size(DVX.bvecs[ii],2), size(DVX.bvecs[ii],2), size(dataX,2))

    for l=1:size(coreX,3), q1 = 1:size(DVX.bvecs[ii],2), q2 = 1:size(DVX.bvecs[ii],2), k1 = 1:size(JX,2), k2 = 1:size(JX,2)
        for s = 1:size(JX,1)
            @inbounds coreX[q1,q2,l] += DVX.bvecs[ii][k1,q1,l] * JX[s,k1,l] * JX[s,k2,l] * DVX.bvecs[ii][k2,q2,l] / scale[l]
            @inbounds coreY[q1,q2,l] += DVY.bvecs[ii][k1,q1,l] * JY[s,k1,l] * JY[s,k2,l] * DVY.bvecs[ii][k2,q2,l] / scale[l]
            @inbounds coreXY[q1,q2,l] += DVX.bvecs[ii][k1,q1,l] * JX[s,k1,l] * JY[s,k2,l] * DVY.bvecs[ii][k2,q2,l] / scale[l]
        end
        @inbounds coreX[q1,q2,l] -= DVX.bvecs[ii][k1,q1,l] * L0J2X[k1,k2,l] * DVX.bvecs[ii][k2,q2,l]
        @inbounds coreY[q1,q2,l] += DVY.bvecs[ii][k1,q1,l] * L0J2Y[k1,k2,l] * DVY.bvecs[ii][k2,q2,l]
    end
    if is_leaf(M, ii)
        XO = zeros(T, size(dataX,1), size(DVX.bvecs[ii],2), size(dataX,1), size(DVX.bvecs[ii],2))
#         @show size(coreX,3) * size(XO,4) * size(XO,3) * size(XO,2) * size(XO,1)
        for l=1:size(coreX,3), q2 = 1:size(XO,4), p2 = 1:size(XO,3), q1 = 1:size(XO,2), p1 = 1:size(XO,1)
            @inbounds XO[p1,q1,p2,q2] += (
                 dataX[p1,l] * coreX[q1,q2,l] * dataX[p2,l]
                 + dataY[p1,l] * coreY[q1,q2,l] * dataY[p2,l]
                 - dataX[p1,l] * coreXY[q1,q2,l] * dataY[p2,l]
                 - dataX[p2,l] * coreXY[q2,q1,l] * dataY[p1,l] )
        end
    else
        # it is a node
        ii_left = M.children[ii,1]
        ii_right = M.children[ii,2]
        Xvecs_l = DVX.vecs[ii_left]
        Yvecs_l = DVY.vecs[ii_left]
        Xvecs_r = DVX.vecs[ii_right]
        Yvecs_r = DVY.vecs[ii_right]
        s_l = size(Xvecs_l,1)
        s_r = size(Xvecs_r,1)
        XO = zeros(T, s_l*s_r, size(DVX.bvecs[ii],2), s_l*s_r, size(DVX.bvecs[ii],2))
#         @show size(coreX,3) * s_l * s_r * size(XO,4) * size(XO,2) * s_r * s_l
        node_XO!(XO, coreX, coreY, coreXY, Xvecs_l, Yvecs_l, Xvecs_r, Yvecs_r)
#         for l=1:size(coreX,3), p2 = 1:s_l, r2 = 1:s_r, q2 = 1:size(XO,4), q1 = 1:size(XO,2), r1 = 1:s_r, p1 = 1:s_l
#             @inbounds XO[p1 + (r1-1)*s_l,q1,p2 + (r2-1)*s_l,q2] += (
#                 Xvecs_l[p1,1,l] * Xvecs_r[r1,1,l] * coreX[q1,q2,l] * Xvecs_l[p2,1,l] * Xvecs_r[r2,1,l]
#                 + Yvecs_l[p1,1,l] * Yvecs_r[r1,1,l] * coreY[q1,q2,l] * Yvecs_l[p2,1,l] * Yvecs_r[r2,1,l]
#                 - Xvecs_l[p1,1,l] * Xvecs_r[r1,1,l] * coreXY[q1,q2,l] * Yvecs_l[p2,1,l] * Yvecs_r[r2,1,l]
#                 - Xvecs_l[p2,1,l] * Xvecs_r[r2,1,l] * coreXY[q2,q1,l] * Yvecs_l[p1,1,l] * Yvecs_r[r1,1,l])
#         end
    end
    return XO
end

# this is not what is says on the tin
function wDF(M::TensorManifold{field}, X, data, topdata; second::Integer = 0, dt::ProductRepr = ProductRepr(), rep::Integer = 0) where field
#     if length(data) == 1
#         println("only supports full contraction")
#         return nothing
#     end
    XO = [zeros(Float64, 1, 0, 0) for k = 1:size(M.children, 1)]
    DV = tensorVecs(M, X, data; second = second, dt = dt, rep = rep)
    tensorBVecs!(DV, M, X, topdata; dt = dt, rep = rep)
    datalen = size(DV.vecs[1],3)
    for ii=size(M.children, 1):-1:1
        if is_leaf(M, ii)
            # it is a leaf
            dim = findfirst(isequal(ii), M.dim2ind)
            if second == dim
                wdim = 2
            elseif second > 0
                wdim = 1
            elseif dim > length(data)
                wdim = length(data)
            else
                wdim = dim
            end
            XO[ii] = zeros(size(DV.bvecs[ii],1), size(X.parts[ii],1), size(X.parts[ii],2))
            for q=1:size(XO[ii],3)
                for p=1:size(XO[ii],2)
                    for l=1:size(DV.bvecs[ii],1)
                        @inbounds XO[ii][l,p,q] += sum(data[wdim][p,:] .* DV.bvecs[ii][l,q,:])
                    end
                end
            end
        else
            # it is a node
            s_l = size(X.parts[M.children[ii,1]],2)
            s_r = size(X.parts[M.children[ii,2]],2)
            XO[ii] = zeros(size(DV.bvecs[ii],1), size(X.parts[ii],1), size(X.parts[ii],2))
            for r=1:size(XO[ii],3)
                for q=1:s_r
                    for p=1:s_l
                        for l=1:size(DV.bvecs[ii],1)
                            @inbounds XO[ii][l,p + (q-1)*s_l,r] += sum(DV.vecs[M.children[ii,1]][p,1,:] .* DV.vecs[M.children[ii,2]][q,1,:] .* DV.bvecs[ii][l,r,:])
                        end
                    end
                end
            end
        end
    end
    return ProductRepr(map(x->dropdims(x,dims=1), XO)...)
end

# this is like Eval:
#  -> we need to replace X with dt in all factors and sum them up
function DFdt(M::TensorManifold{field}, X, data, dt) where field
    DV = tensorVecs(M, X, data; dt = dt, rep = 1)
    tmp = deepcopy(DV)
    for ii=2:nr_nodes(M)
        invalidateVecs(tmp)
        tensorVecsRecursive!(tmp, M, X, data, 1; dt = dt, rep = ii)
        DV.vecs[1] .+= tmp.vecs[1]
    end
    return dropdims(DV.vecs[1], dims=2)
end

#
function DwDFdt(M::TensorManifold{field}, X, data, w, dt) where field
    res = wDF(M, X, data, w; dt = dt, rep = 1)
    res.parts[1] .= 0
    for ii=2:size(M.children, 1)
        tmp = wDF(M, X, data, w; dt = dt, rep = ii)
        tmp.parts[ii] .= 0
        res .+= tmp
    end
    return res
end

# topdata is v
# data[2] is w
function vD2Fw(M::TensorManifold{field}, X, data, topdata) where field
    res = zero(data[1])
    for k=1:length(M.dim2ind)
        for l=1:length(M.dim2ind)
            if k != l
                DV = tensorVecs(M, X, data; second = l, third = k)
                @views res .+= dropdims(sum(DV.vecs[1] .* reshape(topdata, size(topdata,1), 1, size(topdata,2)), dims=1), dims=1)
            end
        end
    end
    return res
end

function Gradient(M::TensorManifold{field}, X, data, topdata) where field
    deri = wDF(M, X, data, topdata)
    return ProductRepr(map(project!, M.M.manifolds, deri.parts, X.parts, deri.parts))
end

# this calculates the derivative times vector V
# 3 inputs d[1] * [nabla T(d[2])]
function wJF(M::TensorManifold{field}, X, data, topdata) where field
    res = zero(data[1])
    for k=1:length(M.dim2ind)
        DV = tensorVecs(M, X, data; third = k)
        @views res .+= dropdims(sum(DV.vecs[1] .* reshape(topdata, size(topdata,1), 1, size(topdata,2)), dims=1), dims=1)
    end
    return res
end

function DwJFv(M::TensorManifold{field}, X, data, topdata) where field
    res = wDF(M, X, data, topdata; second = 1)
    for k=2:length(M.dim2ind)
        res .+= wDF(M, X, data, topdata; second = k)
    end
    return res
end

function DwJFdt(M::TensorManifold{field}, X, data, topdata, dt) where field
    RTV = tensorVecs(M, X, data; third = 1, dt = dt, rep = 1)
    for k=1:length(M.dim2ind)
        for ii=1:nr_nodes(M)
            if (k==1) && (ii==1)
                # skip what we already calculated
                continue
            end
            tmp = tensorVecs(M, X, data; third = k, dt = dt, rep = ii)
            RTV.vecs[1] .+= tmp.vecs[1]
        end
    end
    return dropdims(sum(RTV.vecs[1] .* reshape(topdata, size(topdata,1), 1, size(topdata,2)), dims=1), dims=1)
end

# evaluates the Jacobian at 'data' as many times as there are columns in 'data'
# output dim 1: row, dim 2: column, dim 3: the column of 'data'
function Jacobian(M::TensorManifold{field}, X, data) where field
    res = zeros(size(X.parts[1],2), size(data,1), size(data,2))
    # needs to add up all permutations
    for k=1:length(M.dim2ind)
        DV = tensorVecs(M, X, [data]; third = k)
#         @show size(res), size(vecs[1])
        @views res .+= DV.vecs[1]
    end
    return res
end

function testTensor()
    M1 = MinimalTensorManifold([4,4,4,4], 3)
    M2 = RandomTensorManifold([4,4,4,4], 3)
    zero(M1)
    x1 = randn(M1)
    zero(M2)
    x2 = randn(M2)
    dataIN = randn(4,10)
    dataIN2 = randn(4,10)
    dataOUT = randn(3,10)

    getel(M1, x1, (1, 3, 2, 4, 3))
    getel(M2, x2, (1, 3, 2, 4, 3))
    Eval(M1, x1, [dataIN], dataOUT)
    Eval(M2, x2, [dataIN], dataOUT)
    wDF(M1, x1, [dataIN], dataOUT)
    
    Gradient(M1, x1, [dataIN], dataOUT)
    Gradient(M2, x2, [dataIN], dataOUT)
    wJF(M1, x1, [dataIN], dataOUT)
    wJF(M2, x2, [dataIN], dataOUT)
    
    grad = wDF(M2, x2, [dataIN], dataOUT)
    xp = deepcopy(x2)
    gradp = deepcopy(grad)
    eps = 1e-6
    flag = false
    for k=1:length(x2.parts)
        for l=1:length(x2.parts[k])
            xp.parts[k][l] += eps
            gradp.parts[k][l] = sum(Eval(M2, xp, [dataIN], dataOUT) - Eval(M2, x2, [dataIN], dataOUT)) / eps
            relErr = (gradp.parts[k][l] - grad.parts[k][l]) / grad.parts[k][l]
            if abs(relErr) > 1e-4
                flag = true
                println("k = ", k, "/", length(x2.parts), " leaf=", is_leaf(M2,k), " l = ", l, "/", length(x2.parts[k]), " E = ", relErr)
            end
            xp.parts[k][l] = x2.parts[k][l]
        end
    end
    if flag
        println("Tensor wDF")
        @show M2.children
        @show diff = gradp - grad
        return nothing
    end
    
    # DFdt
    w = randn(M2)
    grad = DFdt(M2, x2, [dataIN], w)

    xp = deepcopy(x2)
    gradp = zero(grad)
    eps = 1e-6
    flag = false
    for k=1:length(x2.parts)
        for l=1:length(x2.parts[k])
            xp.parts[k][l] += eps
            tmp = (Eval(M2, xp, [dataIN]) - Eval(M2, x2, [dataIN])) / eps
            gradp .+= tmp * w.parts[k][l]
            xp.parts[k][l] = x2.parts[k][l]
        end
    end
    if flag
        println("Tensor DFdt")
        @show maximum(abs.(gradp - grad))
    end
    
    # now the hessian
    w = randn(M2)
    hess = DwDFdt(M2, x2, [dataIN], dataOUT, w)
    
    # test accuracy
    xp = deepcopy(x2)
    hessp = deepcopy(hess)
    eps = 1e-6
    flag = false
    for k=1:length(x2.parts)
        for l=1:length(x2.parts[k])
            xp.parts[k][l] += eps
#             tmp = map((x,y) -> broadcast(*, x, y), (wDF(M2, xp, [dataIN], dataOUT) .- wDF(M2, x2, [dataIN], dataOUT)).parts, w.parts)
#             d2 = mapreduce(sum, +, tmp)/eps
            hessp.parts[k][l] = inner(M2.M, x2, wDF(M2, xp, [dataIN], dataOUT) .- wDF(M2, x2, [dataIN], dataOUT), w)/eps
#             if abs(hessp.parts[k][l] - d2) > 1e-8
#                 println("@DIFFPROBLEM")
#             end
            relErr = (hessp.parts[k][l] - hess.parts[k][l]) / hess.parts[k][l]
            if abs(relErr) > 1e-4
                flag = true
                println("k = ", k, "/", length(x2.parts), " leaf=", is_leaf(M2,k), " l = ", l, " E = ", relErr)
            end
            xp.parts[k][l] = x2.parts[k][l]
        end
    end
    if flag
        println("Tensor DwDFdt")
        @show diff = hessp - hess
        @show diff.parts[3]
        @show hessp.parts[3]
        @show hess.parts[3]
    end

    # test wJF
    # the Jacobian is a list of matrices
    # that is Eval differentiated with respect to dimensions, but for each element in the list at the same time
    println("Tensor wJF")
    res_orig = wJF(M2, x2, [dataIN], dataOUT)
    eps = 1e-6
    res = deepcopy(res_orig)
    dataINp = deepcopy(dataIN)
    for k=1:size(dataIN,1)
        dataINp[k,:] .+= eps
#         @show size(Eval(M2, x2, [dataINp], dataOUT))
        res[k,:] = (Eval(M2, x2, [dataINp], dataOUT) - Eval(M2, x2, [dataIN], dataOUT)) / eps
        dataINp[k,:] = dataIN[k,:]
    end
    @show maximum(abs.(res_orig .- res))

    println("Tensor Jacobian")
    res_orig = Jacobian(M2, x2, dataIN)
    eps = 1e-6
    res = deepcopy(res_orig)
    dataINp = deepcopy(dataIN)
    for k=1:size(dataIN,1)
        dataINp[k,:] .+= eps
#         @show size(Eval(M2, x2, [dataINp], dataOUT))
        res[:,k,:] = (Eval(M2, x2, [dataINp]) - Eval(M2, x2, [dataIN])) / eps
        dataINp[k,:] = dataIN[k,:]
    end
    @show maximum(abs.(res_orig .- res))
    
    println("Tensor vD2Fw")
    res_orig = vD2Fw(M2, x2, [dataIN, dataIN2], dataOUT)
    eps = 1e-6
    res = deepcopy(res_orig)
    dataINp = deepcopy(dataIN)
    for k=1:size(dataIN,1)
        dataINp[k,:] .+= eps
        tmp = (wJF(M2, x2, [dataINp], dataOUT) - wJF(M2, x2, [dataIN], dataOUT)) / eps
        res[k,:] = dropdims(sum(tmp .* dataIN2,dims=1),dims=1)
        dataINp[k,:] = dataIN[k,:]
    end
    @show maximum(abs.(res_orig .- res))

    return nothing
end

function testTensorLoss()
    dataIN = rand(4,100)
    dataOUT = 0.2*rand(4,100)
    MP = RandomTensorManifold([2,2,2,2], 2)
    MQ = RandomTensorManifold([2,2,2,2], 2)
    MU = RandomTensorManifold([4,4,4,4], 2)

    XP = randn(MP)
    XQ = randn(MQ)
    XU = randn(MU)
    @time tensorLoss(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT)
    @time tensorLoss(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT)

    @time gradP = tensorGradientP(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, 1)
    @time gradP = tensorGradientP(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, 1)
    
    # checking straight hessian
    flag = false
    dataINp = deepcopy(dataIN)
    eps = 1e-6
    hessU = Hessian(MU, XU, dataIN)
    hessUp = deepcopy(hessU)
    for k=1:size(dataIN,1)
        dataINp[k,:] .+= eps
        hessUp[:,:,k,:] = (Jacobian(MU, XU, dataINp) .- Jacobian(MU, XU, dataIN)) / eps
        dataINp[k,:] .= dataIN[k,:]
    end
    println("Hess diff=", maximum(abs.(hessU .- hessUp)))
#     display(hessU .- permutedims(hessUp,[1,2,3,4]))
        
    # checking P derivatives
    flag = false
    XPp = deepcopy(XP)
    eps = 1e-6
    for ii=1:nr_nodes(MP)
        gradP = tensorGradientP(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii)
        gradPp = deepcopy(gradP)
        for k1=1:size(XP.parts[ii],1), k2=1:size(XP.parts[ii],2)
            XPp.parts[ii][k1,k2] += eps
#             @show gradPp[k1,k2]
#             @show gradP[k1,k2]
            gradPp[k1,k2] = (tensorLoss(MP, MQ, MU, XPp, XQ, XU, dataIN, dataOUT) - tensorLoss(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT))/eps
            relErr = (gradPp[k1,k2] - gradP[k1,k2]) / gradP[k1,k2]
            if abs(relErr) > 1e-4
                flag = true
                println("GP node=", ii, "/", nr_nodes(MP), " el=", k1, ",", k2, "/", size(XP.parts[ii],1), ",", size(XP.parts[ii],2), " E = ", relErr, " G=", gradP[k1,k2], " A=", gradPp[k1,k2])
            end
            XPp.parts[ii][k1,k2] = XP.parts[ii][k1,k2]
        end
    end
    
    # checking Q derivatives
    flag = false
    XQp = deepcopy(XQ)
    eps = 1e-6
    for ii=1:nr_nodes(MQ)
        gradQ = tensorGradientQ(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii)
        gradQp = deepcopy(gradQ)
        for k1=1:size(XQ.parts[ii],1), k2=1:size(XQ.parts[ii],2)
            XQp.parts[ii][k1,k2] += eps
#             @show gradQp[k1,k2]
#             @show gradQ[k1,k2]
            gradQp[k1,k2] = (tensorLoss(MP, MQ, MU, XP, XQp, XU, dataIN, dataOUT) - tensorLoss(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT))/eps
            relErr = (gradQp[k1,k2] - gradQ[k1,k2]) / gradQ[k1,k2]
            if abs(relErr) > 1e-4
                flag = true
                println("GQ node=", ii, "/", nr_nodes(MP), " el=", k1, ",", k2, "/", size(XP.parts[ii],1), ",", size(XP.parts[ii],2), " E = ", relErr, " G=", gradQ[k1,k2], " A=", gradQp[k1,k2])
            end
            XQp.parts[ii][k1,k2] = XQ.parts[ii][k1,k2]
        end
    end
    
    # checking U derivatives
    flag = false
    XUp = deepcopy(XU)
    eps = 1e-6
    for ii=1:nr_nodes(MU)
        gradU = tensorGradientU(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii)
        gradUp = deepcopy(gradU)
        for k1=1:size(XU.parts[ii],1), k2=1:size(XU.parts[ii],2)
            XUp.parts[ii][k1,k2] += eps
#             @show gradUp[k1,k2]
#             @show gradU[k1,k2]
            gradUp[k1,k2] = (tensorLoss(MP, MQ, MU, XP, XQ, XUp, dataIN, dataOUT) - tensorLoss(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT))/eps
            relErr = (gradUp[k1,k2] - gradU[k1,k2]) / gradU[k1,k2]
            if abs(relErr) > 1e-4
                flag = true
                println("GU node=", ii, "/", nr_nodes(MU), " el=", k1, ",", k2, "/", size(XU.parts[ii],1), ",", size(XU.parts[ii],2), " E = ", relErr, " G=", gradU[k1,k2], " A=", gradUp[k1,k2])
            end
            XUp.parts[ii][k1,k2] = XU.parts[ii][k1,k2]
        end
    end

    # checking P hessians
    flag = false
    XPp = deepcopy(XP)
    eps = 1e-6
    for ii=1:nr_nodes(MP)
        gradP, hessP = tensorGradientHessianP(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii)
        hessPp = deepcopy(hessP)
#         @show size(XP.parts[ii])
        for k1=1:size(XP.parts[ii],1), k2=1:size(XP.parts[ii],2)
            XPp.parts[ii][k1,k2] += eps
            hessPp[:,:,k1,k2] .= (tensorGradientP(MP, MQ, MU, XPp, XQ, XU, dataIN, dataOUT, ii) .- tensorGradientP(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii))/eps
            relErr = maximum(abs.((hessPp[:,:,k1,k2] - hessP[:,:,k1,k2]) ./ hessP[:,:,k1,k2]))
            if abs(relErr) > 1e-4
                flag = true
                println("HP node=", ii, "/", nr_nodes(MP), " el=", k1, ",", k2, "/", size(XP.parts[ii],1), ",", size(XP.parts[ii],2), " E = ", relErr, " HP=", maximum(abs.(hessP[:,:,k1,k2])), " A=", maximum(abs.(hessPp[:,:,k1,k2])))
                println("diff")
                display(hessPp[:,:,k1,k2] - hessP[:,:,k1,k2])
                println("analytic")
                display(hessP[:,:,k1,k2])
                println("approximate")
                display(hessPp[:,:,k1,k2])
            end
            XPp.parts[ii][k1,k2] = XP.parts[ii][k1,k2]
        end
    end

    # checking Q hessians
    flag = false
    XQp = deepcopy(XQ)
    eps = 1e-6
    for ii=1:nr_nodes(MQ)
        gradQ, hessQ = tensorGradientHessianQ(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii)
        hessQp = deepcopy(hessQ)
#         @show size(XQ.parts[ii])
        for k1=1:size(XQ.parts[ii],1), k2=1:size(XQ.parts[ii],2)
            XQp.parts[ii][k1,k2] += eps
            hessQp[:,:,k1,k2] .= (tensorGradientQ(MP, MQ, MU, XP, XQp, XU, dataIN, dataOUT, ii) .- tensorGradientQ(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii))/eps
            relErr = maximum(abs.((hessQp[:,:,k1,k2] - hessQ[:,:,k1,k2]) ./ hessQ[:,:,k1,k2]))
            if abs(relErr) > 1e-4
                flag = true
                println("HQ node=", ii, "/", nr_nodes(MQ), " el=", k1, ",", k2, "/", size(XQ.parts[ii],1), ",", size(XQ.parts[ii],2), " E = ", relErr, " HQ=", maximum(abs.(hessQ[:,:,k1,k2])), " A=", maximum(abs.(hessQp[:,:,k1,k2])))
                println("diff")
                display(hessQp[:,:,k1,k2] - hessQ[:,:,k1,k2])
                println("analytic")
                display(hessQ[:,:,k1,k2])
                println("approximate")
                display(hessQp[:,:,k1,k2])
            end
            XQp.parts[ii][k1,k2] = XQ.parts[ii][k1,k2]
        end
    end
        
    # checking U hessians
    flag = false
    XUp = deepcopy(XU)
    eps = 1e-6
    for ii=1:nr_nodes(MU)
        gradU, hessU = tensorGradientHessianU(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii)
        hessUp = deepcopy(hessU)
#         @show size(XU.parts[ii])
        for k1=1:size(XU.parts[ii],1), k2=1:size(XU.parts[ii],2)
            XUp.parts[ii][k1,k2] += eps
            hessUp[:,:,k1,k2] .= (tensorGradientU(MP, MQ, MU, XP, XQ, XUp, dataIN, dataOUT, ii) .- tensorGradientU(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii))/eps
            relErr = maximum(abs.((hessUp[:,:,k1,k2] - hessU[:,:,k1,k2]) ./ hessU[:,:,k1,k2]))
            if abs(relErr) > 1e-4
                flag = true
                println("HU node=", ii, "/", nr_nodes(MQ), " leaf = ", is_leaf(MU,ii) , " el=", k1, ",", k2, "/", size(XU.parts[ii],1), ",", size(XU.parts[ii],2), " E = ", relErr, " HU=", maximum(abs.(hessU[:,:,k1,k2])), " A=", maximum(abs.(hessUp[:,:,k1,k2])))
#                 println("diff")
#                 display(hessUp[:,:,k1,k2] - hessU[:,:,k1,k2])
#                 println("analytic")
#                 display(hessU[:,:,k1,k2])
#                 println("approximate")
#                 display(hessUp[:,:,k1,k2])
            end
            XUp.parts[ii][k1,k2] = XU.parts[ii][k1,k2]
        end
    end

    return nothing
end

function testTensorMinimize()
    dataIN = rand(4,10000)
    dataOUT = rand(4,10000)
    MP = RandomTensorManifold([2,2,2,2], 2)
    MQ = RandomTensorManifold([2,2,2,2], 2)
    MU = RandomTensorManifold([4,4,4,4], 2)
    XP = randn(MP)
    XQ = randn(MQ)
    XU = randn(MU)
   
    # we are solving the equation
    # DL + D2L X = 0
    # for each component, using the coordinate descent method
    @show tensorLoss(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT)
    for k=1:100
    println("U", k)
    for ii=2:nr_nodes(MU)
        G, H = tensorGradientHessianU(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii)
        delta = reshape(H, size(H,1)*size(H,2), :)\reshape(G,:)
        XU.parts[ii] .-= reshape(delta, size(G))
        @show tensorLoss(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT)
        orthog!(MU, XU)
        println("orthog")
        @show tensorLoss(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT)
    end
    println("P",k)
    for ii=1:nr_nodes(MP)
        G, H = tensorGradientHessianP(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii)
        delta = reshape(H, size(H,1)*size(H,2), :)\reshape(G,:)
        XP.parts[ii] .-= reshape(delta, size(G))
        @show tensorLoss(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT)
        orthog!(MP, XP)
        println("orthog")
        @show tensorLoss(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT)
    end
    println("Q",k)
    for ii=2:nr_nodes(MQ)
        G, H = tensorGradientHessianQ(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT, ii)
        delta = reshape(H, size(H,1)*size(H,2), :)\reshape(G,:)
        XQ.parts[ii] .-= reshape(delta, size(G))
        @show tensorLoss(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT)
        orthog!(MQ, XQ)
        println("orthog")
        @show tensorLoss(MP, MQ, MU, XP, XQ, XU, dataIN, dataOUT)
    end
    end
end
