
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
function TensorManifold(dims::Array{T,1}, topdim::T, ranks::Array{T,1}, children, dim2ind, tree_type = :balanced; field::AbstractNumbers=ℝ) where T <: Integer
    M = []
    R = []
    VT = []
    for ii = 1:nr_nodes(children)
        if is_leaf(children, ii)
            dim_id = findfirst(isequal(ii), dim2ind)
            n_ii = dims[dim_id]
            push!(M, Stiefel(n_ii, ranks[ii]))
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

# create a rank structure such that 'ratio' is the lost rank
function cascade_ranks(children, dim2ind, topdim, dims; node_ratio = 1.0, leaf_ratio = min(1.0, 2/minimum(dims)), max_rank = 18)
    ranks = zeros(size(children[:,1]))
    ranks[1] = topdim
    for ii = nr_nodes(children):-1:2
        if is_leaf(children, ii)
            ldim = dims[findfirst(isequal(ii), dim2ind)]
            ranks[ii] = ldim * leaf_ratio
            n_ii = ldim
            if ranks[ii] > min(n_ii, max_rank)
                ranks[ii] = min(n_ii, max_rank)
            end
        else
            ii_left = children[ii,1]
            ii_right = children[ii,2]
            ranks[ii] = ranks[ii_left] * ranks[ii_right] * node_ratio
            if ranks[ii] > min(ranks[ii_left] * ranks[ii_right], max_rank)
                ranks[ii] = min(ranks[ii_left] * ranks[ii_right], max_rank)
            end
        end
    end
    ranks_int = convert(typeof(children[:,1]), round.(ranks))
    return ranks_int
end

#  the output rank cannot be greater than the input rank
function prune_ranks!(dims::Array{T,1}, topdim::T, ranks::Array{T,1}, children, dim2ind) where T <: Integer
    ranks[1] = topdim
    for ii = nr_nodes(children):-1:2
        if is_leaf(children, ii)
            n_ii = dims[findfirst(isequal(ii), dim2ind)]
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

function MinimalTensorManifold(dims::Array{T,1}, topdim::T = 1, tree_type = :balanced) where T <: Integer
    children, dim2ind = define_tree(length(dims), tree_type)
    nodes = nr_nodes(children)
    # create ranks at each node
    ranks = ones(Int, nodes)
    # the root node is singular
    prune_ranks!(dims, topdim, ranks, children, dim2ind)
    return TensorManifold(dims, topdim, ranks, children, dim2ind, tree_type)
end

"""
    TODO: documentation
"""
function HTTensor(dims::Array{T,1}, topdim::T = 1, tree_type = :balanced; node_rank = 4) where T <: Integer
    children, dim2ind = define_tree(length(dims), tree_type)
    nodes = nr_nodes(children)
    # create ranks at each node
    ranks = cascade_ranks(children, dim2ind, topdim, dims, node_ratio = 1.0, leaf_ratio = min(1.0, node_rank/minimum(dims)), max_rank = 18)
    @show ranks
    return TensorManifold(dims, topdim, ranks, children, dim2ind, tree_type)
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
#     println("LmulBmulR!")
#     @time for k=1:size(vecs_right,3)
#         for jr = 1:size(vecs_right, 2)
#             for jl = 1:size(vecs_left, 2)
#                 for jb=1:size(B,2)
#                     for q=1:size(vecs_right, 1)
#                         for p=1:size(vecs_left, 1)
#                             @inbounds vecs_ii[jb, jr*jl, k] += vecs_left[p,jl,k] * B[p + (q-1)*size(vecs_left, 1), jb] .* vecs_right[q,jr,k]
#                         end
#                     end
#                 end
#             end
#         end
#     end
    C = reshape(B, size(vecs_left, 1), size(vecs_right, 1), :)
    if size(vecs_left,2) == 1
        @tullio vecs_ii[jb, jr, k] = vecs_left[p,1,k] * C[p,q,jb] * vecs_right[q,jr,k]
    elseif size(vecs_right,2) == 1
        @tullio vecs_ii[jb, jl, k] = vecs_left[p,jl,k] * C[p,q,jb] * vecs_right[q,1,k]
    end
    return nothing
end

function BmulData!(vecs_ii, B, data_wdim)
#     println("BmulData!")
#     @time for k=1:size(data_wdim,2)
#         for p=1:size(B,2)
#             for q=1:size(B,1)
#                 vecs_ii[p,1,k] += B[q,p] * data_wdim[q,k]
#             end
#         end
#     end
    @tullio vecs_ii[p,1,k] = B[q,p] * data_wdim[q,k]
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
function tensorVecsRecursive!(DV::diadicVectors{T}, M::TensorManifold{field}, X, data::Vararg{Matrix{T}}; ii) where {field, T}
#     print("v=", ii, "/", length(X.parts), "_")
    if DV.valid_vecs[ii]
        return nothing
    end
    B = X.parts[ii]
    if is_leaf(M, ii)
        # it is a leaf
        dim = findfirst(isequal(ii), M.dim2ind)
        if dim > length(data)
            wdim = length(data)
        else
            wdim = dim
        end
        if ~isassigned(DV.vecs, ii)
            DV.vecs[ii] = zeros(T, size(B,2), 1, size(data[1],2) ) # Array{T}(undef, (size(B,2), 1, size(data[1],2)))
        end
#             @show size(DV.vecs[ii]), size(B), size(data[wdim])
        BmulData!(DV.vecs[ii], B, data[wdim])
    else
        # it is a node
        ii_left = M.children[ii,1]
        ii_right = M.children[ii,2]
        tensorVecsRecursive!(DV, M, X, data..., ii = ii_left)
        tensorVecsRecursive!(DV, M, X, data..., ii = ii_right)
        vs_l = size(DV.vecs[ii_left], 2)
        vs_r = size(DV.vecs[ii_right], 2)
        vs = max(vs_l,vs_r)
        if ~isassigned(DV.vecs, ii)
            DV.vecs[ii] = zeros( size(B,2), vs, size(data[1],2) )
        end
        
        LmulBmulR!(DV.vecs[ii], DV.vecs[ii_left], B, DV.vecs[ii_right])
    end
    DV.valid_vecs[ii] = true
    return nothing
end

# invalidates vecs that are dependent on node "ii"
function tensorVecsInvalidate(DV::diadicVectors{T}, M::TensorManifold{field}, ii) where {field, T}
    DV.valid_vecs[ii] = false
    # not the root node
    if ii != 1
        # find parent. Everything has a parent!
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

# create partial results at the end of each node or leaf when multiplied with 'data'
function tensorVecs(M::TensorManifold{field}, X, data::Vararg{Matrix{T}}; second::Integer = 0, third::Integer = 0, dt::ProductRepr = ProductRepr(), rep::Integer = 0) where {field, T}
    DV = diadicVectors(T, size(M.children, 1))
    tensorVecsRecursive!(DV, M, X, data..., ii = 1)
    return DV
end

# L0 is a multiplication from the left
function Eval(M::TensorManifold{field}, X, data::Vararg{Matrix{T}}; L0 = nothing, DV::diadicVectors = tensorVecs(M, X, data)) where {T, field}
    if L0 == nothing
        return dropdims(DV.vecs[1], dims=2)
    else
        return dropdims(sum(DV.vecs[1] .* reshape(L0, size(L0,1), 1, size(L0,2)), dims=1), dims=(1,2))
    end
end


function BVpmulBmulV!(bvecs_ii::AbstractArray{T,3}, B::AbstractArray{U,2}, vecs_sibling::AbstractArray{T,3}, bvecs_parent::AbstractArray{T,3}) where {T, U}
    C = reshape(B, size(bvecs_ii, 2), size(vecs_sibling, 1), :)
    @tullio bvecs_ii[l,p,k] = C[p,q,r] * bvecs_parent[l,r,k] * vecs_sibling[q,1,k]
    return nothing
end

function VmulBmulBVp!(bvecs_ii::AbstractArray{T,3}, vecs_sibling::AbstractArray{T,3}, B::AbstractArray{U,2}, bvecs_parent::AbstractArray{T,3}) where {T, U}
    C = reshape(B, size(vecs_sibling, 1), size(bvecs_ii, 2), :)
    @tullio bvecs_ii[l,q,k] = C[p,q,r] * bvecs_parent[l,r,k] * vecs_sibling[p,1,k]
    return nothing
end

# Only supports full contraction
# dt is used for second derivatives. dt replaces X at node [rep].
# This is the same as taking the derivative w.r.t. node [rep] and multiplying by dt[rep]
# L0 is used to contract with tensor output
function tensorBVecsIndexed!(DV::diadicVectors{T}, M::TensorManifold{field}, X; ii) where {field, T}
#     print("b=", ii, "/", length(X.parts), "_")
    # find the parent and multiply with the vecs from the other brancs and the becs fron the bottom
    if DV.valid_bvecs[ii]
        return nothing
    end
    datalen = size(DV.vecs[1],3)
    if ii==1
        DV.bvecs[ii] = zeros(T, size(X.parts[ii],2), size(X.parts[ii],2), datalen)
        for k=1:size(X.parts[ii],2)
            DV.bvecs[ii][k,k,:] .= 1
        end
    else
        # find parent. Everything has a parent!
        left = findfirst(isequal(ii), M.children[:,1])
        right = findfirst(isequal(ii), M.children[:,2])
        if left != nothing
            # B[left,right,parent] <- parent = BVecs[parent], right = vecs[sibling]
            parent = left
            B = X.parts[parent]
            s_l = size(X.parts[M.children[parent,1]],2)
            s_r = size(X.parts[M.children[parent,2]],2)
            sibling = M.children[left,2] # right sibling
            tensorBVecsIndexed!(DV, M, X, ii = parent)
            if ~isassigned(DV.bvecs, ii)
                DV.bvecs[ii] = zeros(T, size(DV.bvecs[parent],1), s_l, datalen)
            end
            BVpmulBmulV!(DV.bvecs[ii], B, DV.vecs[sibling], DV.bvecs[parent])
        end
        if right != nothing
            # B[left,right,parent] <- parent = BVecs[parent], right = vecs[sibling]
            parent = right
            B = X.parts[parent]
            s_l = size(X.parts[M.children[parent,1]],2)
            s_r = size(X.parts[M.children[parent,2]],2)

            sibling = M.children[right,1] # right sibling
            tensorBVecsIndexed!(DV, M, X, ii = parent)
#             @show size(X.B[parent]), size(DV.bvecs[parent]), size(vec(vecs[sibling]))
            if ~isassigned(DV.bvecs, ii)
                DV.bvecs[ii] = zeros(T, size(DV.bvecs[parent],1), s_r, datalen)
            end
            VmulBmulBVp!(DV.bvecs[ii], DV.vecs[sibling], B, DV.bvecs[parent])
        end
    end
    DV.valid_bvecs[ii] = true
    return nothing
end


# if ii > 0  -> all children nodes get invalidate
function tensorBVecsInvalidate(DV::diadicVectors{T}, M::TensorManifold{field}, ii) where {field, T}
    # the root is always the identity, no need to update
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
    return nothing
end

function tensorBVecs!(DV::diadicVectors, M::TensorManifold{field}, X) where field
    if size(DV.vecs[1],2) != 1
        println("only supports full contraction")
        return nothing
    end
    datalen = size(DV.vecs[1],3)
    for ii=1:nr_nodes(M)
        tensorBVecsIndexed!(DV, M, X, ii = ii)
    end
    return nothing
end

function makeCache(M::TensorManifold, X, data::Vararg{Matrix{T}}) where T
    DV = diadicVectors(T, size(M.children, 1))
    tensorVecsRecursive!(DV, M, X, data...; ii = 1)
    tensorBVecs!(DV, M, X)
    return DV
end

# update the content which is invalidated
function updateCache!(DV::diadicVectors, M::TensorManifold, X, data::Vararg{Matrix{T}}) where T
    DV.valid_vecs .= false
    DV.valid_bvecs .= false
    tensorVecsRecursive!(DV, M, X, data..., ii = 1)
    tensorBVecs!(DV, M, X)
    return nothing
end

function updateCachePartial!(DV::diadicVectors, M::TensorManifold, X, data::Vararg{Matrix{T}}; ii) where T
    tensorVecsInvalidate(DV, M, ii)
    tensorBVecsInvalidate(DV, M, ii)
    # make sure to update all the marked components
    # Vecs can start at the root (ii = 1), hence it covers the full tree
    tensorVecsRecursive!(DV, M, X, data..., ii = 1)
    # BVecs has to start from all the leaves to cover the whole tree
    tensorBVecs!(DV, M, X)
    return nothing
end

# same as wDF, except that it only applies to node ii
function L0_DF_parts(M::TensorManifold{field}, X, data; L0, ii::Integer = -1, DV = makeCache(M, X, data)) where field
#     t0 = time()
    if is_leaf(M, ii)
        @tullio XO[p,q] := L0[l,k] * data[p,k] * DV.bvecs[ii][l,q,k]
    else
        # it is a node
        ch1 = DV.vecs[M.children[ii,1]]
        ch2 = DV.vecs[M.children[ii,2]]
        bv = DV.bvecs[ii]
        @tullio XOp[p,q,r] := L0[l,k] * ch1[p,1,k] * ch2[q,1,k] * bv[l,r,k]
        XO = reshape(XOp, :, size(XOp,3))
    end
#     t1 = time()
#     println("\n -> L0_DF = ", 100*(t1-t0))
    return XO
end

function L0_DF(M::TensorManifold{field}, X, data; L0, DV = makeCache(M, X, data)) where field
#     @show Tuple(collect(1:length(M.M.manifolds)))
# @show size(data[1]), size(L0), size(DV.bvecs)
    return ProductRepr(map((x) -> L0_DF_parts(M, X, data, L0 = L0, ii = x, DV = DV), Tuple(collect(1:length(M.M.manifolds)))) )
end

# L0 is a square matrix ...
function L0_DF1_DF2_parts(M::TensorManifold{field}, X, L0, dataX, dataY; ii::Integer = -1, DVX = makeCache(M, X, dataX), DVY = makeCache(M, X, dataY)) where field
    if is_leaf(M, ii)
        @tullio XO[p1,q1,p2,q2] := L0[r1,r2,k] * dataX[p1,k] * DVX.bvecs[ii][r1,q1,k] * dataY[p2,k] * DVY.bvecs[ii][r2,q2,k]
        return XO
    else
        # it is a node
#         @show ii, M.children[ii,1], M.children[ii,2]
        chX1 = DVX.vecs[M.children[ii,1]]
        chX2 = DVX.vecs[M.children[ii,2]]
        bvX = DVX.bvecs[ii]
        chY1 = DVY.vecs[M.children[ii,1]]
        chY2 = DVY.vecs[M.children[ii,2]]
        bvY = DVY.bvecs[ii]
        @tullio XOp[p1,q1,r1,p2,q2,r2] := L0[l1,l2,k] * chX1[p1,1,k] * chX2[q1,1,k] * bvX[l1,r1,k] * chY1[p2,1,k] * chY2[q2,1,k] * bvY[l2,r2,k]
        XO = reshape(XOp, size(XOp,1)*size(XOp,2), size(XOp,3), size(XOp,4)*size(XOp,5), size(XOp,6))
        return XO
    end
end

# instead of multiplying the gradient from the left, we are multiplying it from the right
# there is no contraction along the indices of data...
function DF_dt_parts(M::TensorManifold{field}, X, data; dt, ii, DV = makeCache(M, X, data)) where field
    if is_leaf(M, ii)
        @tullio XO[l,k] := DV.bvecs[ii][l,q,k] * data[p,k] * dt[p,q]
    else
        s_l = size(X.parts[M.children[ii,1]],2)
        s_r = size(X.parts[M.children[ii,2]],2)
        dtp = reshape(dt, s_l, s_r, :)
        @tullio XO[l,k] := DV.vecs[M.children[ii,1]][p,1,k] * DV.vecs[M.children[ii,2]][q,1,k] * DV.bvecs[ii][l,r,k] * dtp[p,q,r]
    end
#     @show size(data), size(dt), size(XO)
    return XO
end

function DF_dt(M::TensorManifold{field}, X, data; dt, DV = makeCache(M, X, data)) where field
    return ProductRepr(map((x, y) -> DF_dt_parts(M, X, data, dt = y, ii = x, DV = DV), Tuple(collect(1:length(M.M.manifolds))), dt.parts) )
end

# these are highly optimised and do not improve with tullio
function node_XO!(XO::AbstractArray{T,4}, coreX::AbstractArray{T,3}, coreY::AbstractArray{T,3}, coreXY::AbstractArray{T,3}, 
                  Xvecs_l::AbstractArray{T,3}, Yvecs_l::AbstractArray{T,3}, Xvecs_r::AbstractArray{T,3}, Yvecs_r::AbstractArray{T,3}) where T
#     println("XO node")
    s_l = size(Xvecs_l,1)
    s_r = size(Xvecs_r,1)
    for l=1:size(coreX,3)
        for p2 = 1:s_l, r2 = 1:s_r, q2 = 1:size(XO,4), q1 = 1:size(XO,2), r1 = 1:s_r, p1 = 1:s_l
            @inbounds XO[p1 + (r1-1)*s_l,q1,p2 + (r2-1)*s_l,q2] += (
                Xvecs_l[p1,1,l] * Xvecs_r[r1,1,l] * (coreX[q1,q2,l] * Xvecs_l[p2,1,l] * Xvecs_r[r2,1,l]
                - coreXY[q1,q2,l] * Yvecs_l[p2,1,l] * Yvecs_r[r2,1,l])
                + Yvecs_l[p1,1,l] * Yvecs_r[r1,1,l] * (coreY[q1,q2,l] * Yvecs_l[p2,1,l] * Yvecs_r[r2,1,l]
                - coreXY[q2,q1,l] * Xvecs_l[p2,1,l] * Xvecs_r[r2,1,l]))
        end
    end
#     @time @tullio XOp[p1,r1,q1,p2,r2,q2] := @inbounds (Xvecs_l[p1,1,l] * Xvecs_r[r1,1,l] * (coreX[q1,q2,l] * Xvecs_l[p2,1,l] * Xvecs_r[r2,1,l]
#             - coreXY[q1,q2,l] * Yvecs_l[p2,1,l] * Yvecs_r[r2,1,l])
#             + Yvecs_l[p1,1,l] * Yvecs_r[r1,1,l] * (coreY[q1,q2,l] * Yvecs_l[p2,1,l] * Yvecs_r[r2,1,l]
#             - coreXY[q2,q1,l] * Xvecs_l[p2,1,l] * Xvecs_r[r2,1,l]))
#     XO2 = reshape(XOp, s_l*s_r, size(coreXY,1), s_l*s_r, size(coreXY,2))
#     @show norm(XO .- XO2)
    nothing
end

# these are highly optimised and do not improve with tullio
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
    for l=1:size(coreX,3), q1 = 1:size(DVX.bvecs[ii],2), q2 = 1:size(DVX.bvecs[ii],2)
        cX = zero(T)
        cY = zero(T)
        cXY = zero(T)
        cnX = zero(T)
        cnY = zero(T)
        for k1 = 1:size(JX,2), k2 = 1:size(JX,2)
            cpX = zero(T)
            cpY = zero(T)
            cpXY = zero(T)
            for s = 1:size(JX,1)
                @inbounds cpX += JX[s,k1,l] * JX[s,k2,l]
                @inbounds cpY += JY[s,k1,l] * JY[s,k2,l]
                @inbounds cpXY += JX[s,k1,l] * JY[s,k2,l]
            end
            @inbounds cX += cpX * DVX.bvecs[ii][k1,q1,l] * DVX.bvecs[ii][k2,q2,l]
            @inbounds cY += cpY * DVY.bvecs[ii][k1,q1,l] * DVY.bvecs[ii][k2,q2,l]
            @inbounds cXY += cpXY * DVX.bvecs[ii][k1,q1,l] * DVY.bvecs[ii][k2,q2,l]
            @inbounds cnX -= DVX.bvecs[ii][k1,q1,l] * L0J2X[k1,k2,l] * DVX.bvecs[ii][k2,q2,l]
            @inbounds cnY += DVY.bvecs[ii][k1,q1,l] * L0J2Y[k1,k2,l] * DVY.bvecs[ii][k2,q2,l]
        end
        @inbounds coreX[q1,q2,l] = cX / scale[l] + cnX
        @inbounds coreY[q1,q2,l] = cY / scale[l] + cnY
        @inbounds coreXY[q1,q2,l] = cXY / scale[l]
    end
#     @time begin
#     @tullio coreX[q1,q2,l] := DVX.bvecs[ii][k1,q1,l] * JX[s,k1,l] * JX[s,k2,l] * DVX.bvecs[ii][k2,q2,l] / scale[l] - DVX.bvecs[ii][k1,q1,l] * L0J2X[k1,k2,l] * DVX.bvecs[ii][k2,q2,l]
#     @tullio coreY[q1,q2,l] := DVY.bvecs[ii][k1,q1,l] * JY[s,k1,l] * JY[s,k2,l] * DVY.bvecs[ii][k2,q2,l] / scale[l] + DVY.bvecs[ii][k1,q1,l] * L0J2Y[k1,k2,l] * DVY.bvecs[ii][k2,q2,l]
#     @tullio coreXY[q1,q2,l] := DVX.bvecs[ii][k1,q1,l] * JX[s,k1,l] * JY[s,k2,l] * DVY.bvecs[ii][k2,q2,l] / scale[l]
#     end
    if is_leaf(M, ii)
        XO = zeros(T, size(dataX,1), size(DVX.bvecs[ii],2), size(dataX,1), size(DVX.bvecs[ii],2))
        for l=1:size(coreX,3), q2 = 1:size(XO,4), p2 = 1:size(XO,3), q1 = 1:size(XO,2), p1 = 1:size(XO,1)
            @inbounds XO[p1,q1,p2,q2] += (
                 dataX[p1,l] * (coreX[q1,q2,l] * dataX[p2,l]
                 - coreXY[q1,q2,l] * dataY[p2,l])
                 + dataY[p1,l] * (coreY[q1,q2,l] * dataY[p2,l]
                 - coreXY[q2,q1,l] * dataX[p2,l]) )
        end
#         @time @tullio XO[p1,q1,p2,q2] := (
#                  dataX[p1,l] * (coreX[q1,q2,l] * dataX[p2,l]
#                  - coreXY[q1,q2,l] * dataY[p2,l])
#                  + dataY[p1,l] * (coreY[q1,q2,l] * dataY[p2,l]
#                  - coreXY[q2,q1,l] * dataX[p2,l]) )
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
        node_XO!(XO, coreX, coreY, coreXY, Xvecs_l, Yvecs_l, Xvecs_r, Yvecs_r)
    end
    return XO
end
