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

# filter according to amplitude
# number of boxes to categorise
# removes the zeros from the input
function dataPrune(dataINarg, dataOUTarg; nbox=10, curve=3.4, perbox=300, retbox = nbox, scale = nothing, measure = nothing, cut = false)
    NDIM = length(dataINarg[:,1])
    # removes the zeros from the input
    if measure == nothing
        idnz = findall(sqrt.(dropdims(sum(dataINarg .^ 2, dims = 1),dims=1)) .> 1e-10)
        dataIN = @views dataINarg[:,idnz]
        dataOUT = @views dataOUTarg[:,idnz]
        iamps = sqrt.(dropdims(sum(dataIN .^ 2, dims = 1),dims=1))
    else
        idnz = findall(sqrt.(dropdims(sum((measure * dataINarg) .^ 2, dims = 1),dims=1)) .> 1e-10)
        dataIN = @views dataINarg[:,idnz]
        dataOUT = @views dataOUTarg[:,idnz]
        iamps = sqrt.(dropdims(sum((measure * dataIN) .^ 2, dims = 1),dims=1))
    end
#     @show iamps
    @show maximum(vec(iamps))
    imaxamp, imaxid = findmax(vec(iamps))
    lims = imaxamp*(range(0, 1, length=nbox+1) .^ curve)
    elems = [findall( (iamps .>= lims[k-1]) .& (iamps .<= lims[k]) ) for k=2:retbox+1]
    fulllens = [length(e) for e in elems]
    if cut
        lens = [min(length(e) < div(perbox,3) ? 0 : length(e), perbox) for e in elems]
    else
        lens = [min(length(e), perbox) for e in elems]
    end
    print("dataPrune: FULL BOX SIZES: ")
    display(Tuple(fulllens))
    print("dataPrune: CUT BOX SIZES:  ")
    display(Tuple(lens))
    
    cumlens = [0; cumsum(lens)]

    tmpIN = zeros(NDIM, cumlens[end])
    tmpOUT = zeros(NDIM, cumlens[end])
    scalePTS = zeros(cumlens[end])
    for k=1:length(elems)
        perm = randperm(length(elems[k]))
        tmpIN[:,1+cumlens[k]:cumlens[k+1]] .= dataIN[:, elems[k][perm[1:lens[k]]]]
        tmpOUT[:,1+cumlens[k]:cumlens[k+1]] .= dataOUT[:, elems[k][perm[1:lens[k]]]]
        if lens[k] > 0
            scalePTS[1+cumlens[k]:cumlens[k+1]] .= perbox/lens[k]
        else
            scalePTS[1+cumlens[k]:cumlens[k+1]] .= 1.0
        end
    end

    scaleIN = maximum(sqrt.(sum(tmpIN .^ 2, dims = 1)))
    scaleOUT = maximum(sqrt.(sum(tmpOUT .^ 2, dims = 1)))
    if scale == nothing
        datascale = max(scaleIN, scaleOUT)
    else
        datascale = scale
    end
    return tmpIN/datascale, tmpOUT/datascale, datascale, scalePTS
end

function PCAEmbed(zs, Tstep, dims, freqs)
    # PARAMATERS
    periods = 1
    embedfit = 1.0
    maxtrajlength = 12000
    # END PARAMETERS

    # here the window is the actual window
    window = Int(periods*floor(1/minimum(freqs)/Tstep)) # 2 full periods of the lowest frequency
    @show window
    skip = Int(ceil(1/Tstep/maximum(freqs)/5))
    @show skip

    trajnorms = [maximum(abs.(zs[k])) for k=1:length(zs)]
    maxtrajnorm = maximum(trajnorms)

    AA = zeros(window, window)
    for k=1:length(zs)
        if trajnorms[k] <= maxtrajnorm * embedfit
            for p=1:window, q=1:window
                @views AA[p,q] += dot(zs[k][p:end-(window-p)], zs[k][q:end-(window-q)]) / (length(zs[k]) - (window-1))
            end
#         @show k
        end
    end
    F = svd(AA)
    tab = F.Vt[1:dims,:]
    embedscales = tab[:,1]'
    println("Singular values")
    id = findall(F.S .> 1e-8)
    @show F.S[id]
    @show length(id)
    len = 0
    for k=1:length(zs)
        len += min(length(1:skip:(length(zs[k])-skip-(window)+1)), maxtrajlength)
#         @show min(length(1:skip:(length(zs[k])-skip-(window)+1)), maxtrajlength)
    end
#     @show len
    dataINor = zeros(size(tab,1), len)
    dataOUTor = zeros(size(tab,1), len)
    errIN =  zeros(len)
    errOUT = zeros(len)
    peaks = zeros(length(zs))
    p=1
    embederr = 0.0
    for k=1:length(zs)
        for q=1:skip:min(length(zs[k])-skip-(window)+1, maxtrajlength*skip)
            dataINor[:,p] .= tab*zs[k][q:(q+(window)-1)]
            dataOUTor[:,p] .= tab*zs[k][(q+skip):(q+skip+(window)-1)]
            rms = sqrt(sum(zs[k][q:(q+(window)-1)] .^ 2))/length(q:(q+(window)-1))
            if peaks[k] < rms
                peaks[k] = rms
            end
            errIN[p] = abs.(embedscales *  dataINor[:,p] - zs[k][q])
            errOUT[p] = abs.(embedscales *  dataOUTor[:,p] - zs[k][q+skip])
            p += 1
        end
    end
    println("PCAEmbed relative reproduction error = ", maximum(errIN./vec(sqrt.(sum(dataINor .^ 2,dims=1)))), ", ", maximum(errOUT./vec(sqrt.(sum(dataOUTor .^ 2,dims=1)))))
    @show sort(peaks)
#         @show sum(embedscales' * tab, dims=1)
    return dataINor, dataOUTor, Tstep*skip, embedscales, tab, errIN, errOUT
end

    function frequencyEmbed(zs, Tstep, freqs; period = 1)
        window = Int(floor(period/minimum(freqs)/Tstep/2)) # 1 * -> 2 * period; # 2 * the minimum period -> 2w+1 = 4 * the period
        @show window
        skip = Int(ceil(1/Tstep/maximum(freqs)/5))
        @show skip
        AA = zeros(2*window+1, 2*window+1)
        AA[1,:].=1/sqrt(2)
        for k=1:window
           AA[2*k,:] .= cos.(range(0,k*2*pi,length=2*window+2)[1:end-1])
           AA[2*k+1,:] .= sin.(range(0,k*2*pi,length=2*window+2)[1:end-1])
        end
        
        # MIDDLE
        m, delay = findmax(vec(minimum(AA .^ 2, dims=1)) .* [range(1,1.2,length=window); 1.21; range(1.2,1,length=window)])
        # BEG
#         m, delay = findmax(vec(minimum(AA .^ 2, dims=1)) .* range(1,1.1,length=2*window+1))
        # END
#         m, delay = findmax(vec(minimum(AA .^ 2, dims=1)) .* range(1.1,1.0,length=2*window+1))
        @show delay

        ffreqs = (1:window)/(Tstep*(2*window+1))
        midfreqs = [0; (freqs[2:end] + freqs[1:end-1])/2; Inf]
        tab = zeros(2*length(freqs), 2*window+1)
        embedscales = zeros(1,2*length(freqs))
        for k=1:length(midfreqs)-1
            ids = findall((ffreqs .> midfreqs[k]) .& (ffreqs .<= midfreqs[k+1]))
#             @show ids
#             @show ffreqs[ids]
            v = zero(AA[:,delay])
            v[2*ids] .= AA[2*ids,delay]
            # include the constant, to get rid of Gibb's phenomenon
            if k==1
                v[1] = AA[1,delay]
            end
            tab[2*k-1,:] = (transpose(v) * AA)
            embedscales[2*k-1] = norm(tab[2*k-1,:])/((2*window + 1)/2)
            tab[2*k-1,:] ./= norm(tab[2*k-1,:])
            v = zero(AA[:,delay])
            v[2*ids .+ 1] .= AA[2*ids .+ 1,delay]
            tab[2*k,:] = (transpose(v) * AA)
            embedscales[2*k] = norm(tab[2*k,:])/((2*window + 1)/2)
            tab[2*k,:] ./= norm(tab[2*k,:])
        end
#         tab[1,:] .+= AA[1,1]*AA[1,:]
#         tab ./= (window + 1/2)
        
        len = 0
        for k=1:length(zs)
            len += length(1:skip:(length(zs[k])-skip-(2*window+1)+1))
#             @show length(1:skip:(length(zs[k])-skip-(2*window+1)+1))
        end
#         @show len
        dataINor = zeros(size(tab,1), len)
        dataOUTor = zeros(size(tab,1), len)
        errIN =  zeros(len)
        errOUT = zeros(len)
        p=1
        for k=1:length(zs)
            for q=1:skip:(length(zs[k])-skip-(2*window+1)+1)
                dataINor[:,p] .= tab*zs[k][(q+(2*window+1)-1):-1:q]
                dataOUTor[:,p] .= tab*zs[k][(q+skip+(2*window+1)-1):-1:(q+skip)]
                errIN[p] = abs.(embedscales * dataINor[:,p] .- zs[k][q+(2*window+1-delay)])[1]
                errOUT[p] = abs.(embedscales * dataOUTor[:,p] .- zs[k][q+skip+(2*window+1-delay)])[1]
                p += 1
            end
        end
#         @show abs.(embedscales*tab) .> 1e-6
#         @show findall(abs.(embedscales*tab) .> 1e-6)
        println("frequencyEmbed relative reproduction error = ", maximum(errIN./vec(sqrt.(sum(dataINor .^ 2,dims=1)))), ", ", maximum(errOUT./vec(sqrt.(sum(dataOUTor .^ 2,dims=1)))))

#         @show sum(embedscales' * tab, dims=1)
        return dataINor, dataOUTor, Tstep*skip, embedscales
    end
    
# the output is
# Wout, Rout, PW, PR
# Wout : real submersion
# Rout : real vector field
@doc raw"""
    Wout, Rout, PW, PR = ISFNormalForm(M, X)
    
Calculates the normal form of the polynomial map represented by `M, X`. 
It returns the real normal form `Wout`, `Rout` 
and the complex normal form `PW`, `PR`
"""
function ISFNormalForm(M, X)
    MP, XP = toFullDensePolynomial(PadeP(M), PadePpoint(X))
    # usually NDIM = 2
    NDIM = size(PadePpoint(X),1)
    MWr, XWr, MRr, XRr, MW, XW, MR, XR = iFoliationMAP(MP, XP, collect(1:NDIM), [])
    return MWr, XWr, MRr, XRr, MW, XW, MR, XR
end
