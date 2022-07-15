using FoliationsManifoldsAutoencoders
using BSON: @load, @save
using LinearAlgebra
using DifferentialEquations

function CalculateDecoder(MF, XF)
    MWr, XWr, MRr, XRr = iManifoldVF(MF, XF, [9, 10], [])
    return MWr, XWr
end

function PEV!(mexp, XF, x::Array{T,1}, y::Array{T,1}) where T
    x .= T(0)
    @inbounds for l=1:size(XF,2)
        b = T(1.0)
        @inbounds for p=1:size(mexp,1)
            b *= y[p] ^ mexp[p,l]
        end
        @inbounds for k=1:size(XF,1)
            x[k] += b * XF[k,l]
        end
    end
    return x
end



function generate(MF, XF, MW, XW)
    function model!(x, y, p, t)
        PEV!(MF.mexp, XF, vec(x),vec(y))
        return x
    end

    ndim = 2
    odim = 10
    nruns = 800 # 1000
    npoints = 24 # 16
    xs = zeros(odim, nruns*npoints)
    ys = zeros(odim, nruns*npoints)
    aa = rand(ndim,nruns) .- ones(ndim,nruns)/2
    ics =  aa ./ sqrt.(sum(aa.^2,dims=1)) .* (2*rand(1,nruns) .- 1)
    Tstep = 0.2
    for j=1:nruns
#             u0 = ics[:,j] * 1.2
        u0 = Eval(MW, XW, [ics[:,j] * 0.6])[:,1] # 0.6
#             @show findall(isequal(NaN), u0)
        tspan = (0.0,Tstep*npoints) # 51 intervals with T=0.8 as in Proc Roy Soc Paper
        prob = ODEProblem(model!, u0, tspan)
        @show j, nruns
        sol = DifferentialEquations.solve(prob, Vern7(), abstol = 1e-7, reltol = 1e-7)
#             sol = DifferentialEquations.solve(prob, Tsit5(), abstol = 1e-7, reltol = 1e-7)
        trange = range(tspan[1], tspan[2], length = npoints+1)
        dsol = sol(trange)
#             @show dsol
#             @show size(dsol[1:end-1]), size(xs[:,1+(j-1)*npoints:j*npoints])
#             @show eltype(dsol)
#             @show eltype(xs)
#             xs[:,1+(j-1)*npoints:j*npoints] .= W(dsol[:,1:end-1])
#             ys[:,1+(j-1)*npoints:j*npoints] .= W(dsol[:,2:end])
        xs[:,1+(j-1)*npoints:j*npoints] .= dsol[:,1:end-1]
        ys[:,1+(j-1)*npoints:j*npoints] .= dsol[:,2:end]
        @show norm(u0), size(xs), maximum(sqrt.(sum(ys .^ 2, dims=1))), minimum(sqrt.(sum(ys .^ 2, dims=1)))
    end
    return xs, ys, Tstep
end


function ManifoldGenerate()
    
    modelorder = 5
    if stat("../sysSynth10dimCRT-model.bson").inode == 0
        println("taylor expanding model!")
        include("sys10dimvectorfield.jl")
        function vfrhsPar(y)
            x = zero(y)
            vfrhs!(x, y, 0, 0)
            return x
        end
        MF = DensePolyManifold(10, 10, 5)
        XF = fromFunction(MF, vfrhsPar)
        @save "../sysSynth10dimCRT-model.bson" MF XF
    else
        println("loading model!")
        @load "../sysSynth10dimCRT-model.bson" MF XF
    end
    println("model done!")

    MW, XW = CalculateDecoder(MF, XF)

    xs, ys, Tstep = generate(MF, XF, MW, XW)
    @save "AENC-on-manifold-bw.bson" xs ys Tstep
end

ManifoldGenerate()
