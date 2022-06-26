module s
    using FoliationsManifoldsAutoencoders
    using DifferentialEquations
    using LinearAlgebra
    using BSON:@load,@save
    
    include("sys10dimvectorfield.jl")
    function vfrhsPar(y)
        x = zero(y)
        vfrhs!(x, y, 0, 0)
        return x
    end
    
    modelorder = 5
    if stat("../sysSynth10dimCRT-model.bson").inode == 0
        MF = DensePolyManifold(10, 10, 5)
        XF = fromFunction(MF, vfrhsPar)
        @save "../sysSynth10dimCRT-model.bson" MF XF
    else
        @load "../sysSynth10dimCRT-model.bson" MF XF
    end
    println("model done!")

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
    
    function model!(x, y, p, t)
        PEV!(MF.mexp, XF, x, y)
        return x
    end
    
    function generate(amp)
        ndim = 10
        nruns = 1000
        npoints = 16
        xs = zeros(ndim, nruns*npoints)
        ys = zeros(ndim, nruns*npoints)
        aa = rand(ndim,nruns) .- ones(ndim,nruns)/2
        ics =  aa ./ sqrt.(sum(aa.^2,dims=1)) .* (2*rand(1,nruns) .- 1)
        Tstep = 0.1
        for j=1:nruns
            u0 = ics[:,j] * amp
#             @show findall(isequal(NaN), u0)
            tspan = (0.0,Tstep*npoints) # 51 intervals with T=0.8 as in Proc Roy Soc Paper
            prob = ODEProblem(model!, u0, tspan)
            @show amp, j, nruns
            sol = solve(prob, Vern7(), abstol = 1e-8, reltol = 1e-8)
            trange = range(tspan[1], tspan[2], length = npoints+1)
            dsol = sol(trange)
#             @show dsol
#             @show size(dsol[1:end-1]), size(xs[:,1+(j-1)*npoints:j*npoints])
#             @show eltype(dsol)
#             @show eltype(xs)
            xs[:,1+(j-1)*npoints:j*npoints] .= dsol[:,1:end-1]
            ys[:,1+(j-1)*npoints:j*npoints] .= dsol[:,2:end]
        end
        return xs, ys, Tstep
    end
    
    amps = [0.8 1.0 1.2 1.4]
    for k=1:length(amps)
        xs, ys, Tstep = generate(amps[k])
        @save "sys10dimTrainRED-$(k).bson" xs ys Tstep
        
        # frequencies
        scale = sqrt.(sum(xs.^2,dims=1))
        A = ((ys.*scale)*transpose(xs)) * inv((xs.*scale)*transpose(xs))
        println("frequencies [Hz] = ", sort(unique(abs.(angle.(eigvals(A))/(2*pi))))/Tstep)
        println("frequencies [rad/s] = ", sort(unique(abs.(angle.(eigvals(A)))))/Tstep)

        xs, ys, Tstep = generate(amps[k])
        @save "sys10dimTestRED-$(k).bson" xs ys Tstep
    end
end
