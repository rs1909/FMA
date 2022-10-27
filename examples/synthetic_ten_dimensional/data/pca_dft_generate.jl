using FoliationsManifoldsAutoencoders
using LinearAlgebra
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
        PEV!(MF.mexp, XF,x,y)
        return x
    end
    
    function generate(maxIC)
        ndim = 10
        # PARAMATERS
        nruns = 200
        npoints = 2400
        Tstep = 0.01
        # END PARAMETERS
        zs = Array{Array{Float64},1}(undef,nruns)
        # setting initial conditions
        aa = rand(ndim,nruns) .- ones(ndim,nruns)/2
        ics =  aa ./ sqrt.(sum(aa.^2,dims=1)) .* rand(1,nruns)
        for j=1:nruns
            u0 = ics[:,j] * maxIC

            tspan = (0.0,Tstep*npoints) # 51 intervals with T=0.8 as in Proc Roy Soc Paper
            prob = ODEProblem(model!, u0, tspan)

            sol = solve(prob, Vern7(), abstol = 1e-8, reltol = 1e-8)
            trange = range(tspan[1], tspan[2], length = npoints+1)
#             @show length(1+(j-1)*npoints:j*npoints), size([sum(sol(t))/length(sol(t)) for t in trange])
            zs[j] = [sum(sol(t))/10 for t in trange]
            @show j, nruns
        end
        return zs, Tstep
    end
        
    freqs = [1, exp(1), sqrt(30), pi^2, 13]/(2*pi)
    
    amps = [0.8 1.0 1.2 1.4]
    PCAdim = 16
    for k=1:length(amps)
#         zs, TstepZS = generate(amps[k])
        @load "sys10dimTrainPCA-10-$(k).bson" zs TstepZS
        xst, yst, Tstep, embedscales = PCAEmbed(zs, TstepZS, PCAdim, freqs)
        xs, ys = dataPrune(xst, yst; nbox=20, curve=1, perbox=1000, scale = 1.0, measure = nothing)
        @save "sys10dimTrainPCA-$(PCAdim)-$(k).bson" xs ys Tstep zs TstepZS embedscales
#         xst, yst, Tstep, embedscales = frequencyEmbed(zs, TstepZS, freqs)
#         xs, ys = dataPrune(xst, yst; nbox=20, curve=1, perbox=1000, scale = 1.0, measure = nothing)
#         @save "sys10dimTrainDFT-$(k).bson" xs ys Tstep zs TstepZS embedscales
        
        # frequencies
        scale = sqrt.(sum(xs.^2,dims=1))
        A = ((ys.*scale)*transpose(xs)) * inv((xs.*scale)*transpose(xs))
        println("frequencies [Hz] = ", sort(unique(abs.(angle.(eigvals(A))/(2*pi))))/Tstep)
        println("frequencies [rad/s] = ", sort(unique(abs.(angle.(eigvals(A)))))/Tstep)

        @show size(xs)
        
#         zs, TstepZS = generate(amps[k])
# #        @load "sys10dimZSTestPCA-RED-$(k).bson" zs TstepZS
#         xst, yst, Tstep, embedscales = PCAEmbed(zs, TstepZS, PCAdim, freqs)
#         xs, ys = dataPrune(xst, yst; nbox=20, curve=1, perbox=1000, scale = 1.0, measure = nothing)
#         @save "sys10dimTestPCA-$(PCAdim)-$(k).bson" xs ys Tstep zs TstepZS embedscales
#         xst, yst, Tstep, embedscales = frequencyEmbed(zs, TstepZS, freqs)
#         xs, ys = dataPrune(xst, yst; nbox=20, curve=1, perbox=1000, scale = 1.0, measure = nothing)
#         @save "sys10dimTestDFT-$(k).bson" xs ys Tstep zs TstepZS embedscales
    end
    
end
