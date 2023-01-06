using FoliationsManifoldsAutoencoders
using LinearAlgebra
using BSON: @load, @save
using Printf

function accuracy()
    NDIM = 16
#     for k=1:4
#         @load "data/sys10dimTrainPCA-$(NDIM)-$(k).bson" xs ys Tstep embedscales
#         SysName = "10dim-PCA-CAS4-$(NDIM)-$(k)"
#         @load "ISFdata-$(SysName).bson" Misf Xisf scale Tstep
#         
#         train = FoliationsManifoldsAutoencoders.ISFPadeLossInfinity(Misf, Xisf, xs./scale, ys./scale)
#         @load "data/sys10dimTestPCA-$(NDIM)-$(k).bson" xs ys Tstep embedscales
#         test = FoliationsManifoldsAutoencoders.ISFPadeLossInfinity(Misf, Xisf, xs./scale, ys./scale)
#         println("Train ", train, " Test ", test)
#     end
    # it makes no sense to test accuracy of 

    plot()
    println("FULL")
    pl = []
    for k=1:4
        @load "data/sys10dimTrainRED-$(k).bson" xs ys Tstep
        SysName = "10dim-FULL-CAS4-$(k)"

        @load "ISFdata-$(SysName).bson" Misf Xisf scale Tstep
        train = FoliationsManifoldsAutoencoders.ISFPadeLossInfinity(Misf, Xisf, xs./scale, ys./scale)
        hist = FoliationsManifoldsAutoencoders.ISFPadeLossHistogram(Misf, Xisf, xs./scale, ys./scale)
#         pl = scatter!(hist[1]*scale, hist[2])
        push!(pl, histogram2d(hist[1]*scale, hist[2], yscale=:identity, colorbar_scale=:log10, bins=(20,20),c=cgrad(:YlGnBu_9)))
        
        @load "data/sys10dimTestRED-$(k).bson" xs ys Tstep
        test = FoliationsManifoldsAutoencoders.ISFPadeLossInfinity(Misf, Xisf, xs./scale, ys./scale)
        println("Train ", (@sprintf "%.3e %.3e" train...), " Test ", (@sprintf "%.3e %.3e" test...))
    end
    display(plot(pl...))

    plot()
    println("DFT")
    for k=1:4
        @load "data/sys10dimTrainDFT-$(k).bson" xs ys Tstep embedscales
        SysName = "10dim-DFT-CAS4-$(k)"
        @load "ISFdata-$(SysName).bson" Misf Xisf scale Tstep
        train = FoliationsManifoldsAutoencoders.ISFPadeLossInfinity(Misf, Xisf, xs./scale, ys./scale)
        hist = FoliationsManifoldsAutoencoders.ISFPadeLossHistogram(Misf, Xisf, xs./scale, ys./scale)
        push!(pl, histogram2d(hist[1]*scale, hist[2], yscale=:identity, colorbar_scale=:log10, bins=(20,20),c=cgrad(:YlGnBu_9)))
        
        @load "data/sys10dimTestDFT-$(k).bson" xs ys Tstep embedscales
        test = FoliationsManifoldsAutoencoders.ISFPadeLossInfinity(Misf, Xisf, xs./scale, ys./scale)
        println("Train ", (@sprintf "%.3e %.3e" train...), " Test ", (@sprintf "%.3e %.3e" test...))
    end
    display(plot(pl...))
end

accuracy()
