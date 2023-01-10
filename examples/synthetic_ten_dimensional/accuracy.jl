using FoliationsManifoldsAutoencoders
using LinearAlgebra
using BSON: @load, @save
using Printf
using LaTeXStrings
using Plots
using Plots.PlotMeasures

    pgfplotsx()
    push!(Plots.PGFPlotsX.CUSTOM_PREAMBLE,raw"\usepackage{amsmath,bm}")

function accuracyFULL()
    println("FULL")
    pl = []
    for it in [(1,:red, "(a) -- ST-1", 0) (2,:blue, "(d) -- ST-2", 1) (3,:green, "(g) -- ST-3", 2.1) (4,:black, "(j) -- ST-4", 3.1)]
        @load "data/sys10dimTrainRED-$(it[1]).bson" xs ys Tstep 
        embedscales = ones(1,10)/10
        SysName = "10dim-FULL-$(it[1])"
        @load "ISFdata-$(SysName).bson" Misf Xisf scale Tstep
        MU, XU, MS, XS = ISFNormalForm(Misf, Xisf)
        @load "ManifoldImmersion-$(SysName).bson" Mimm Xres Tstep scale dataParIN dataParOUT
        MWt, XWt = ImmersionReconstruct(Mimm, Xres, Misf, Xisf, MU, XU)

        on_manifold = Eval(MWt, XWt, Eval(MU, XU, Eval(PadeU(Misf), PadeUpoint(Xisf), xs./scale))) .* scale
        amplitude = vec(transpose(vec(embedscales)) * on_manifold)
        
        hist = FoliationsManifoldsAutoencoders.ISFPadeLossHistogram(Misf, Xisf, xs./scale, ys./scale)
        p = sortperm(amplitude)
        xlims = exp.(collect(range(log(minimum(hist[2])), log(maximum(hist[2])), length=30)))
        ylims = [amplitude[p[Int(round(k))]] for k in range(1,length(amplitude), length=30)]
        push!(pl, histogram2d(hist[2], amplitude, 
                              bins=(xlims,ylims), c=cgrad(:YlGnBu_9),
                              xscale=:log10, 
                              colorbar_scale=:identity, 
                              yformatter = :scientific,
                              ylims = [0, 0.07],
                              yticks = [0.01, 0.03, 0.05, 0.07],
                              xlims = [1e-5, 1e-1],
                              xticks = ([1e-5, 1e-3, 1e-1], [L"10^{-5}", L"10^{-3}", L"10^{-1}"]),
                              ylabel = "amplitude", 
                              xlabel = L"$E_\mathrm{rel}(\bm{x}, \bm{y})$ ", 
                              title="$(it[3])"))
        train = FoliationsManifoldsAutoencoders.ISFPadeLossInfinity(Misf, Xisf, xs./scale, ys./scale)
        @load "data/sys10dimTestRED-$(it[1]).bson" xs ys Tstep
        test = FoliationsManifoldsAutoencoders.ISFPadeLossInfinity(Misf, Xisf, xs./scale, ys./scale)
        println("Train ", (@sprintf "%.3e %.3e" train[1:2]...), " Test ", (@sprintf "%.3e %.3e" test[1:2]...))
    end
    plall = plot(pl...,margin=2mm, left_margin=10mm, bottom_margin=8mm, fontsize=10, tickfontsize=10, legend_font_pointsize=10, labelfontsize=10, titlefontsize=10)
    savefig(plall, "histograms-FULL.pdf")
    return pl
end

function accuracyPCA()
    println("PCA")
    pl = []
    for it in [(1,:red, "(b) -- PCA-1", 0) (2,:blue, "(e) -- PCA-2", 1) (3,:green, "(h) -- PCA-3", 2.1) (4,:black, "(k) -- PCA-4", 3.1)]
        @load "data/sys10dimTrainPCA-16-$(it[1]).bson" xs ys Tstep embedscales
        SysName = "10dim-PCA-16-$(it[1])"
        @load "ISFdata-$(SysName).bson" Misf Xisf scale Tstep

        MU, XU, MS, XS = ISFNormalForm(Misf, Xisf)
        @load "ManifoldImmersion-$(SysName).bson" Mimm Xres Tstep scale dataParIN dataParOUT
        MWt, XWt = ImmersionReconstruct(Mimm, Xres, Misf, Xisf, MU, XU)

        on_manifold = Eval(MWt, XWt, Eval(MU, XU, Eval(PadeU(Misf), PadeUpoint(Xisf), xs./scale))) .* scale
        amplitude = vec(transpose(vec(embedscales)) * on_manifold)
        
        hist = FoliationsManifoldsAutoencoders.ISFPadeLossHistogram(Misf, Xisf, xs./scale, ys./scale)
        p = sortperm(amplitude)
        xlims = exp.(collect(range(log(minimum(hist[2])), log(maximum(hist[2])), length=30)))
        ylims = [amplitude[p[Int(round(k))]] for k in range(1,length(amplitude), length=30)]
        push!(pl, histogram2d(hist[2], amplitude, 
                              bins=(xlims,ylims), c=cgrad(:YlGnBu_9),
                              xscale=:log10, 
                              colorbar_scale=:identity, 
                              yformatter = :scientific,
                              ylims = [0, 0.07],
                              yticks = [0.01, 0.03, 0.05, 0.07],
                              xlims = [1e-5, 1e-1],
                              xticks = ([1e-5, 1e-3, 1e-1], [L"10^{-5}", L"10^{-3}", L"10^{-1}"]),
                              ylabel = "amplitude", 
                              xlabel = L"$E_\mathrm{rel}(\bm{x}, \bm{y})$ ", 
                              title="$(it[3])"))
        train = FoliationsManifoldsAutoencoders.ISFPadeLossInfinity(Misf, Xisf, xs./scale, ys./scale)
        @load "data/sys10dimTestPCA-16-$(it[1]).bson" xs ys Tstep
        test = FoliationsManifoldsAutoencoders.ISFPadeLossInfinity(Misf, Xisf, xs./scale, ys./scale)
        println("Train ", (@sprintf "%.3e %.3e" train[1:2]...), " Test ", (@sprintf "%.3e %.3e" test[1:2]...))
    end
    plall = plot(pl...,margin=2mm, left_margin=10mm, bottom_margin=8mm, fontsize=10, tickfontsize=10, legend_font_pointsize=10, labelfontsize=10, titlefontsize=10)
    savefig(plall, "histograms-PCA.pdf")
    return pl
end

function accuracyDFT()
    println("DFT")
    pl = []
    for it in [(1,:red, "(c) -- DFT-1", 0) (2,:blue, "(f) -- DFT-2", 1) (3,:green, "(i) -- DFT-3", 2.1) (4,:black, "(l) -- DFT-4", 3.1)]
        @load "data/sys10dimTrainDFT-$(it[1]).bson" xs ys Tstep embedscales
        SysName = "10dim-DFT-$(it[1])"
        @load "ISFdata-$(SysName).bson" Misf Xisf scale Tstep

        MU, XU, MS, XS = ISFNormalForm(Misf, Xisf)
        @load "ManifoldImmersion-$(SysName).bson" Mimm Xres Tstep scale dataParIN dataParOUT
        MWt, XWt = ImmersionReconstruct(Mimm, Xres, Misf, Xisf, MU, XU)

        on_manifold = Eval(MWt, XWt, Eval(MU, XU, Eval(PadeU(Misf), PadeUpoint(Xisf), xs./scale))) .* scale
        amplitude = vec(transpose(vec(embedscales)) * on_manifold)
        
        hist = FoliationsManifoldsAutoencoders.ISFPadeLossHistogram(Misf, Xisf, xs./scale, ys./scale)
        p = sortperm(amplitude)
        xlims = exp.(collect(range(log(minimum(hist[2])), log(maximum(hist[2])), length=30)))
        ylims = [amplitude[p[Int(round(k))]] for k in range(1,length(amplitude), length=30)]
        push!(pl, histogram2d(hist[2], amplitude, 
                              bins=(xlims,ylims), c=cgrad(:YlGnBu_9),
                              xscale=:log10, 
                              colorbar_scale=:identity, 
                              yformatter = :scientific,
                              ylims = [0, 0.07],
                              yticks = [0.01, 0.03, 0.05, 0.07],
                              xlims = [1e-5, 1e-1],
                              xticks = ([1e-5, 1e-3, 1e-1], [L"10^{-5}", L"10^{-3}", L"10^{-1}"]),
                              ylabel = "amplitude", 
                              xlabel = L"$E_\mathrm{rel}(\bm{x}, \bm{y})$ ", 
                              title="$(it[3])"))
        train = FoliationsManifoldsAutoencoders.ISFPadeLossInfinity(Misf, Xisf, xs./scale, ys./scale)
        @load "data/sys10dimTestDFT-$(it[1]).bson" xs ys Tstep
        test = FoliationsManifoldsAutoencoders.ISFPadeLossInfinity(Misf, Xisf, xs./scale, ys./scale)
        println("Train ", (@sprintf "%.3e %.3e" train[1:2]...), " Test ", (@sprintf "%.3e %.3e" test[1:2]...))
    end
    plall = plot(pl...,margin=2mm, left_margin=10mm, bottom_margin=8mm, fontsize=10, tickfontsize=10, legend_font_pointsize=10, labelfontsize=10, titlefontsize=10)
    savefig(plall, "histograms-DFT.pdf")
    return pl
end
    
    pl1 = accuracyFULL()
    pl2 = accuracyPCA()
    pl3 = accuracyDFT()
    plall = plot(pl1[1], pl2[1], pl3[1], 
                 margin=2mm, left_margin=10mm, bottom_margin=8mm, 
                 fontsize=12, tickfontsize=12, legend_font_pointsize=12, labelfontsize=12, titlefontsize=12, 
                 layout = @layout([a{0.4w,0.8h} b{0.4w,0.8h} c{0.4w,0.8h} d{0.4w,0.8h}]))

    savefig(plall, "histograms-ALL.pdf")

