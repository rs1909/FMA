using FoliationsManifoldsAutoencoders
using LinearAlgebra
using BSON: @load, @save
using Printf
using LaTeXStrings
using Plots
using Plots.PlotMeasures

    pgfplotsx()
    push!(Plots.PGFPlotsX.CUSTOM_PREAMBLE,raw"\usepackage{amsmath,bm}")

function accuracy()
    println("DFT")
    pl = []
    for it in [("0_0Nm",:red, "(a)", 0) ("1_0Nm",:blue, "(b)", 1) ("2_1Nm",:green, "(c)", 2.1) ("3_1Nm",:black, "(d)", 3.1)]
        @load "data/Beam-DFT-$(it[1]).bson" xs ys Tstep embedscales
        SysName = "Beam-DFT-$(it[1])"
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
                              ylims = [0, 0.1],
                              yticks = [0.02, 0.04, 0.06, 0.08],
                              xlims = [1e-5, 1e-1],
                              xticks = ([1e-5, 1e-3, 1e-1], [L"10^{-5}", L"10^{-3}", L"10^{-1}"]),
                              ylabel = "amplitude", 
                              xlabel = L"$E_\mathrm{rel}(\bm{x}, \bm{y})$ ", 
                              title="$(it[3])    $(it[4]) Nm"))
        train = FoliationsManifoldsAutoencoders.ISFPadeLossInfinity(Misf, Xisf, xs./scale, ys./scale)
        println("Train ", (@sprintf "%.3e %.3e" train[1:2]...))
    end
        plall = plot(pl..., margin=2mm, left_margin=10mm, bottom_margin=8mm, fontsize=12, tickfontsize=12, legend_font_pointsize=12, labelfontsize=12, titlefontsize=12, layout = @layout([a{0.4w,0.8h} b{0.4w,0.8h} c{0.4w,0.8h} d{0.4w,0.8h}]))
    savefig(plall, "histograms-DFT.pdf")
end

accuracy()
