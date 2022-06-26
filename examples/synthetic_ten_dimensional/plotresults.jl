using FoliationsManifoldsAutoencoders
using LinearAlgebra
using StatsPlots
using Plots
using Plots.PlotMeasures
using PGFPlotsX
using KernelDensity
using BSON: @load, @save

include("data/sys10dimvectorfield.jl")
    function vfrhsPar(y)
        x = zero(y)
        vfrhs!(x, y, 0, 0)
        return x
    end

function executeVF()
#     MF = DensePolyManifold(10, 10, 5)
#     XF = fromFunction(MF, vfrhsPar)
    @load "sysSynth10dimCRT-model.bson" MF XF

    MWr, XWr, MRr, XRr = iManifoldVF(MF, XF, [9, 10], [])
    
    Dr = 0.0001
    r = range(0,1,step=Dr)
    opscal = ones(1,size(XWr,1))/size(XWr,1)
    return ODEManifoldFrequencyDamping(MWr, XWr, MRr, XRr, r; output=opscal)
end

function executeVF2()
#     MF = DensePolyManifold(10, 10, 5)
#     XF = fromFunction(MF, vfrhsPar)
    @load "sysSynth10dimCRT-model.bson" MF XF

    MWr, XWr, MRr, XRr = iManifoldVF(MF, XF, [7, 8], [])
    
    Dr = 0.0001
    r = range(0,1,step=Dr)
    opscal = ones(1,size(XWr,1))/size(XWr,1)
    return ODEManifoldFrequencyDamping(MWr, XWr, MRr, XRr, r; output=opscal)
end

bbvf = executeVF()
push!(PGFPlotsX.CUSTOM_PREAMBLE,raw"\usepackage{amsmath,bm,luatex85}")
pgfplotsx()

@load "FigureData-10dim-FULL-RED-1.bson" bb
id1 = findfirst(bb[3] .> 0.07)
bbr1 = bb
@load "FigureData-10dim-FULL-RED-2.bson" bb
id2 = findfirst(bb[3] .> 0.07)
bbr2 = bb
@load "FigureData-10dim-FULL-RED-3.bson" bb
id3 = findfirst(bb[3] .> 0.07)
bbr3 = bb
@load "FigureData-10dim-FULL-RED-4.bson" bb
id4 = findfirst(bb[3] .> 0.07)
bbr4 = bb
idvf = findfirst(bbvf[3] .> 0.07)

pl = plot(
    density([vec(bbr1[9]), vec(bbr2[9]), vec(bbr3[9]), vec(bbr4[9])],linestyle=[:solid :dash :dot :dashdot],linecolor=[:red :blue :darkgreen :purple :black],orientation=:h, ylims=[0,0.07],xlims=[0,40],leg=false,xlabel="density",ylabel="amplitude",title="(a)"),
     plot([bbr1[1][2:id1], bbr2[1][2:id2], bbr3[1][2:id3], bbr4[1][2:id4], bbvf[1][2:idvf]],
          [bbr1[3][2:id1], bbr2[3][2:id2], bbr3[3][2:id3], bbr4[3][2:id4], bbvf[3][2:idvf]], linestyle=[:solid :dash :dot :dashdot],linecolor=[:red :blue :darkgreen :purple :black],leg=false,xlims=[0.95,1.06],ylims=[0,0.07],xlabel="frequency [rad/s]", ylabel="amplitude", title="(b)", xticks = [ 0.95, 1.0, 1.05 ]),
     plot([bbr1[2][2:id1], bbr2[2][2:id2], bbr3[2][2:id3], bbr4[2][2:id4], bbvf[2][2:idvf]],
          [bbr1[3][2:id1], bbr2[3][2:id2], bbr3[3][2:id3], bbr4[3][2:id4], bbvf[3][2:idvf]], linestyle=[:solid :dash :dot :dashdot],linecolor=[:red :blue :darkgreen :purple :black],label=["ST-1" "ST-2" "ST-3" "ST-4" "VF"], legend_position=:bottomright, xlims=[0.001,0.1],ylims=[0,0.07],xlabel="damping ratio [-]", ylabel="amplitude", xscale=:log10, title="(c)"),
     layout = @layout([a{0.3w} b{0.35w} c{0.35w}]), size=(900,div(900,3)),margin=5mm, left_margin=5mm, bottom_margin=5mm, fontsize=14, tickfontsize=14, legend_font_pointsize=14, labelfontsize=14, titlefontsize=14)
savefig(pl, "FullState.pdf")


@load "FigureData-10dim-PCA-200-1.bson" bb
id1 = findfirst(bb[3] .> 0.07)
bbr1 = bb
@load "FigureData-10dim-PCA-200-2.bson" bb
id2 = findfirst(bb[3] .> 0.07)
bbr2 = bb
@load "FigureData-10dim-PCA-200-3.bson" bb
id3 = findfirst(bb[3] .> 0.07)
bbr3 = bb
@load "FigureData-10dim-PCA-200-4.bson" bb
id4 = findfirst(bb[3] .> 0.07)
bbr4 = bb
idvf = findfirst(bbvf[3] .> 0.07)

pl = plot(
    density([vec(bbr1[9]), vec(bbr2[9]), vec(bbr3[9]), vec(bbr4[9])],linestyle=[:solid :dash :dot :dashdot],linecolor=[:red :blue :darkgreen :purple :black],orientation=:h, ylims=[0,0.07],xlims=[0,40],leg=false,xlabel="density",ylabel="amplitude",title="(d)"),
     plot([bbr1[1][2:id1], bbr2[1][2:id2], bbr3[1][2:id3], bbr4[1][2:id4], bbvf[1][2:idvf]],
          [bbr1[3][2:id1], bbr2[3][2:id2], bbr3[3][2:id3], bbr4[3][2:id4], bbvf[3][2:idvf]], linestyle=[:solid :dash :dot :dashdot],linecolor=[:red :blue :darkgreen :purple :black],leg=false,xlims=[0.95,1.06],ylims=[0,0.07],xlabel="frequency [rad/s]", ylabel="amplitude", title="(e)", xticks = [ 0.95, 1.0, 1.05 ]),
     plot([bbr1[2][2:id1], bbr2[2][2:id2], bbr3[2][2:id3], bbr4[2][2:id4], bbvf[2][2:idvf]],
          [bbr1[3][2:id1], bbr2[3][2:id2], bbr3[3][2:id3], bbr4[3][2:id4], bbvf[3][2:idvf]], linestyle=[:solid :dash :dot :dashdot],linecolor=[:red :blue :darkgreen :purple :black],label=["PCA-1" "PCA-2" "PCA-3" "PCA-4" "VF"], legend_position=:bottomright, xlims=[0.001,0.1],ylims=[0,0.07],xlabel="damping ratio [-]", ylabel="amplitude", xscale=:log10, title="(f)"),
     layout = @layout([a{0.3w} b{0.35w} c{0.35w}]), size=(900,div(900,3)),margin=5mm, left_margin=5mm, bottom_margin=5mm, fontsize=14, tickfontsize=14, legend_font_pointsize=14, labelfontsize=14, titlefontsize=14)
savefig(pl, "PCAState.pdf")

    
@load "FigureData-10dim-DFT-1.bson" bb
id1 = findfirst(bb[3] .> 0.07)
bbr1 = bb
@load "FigureData-10dim-DFT-2.bson" bb
id2 = findfirst(bb[3] .> 0.07)
bbr2 = bb
@load "FigureData-10dim-DFT-3.bson" bb
id3 = findfirst(bb[3] .> 0.07)
bbr3 = bb
@load "FigureData-10dim-DFT-4.bson" bb
id4 = findfirst(bb[3] .> 0.07)
bbr4 = bb
idvf = findfirst(bbvf[3] .> 0.07)

pl = plot(
    density([vec(bbr1[9]), vec(bbr2[9]), vec(bbr3[9]), vec(bbr4[9])],linestyle=[:solid :dash :dot :dashdot],linecolor=[:red :blue :darkgreen :purple :black],orientation=:h, ylims=[0,0.07],xlims=[0,40],leg=false,xlabel="density",ylabel="amplitude",title="(g)"),
     plot([bbr1[1][2:id1], bbr2[1][2:id2], bbr3[1][2:id3], bbr4[1][2:id4], bbvf[1][2:idvf]],
          [bbr1[3][2:id1], bbr2[3][2:id2], bbr3[3][2:id3], bbr4[3][2:id4], bbvf[3][2:idvf]], linestyle=[:solid :dash :dot :dashdot],linecolor=[:red :blue :darkgreen :purple :black],leg=false,xlims=[0.95,1.06],ylims=[0,0.07],xlabel="frequency [rad/s]", ylabel="amplitude", title="(h)", xticks = [ 0.95, 1.0, 1.05 ]),
     plot([bbr1[2][2:id1], bbr2[2][2:id2], bbr3[2][2:id3], bbr4[2][2:id4], bbvf[2][2:idvf]],
          [bbr1[3][2:id1], bbr2[3][2:id2], bbr3[3][2:id3], bbr4[3][2:id4], bbvf[3][2:idvf]], linestyle=[:solid :dash :dot :dashdot],linecolor=[:red :blue :darkgreen :purple :black],label=["DFT-1" "DFT-2" "DFT-3" "DFT-4" "VF"], legend_position=:bottomright, xlims=[0.001,0.1],ylims=[0,0.07],xlabel="damping ratio [-]", ylabel="amplitude", xscale=:log10, title="(i)"),
     layout = @layout([a{0.3w} b{0.35w} c{0.35w}]), size=(900,div(900,3)),margin=5mm, left_margin=5mm, bottom_margin=5mm, fontsize=14, tickfontsize=14, legend_font_pointsize=14, labelfontsize=14, titlefontsize=14)
savefig(pl, "DFTState.pdf")


@load "FigureData-10dim-KOOPMAN-1.bson" bb
id1 = findfirst(bb[3] .> 0.07)
bbr1 = bb
@load "FigureData-10dim-KOOPMAN-2.bson" bb
id2 = findfirst(bb[3] .> 0.07)
bbr2 = bb
@load "FigureData-10dim-KOOPMAN-3.bson" bb
id3 = findfirst(bb[3] .> 0.07)
bbr3 = bb
@load "FigureData-10dim-KOOPMAN-4.bson" bb
id4 = findfirst(bb[3] .> 0.07)
bbr4 = bb
idvf = findfirst(bbvf[3] .> 0.07)

pl = plot(
     plot([bbr1[1][2:id1], bbr2[1][2:id2], bbr3[1][2:id3], bbr4[1][2:id4], bbvf[1][2:idvf]],
          [bbr1[3][2:id1], bbr2[3][2:id2], bbr3[3][2:id3], bbr4[3][2:id4], bbvf[3][2:idvf]], linestyle=[:solid :dash :dot :dashdot],linecolor=[:red :blue :darkgreen :purple :black],leg=false,xlims=[0.95,1.06],ylims=[0,0.07],xlabel="frequency [rad/s]", ylabel="amplitude", title="(a)"),
     plot([bbr1[2][2:id1], bbr2[2][2:id2], bbr3[2][2:id3], bbr4[2][2:id4], bbvf[2][2:idvf]],
          [bbr1[3][2:id1], bbr2[3][2:id2], bbr3[3][2:id3], bbr4[3][2:id4], bbvf[3][2:idvf]], linestyle=[:solid :dash :dot :dashdot],linecolor=[:red :blue :darkgreen :purple :black],label=["ST-1" "ST-2" "ST-3" "ST-4" "VF"], legend_position=:bottomright, xlims=[0.001,0.1],ylims=[0,0.07],xlabel="damping ratio [-]", ylabel="amplitude", xscale=:log10, title="(b)"),
     layout = @layout([b{0.5w} c{0.5w}]), size=(900,div(900,3)),margin=5mm, left_margin=5mm, bottom_margin=5mm, fontsize=14, tickfontsize=14, legend_font_pointsize=14, labelfontsize=14, titlefontsize=14)
savefig(pl, "Koopman.pdf")


bbvf2 = executeVF2()

@load "FigureData-AENC-on-manifold.bson" bb
id1 = findfirst(bb[3] .> 0.07)
bbr1 = bb
@load "FigureData-AENC-densedata.bson" bb
id2 = findfirst(bb[3] .> 0.07)
bbr2 = bb
# @load "FigureData-10dim-KOOPMAN-3.bson" bb
# id3 = findfirst(bb[3] .> 0.07)
# bbr3 = bb
# @load "FigureData-10dim-KOOPMAN-4.bson" bb
# id4 = findfirst(bb[3] .> 0.07)
# bbr4 = bb
idvf = findfirst(bbvf[3] .> 0.07)

pl = plot(
     plot([bbr1[1][2:id1], bbr2[1][2:id2], bbvf[1][2:idvf], bbvf2[1][2:idvf]],
          [bbr1[3][2:id1], bbr2[3][2:id2], bbvf[3][2:idvf], bbvf2[3][2:idvf]], linestyle=[:solid :dash :dot :dashdot],linecolor=[:red :blue :black :darkgreen],xlims=[0.95,3.2],ylims=[0,0.07],xlabel="frequency [rad/s]", ylabel="amplitude", title="(a)",label=["MAN" "ST-1" "VF-1" "VF-2"], legend_position=:topright),
     plot([bbr1[2][2:id1], bbr2[2][2:id2], bbvf[2][2:idvf], bbvf2[2][2:idvf]],
          [bbr1[3][2:id1], bbr2[3][2:id2], bbvf[3][2:idvf], bbvf2[3][2:idvf]], linestyle=[:solid :dash :dot :dashdot],linecolor=[:red :blue :black :darkgreen], xlims=[0.001,10.0],ylims=[0,0.07],xlabel="damping ratio [-]", ylabel="amplitude", xscale=:log10, title="(b)",leg=false),
     layout = @layout([b{0.7w} c{0.3w}]), size=(900,div(900,3)),margin=5mm, left_margin=5mm, bottom_margin=5mm, fontsize=14, tickfontsize=14, legend_font_pointsize=14, labelfontsize=14, titlefontsize=14)
savefig(pl, "AENC-figure.pdf")
