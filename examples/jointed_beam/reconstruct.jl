# module c
    using FoliationsManifoldsAutoencoders
    using BSON:@load
    using MAT
    using Plots
    using Plots.PlotMeasures
    using LaTeXStrings
#     using PGFPlotsX
    using LinearAlgebra
    using Manifolds

names = ["Point1.mat", "Point2.mat", "Point3.mat", "Point4.mat", "Point5.mat"]
freqs = [60.5, 172, 323, 553, 810]

# zs, Tstep = dataRead(names)

suffix = ["0_0Nm" "1_0Nm" "2_1Nm" "3_1Nm"]
points = [[1,2],[3,4],[5,6],[7,8]]

    pgfplotsx()
    push!(Plots.PGFPlotsX.CUSTOM_PREAMBLE,raw"\usepackage{amsmath,bm}")
#     gr()
    for pt = 1:length(suffix)
        @load "data/Beam-DFT-$(suffix[pt]).bson" xs ys zs bandzs ampzs freqzs dampzs freqfitzs dampfitzs Tstep TstepZS embedscales
        xst, yst, Tstep, embedscales, ids = frequencyEmbed(zs, TstepZS, freqs)
        @load "ISFdata-Beam-DFT-$(suffix[pt]).bson" Misf Xisf Tstep scale
        MU, XU, MS, XS = ISFNormalForm(Misf, Xisf)
        for k=2:length(ids)
            xs = xst[:,ids[k-1]:ids[k]-1]./scale

            outs = Eval(MU, XU, Eval(PadeU(Misf), PadeUpoint(Xisf), xs))
            out_re = zero(outs)
            out_re[:,1] .= outs[:,1]
            for q = 2:size(outs,2)
                out_re[:,q] .= Eval(MS, XS, out_re[:,q-1])
            end
            tm = collect((0:size(outs,2)-1)*TstepZS)
            pl1 = plot([tm, tm], [dropdims(sqrt.(sum(outs.^2,dims=1)),dims=1), dropdims(sqrt.(sum(out_re.^2,dims=1)),dims=1), dropdims(sqrt.(sum(outs.^2,dims=1)),dims=1) - dropdims(sqrt.(sum(out_re.^2,dims=1)),dims=1)], 
                    linestyle=[:solid :solid :dot],
                    linecolor=[:blue :green :red], 
                    linewidth=[:auto 3 :auto],
                    label = :none,
                    xlabel="time [s]", ylabel=L"$\left\Vert \bm{z}\right\Vert$ ", title="(c)", xticks=[0,4,8,12])
            plen=size(outs,2)
            pl2 = plot([tm[1:plen], tm[1:plen], tm[1:plen]], [outs[1,1:plen], out_re[1,1:plen], outs[1,1:plen] - out_re[1,1:plen]], 
                    linestyle=[:solid :solid :dot],
                    linecolor=[:blue :green :red],
                    label=["Measurement" "Prediction" "Error"],
                    xlabel="time [s]", ylabel=L"z_1", title="(d)", xticks=[0,4,8,12])
            pl = plot(pl1, pl2, layout = @layout([a{0.5w} b{0.5w}]),margin=5mm, left_margin=5mm, bottom_margin=5mm, fontsize=18, tickfontsize=18, legend_font_pointsize=18, labelfontsize=18, titlefontsize=18)
            savefig(pl, "reconstruct-beam-$(suffix[pt])-$(k).pdf")
        end
    end
    
#     end
