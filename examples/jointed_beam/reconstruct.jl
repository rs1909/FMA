module c
    using FoliationsManifoldsAutoencoders
    using BSON:@load
    using MAT
    using Plots
    using Plots.PlotMeasures
    using LaTeXStrings
    using PGFPlotsX

function dataRead(names, points)
    zs = []
    Tstep = [1.0]
    apt = zip(names, points)
    for it in apt
        nm, pt = it
        dd = matread(nm)
        for k in pt
            for p = 1:length(dd["ud"]["dblock"][k]["trig_x"])
                sz = length(dd["ud"]["dblock"][k]["trig_x"][p])
                Tstep .= (dd["ud"]["dblock"][k]["trig_x"][p][end] - dd["ud"]["dblock"][k]["trig_x"][p][1])/(sz-1)
                @show Tstep, k, p
                trigend = findlast( log10.(abs.(dd["ud"]["dblock"][k]["trig_y"][p][:,2])) .> -3.2) + 25
                satend = findlast( log10.(abs.(dd["ud"]["dblock"][k]["trig_y"][p][:,1])) .> -3.2)
                if satend == nothing
                    satend = sz
                elseif trigend > satend
                    trigend = findlast( log10.(abs.(dd["ud"]["dblock"][k]["trig_y"][p][1:satend,2])) .> -3.2) + 25
                end
                # filter at 2kHz, as the highest frequency is 810 Hz
    #             datfilt = lowpassfilter(dd["ud"]["dblock"][k]["trig_y"][1][:,2], 1/Tstep[1], 2000.0)
                datfilt = dd["ud"]["dblock"][k]["trig_y"][p][:,1]
                len = div(length(datfilt[trigend:end]),2)
                dat = datfilt[trigend:satend]
                @show size(dat)
                if length(dat) != 0
                    push!(zs, dat)
                else
                    @show sz, trigend, satend
                end
            end
    #         @show Tstep
        end
    end
    Tstep = Tstep[1]
    return zs, Tstep
end
    
names = ["Point1.mat", "Point2.mat", "Point3.mat", "Point4.mat", "Point5.mat"]
freqs = [60.5, 172, 323, 553, 810]

# zs, Tstep = dataRead(names)

suffix = ["0_0Nm" "1_0Nm" "2_1Nm" "3_1Nm"]
points = [[1,2],[3,4],[5,6],[7,8]]

    push!(PGFPlotsX.CUSTOM_PREAMBLE,raw"\usepackage{amsmath,bm}")
    pgfplotsx()
    pt = 1
    zs, TstepZS = dataRead(["data/nidata_20151015T165857_FULL1_3tq.mat"], [points[pt]])
    xst, yst, Tstep, embedscales, tab = PCAEmbed(zs, TstepZS, 12, freqs)

    @load "ISFdata-Beam-PCA-$(suffix[pt]).bson" Misf Xisf Tstep scale
    k=1
    start = 1
    xs = zeros(size(tab,1), length(zs[k])-length(tab)-start+1)
    for q = start:length(zs[k])-length(tab)
        xs[:,q-start+1] .= tab*zs[k][q:(q+size(tab,2)-1)]/scale
    end
    outs = Eval(PadeU(Misf), PadeUpoint(Xisf), [xs])
    out_re = zero(outs)
    out_re[:,1] .= outs[:,1]
    for q = 2:size(outs,2)
        out_re[:,q] .= Eval(PadeP(Misf), PadePpoint(Xisf), [out_re[:,q-1]])
    end
    tm = collect((0:size(outs,2)-1)*TstepZS)
    pl1 = plot([tm, tm], [dropdims(sqrt.(sum(outs.^2,dims=1)),dims=1), dropdims(sqrt.(sum(out_re.^2,dims=1)),dims=1), dropdims(sqrt.(sum(outs.^2,dims=1)),dims=1) - dropdims(sqrt.(sum(out_re.^2,dims=1)),dims=1)], 
              linestyle=[:solid :dash :dot :dashdot :dot :dot :dot :dot],linecolor=[:blue :darkgreen :red  :black :white :white :white :white], label = :none,
              xlabel="time [s]", ylabel=L"$\left\Vert \bm{z}\right\Vert$ ", title="(a)", xticks=[0,4,8,12])
    plen=size(outs,2)
    pl2 = plot([tm[1:plen], tm[1:plen], tm[1:plen]], [outs[1,1:plen], out_re[1,1:plen], outs[1,1:plen] - out_re[1,1:plen]], 
              linestyle=[:solid :dash :dot :dashdot :dot :dot :dot :dot],linecolor=[:blue :darkgreen :red :black :white :white :white :white],
              label=["Measurement" "Prediction" "Error"],
              xlabel="time [s]", ylabel=L"z_1", title="(b)", xticks=[0,4,8,12])
    pl = plot(pl1, pl2, layout = @layout([a{0.5w} b{0.5w}]),margin=5mm, left_margin=5mm, bottom_margin=5mm, fontsize=18, tickfontsize=18, legend_font_pointsize=18, labelfontsize=18, titlefontsize=18)
    savefig(pl, "reconstruct-beam-1.pdf")
    display(pl)

    pt = 3
    zs, TstepZS = dataRead(["data/nidata_20151015T165857_FULL1_3tq.mat"], [points[pt]])
    xst, yst, Tstep, embedscales, tab = PCAEmbed(zs, TstepZS, 12, freqs)
    @load "ISFdata-Beam-PCA-$(suffix[pt]).bson" Misf Xisf Tstep scale
    k=1
    start = 1
    xs = zeros(size(tab,1), length(zs[k])-length(tab)-start+1)
    for q = start:length(zs[k])-length(tab)
        xs[:,q-start+1] .= tab*zs[k][q:(q+size(tab,2)-1)]/scale
    end
    outs = Eval(PadeU(Misf), PadeUpoint(Xisf), [xs])
    out_re = zero(outs)
    out_re[:,1] .= outs[:,1]
    for q = 2:size(outs,2)
        out_re[:,q] .= Eval(PadeP(Misf), PadePpoint(Xisf), [out_re[:,q-1]])
    end
    tm = collect((0:size(outs,2)-1)*TstepZS)
    pl1 = plot([tm, tm], [dropdims(sqrt.(sum(outs.^2,dims=1)),dims=1), dropdims(sqrt.(sum(out_re.^2,dims=1)),dims=1), dropdims(sqrt.(sum(outs.^2,dims=1)),dims=1) - dropdims(sqrt.(sum(out_re.^2,dims=1)),dims=1)], 
              linestyle=[:solid :dash :dot :dashdot :dot :dot :dot :dot],linecolor=[:blue :darkgreen :red  :black :white :white :white :white], label = :none,
              xlabel="time [s]", ylabel=L"$\left\Vert \bm{z}\right\Vert $", title="(c)", xticks=[0,4,8,12])
    plen=size(outs,2)
    pl2 = plot([tm[1:plen], tm[1:plen], tm[1:plen]], [outs[1,1:plen], out_re[1,1:plen], outs[1,1:plen] - out_re[1,1:plen]], 
              linestyle=[:solid :dash :dot :dashdot :dot :dot :dot :dot],linecolor=[:blue :darkgreen :red :black :white :white :white :white],
              label=["Measurement" "Prediction" "Error"],
              xlabel="time [s]", ylabel=L"z_1", title="(d)", xticks=[0,4,8,12])
    pl = plot(pl1, pl2, layout = @layout([a{0.5w} b{0.5w}]),margin=5mm, left_margin=5mm, bottom_margin=5mm, fontsize=18, tickfontsize=18, legend_font_pointsize=18, labelfontsize=18, titlefontsize=18)
    savefig(pl, "reconstruct-beam-3.pdf")
    display(pl)
end
