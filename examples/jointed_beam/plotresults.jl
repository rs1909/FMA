# FOR BEAM DATA
using StatsPlots
using Plots
using Plots.PlotMeasures 
using KernelDensity
using BSON:@load
# using PGFPlotsX
using BSON

tab1 = [60.11398124438379; 38.80743895195061;;
60.14551986703511; 29.150868399821498;;
60.17705848968644; 25.70209320263251;;
60.26378970197758; 17.70093474515412;;
60.299428345573574; 15.962752045770877;;
60.37417488125722; 5.383059332527523;;
60.44188007662284; 1.8365688380848724;;
60.31625441696113; 12.179289026275114;;
60.32332155477032; 12.61205564142194]
tab2 = [60.349417637271216; 51.33720930232559;;
60.36772046589019; 31.56976744186047;;
60.44093178036606; 22.906976744186053;;
60.38103161397671; 20.34883720930233;;
60.445923460898506; 15.465116279069768;;
60.49584026622296; 9.883720930232563;;
60.49417637271215; 9.069767441860478;;
60.51247920133112; 5.348837209302339;;
60.52745424292846; 3.3139534883721]
tab3 = [60.69031263805005; 28.421529400656233;;
60.70408645010302; 35.71017161201992;;
60.706054137539155; 21.385407076930342;;
60.71786026215599; 26.76867195429974;;
60.74737557369807; 10.722508860494834;;
60.76705244805946; 7.887071265019003;;
60.735449735449734; 13.70370370370371;;
60.74779541446208; 13.086419753086425;;
60.673721340388006; 41.85185185185186;;
60.68959435626102; 41.79012345679013]
tab4 = [60.83173017684788; 32.52576735757164;;
60.832844848583264; 29.221316117340585;;
60.84998956146557; 28.11196462954873;;
60.86693944806513; 12.411172118740929;;
60.86758545100268; 16.00351115641636;;
60.87305972774756; 11.562083915379873;;
60.87706016075343; 7.852531430513082;;
60.89600958025493; 3.8321157548986307;;
60.838938053097344; 35.74074074074075;;
60.849557522123895; 35.4320987654321]
dam1 = [0.001949078514812328; 1.9024651455283177;;
0.0019867710403018903; 16.000843096561773;;
0.0019983758214319395; 17.65389825033855;;
0.002028885165370617; 5.427835748090594;;
0.002044794945952136; 25.698766665385563;;
0.0021318308044275046; 38.81300421868136;;
0.002307399912491796; 29.161859373956595;;
0.0018896103896103899; 12.702980472764644;;
0.0018993506493506494; 12.209660842754367]
dam2 = [0.0018287461773700301; 51.55162441381396;;
0.0018929663608562687; 31.768746829923735;;
0.0017308868501529047; 22.971729593607343;;
0.0018899082568807337; 20.47855916140915;;
0.0017828746177370024; 15.583872431682074;;
0.0018501529051987765; 10.000978830564428;;
0.0018073394495412842; 9.18374429386273;;
0.001935779816513761; 5.46667971774086;;
0.0019785932721712537; 3.490509792754871]
dam3 = [0.0015131088146670886; 28.385968938340213;;
0.0015566744541624146; 13.706398768861526;;
0.0015682981048552765; 35.7390979719986;;
0.0015915454062410009; 10.671336566586213;;
0.0016031690569338634; 7.853573418699156;;
0.0016395398349083026; 21.319555672813948;;
0.001661357998913757; 12.998089112673625;;
0.0016962289509923443; 26.729508933577293;;
0.001505691056910569; 41.83316168898043;;
0.0015447154471544713; 41.771369721936146]
dam4 =[0.0010888455643776177; 7.807847764416437;;
0.0011005399035860634; 42.69622999200293;;
0.0011237165164562045; 32.471903616755455;;
0.0011339110297687968; 28.055826358849878;;
0.0011571583311545213; 29.16013302869183;;
0.0011735208548221657; 15.943510345186517;;
0.0010764227642276422; 35.34500514933059;;
0.0011121951219512194; 35.71575695159629;;
0.0011252032520325203; 12.296601441812555;;
0.001138211382113821; 11.493305870236867]

plFreq = plot(xlims=[60.0,61.0],ylims=[0,0.1])
for it in [("0_0Nm",:red) ("1_0Nm",:blue) ("2_1Nm",:green) ("3_1Nm",:black)]
    name = it[1]
    color = it[2]
    dt = BSON.load("data/Beam-PCA-$(name).bson")
    for k=1:length(dt[:freqzs])
        plot!(dt[:freqfitzs][k], dt[:ampzs][k],label=nothing, linecolor=color,linewidth=3,linealpha=0.1)
    end
end

plDamp = plot(xlims=[-0.001,0.0025],ylims=[0,0.1])
for it in [("0_0Nm",:red) ("1_0Nm",:blue) ("2_1Nm",:green) ("3_1Nm",:black)]
    name = it[1]
    color = it[2]
    dt = BSON.load("data/Beam-PCA-$(name).bson")
    for k=1:length(dt[:freqzs])
        plot!(dt[:dampfitzs][k], dt[:ampzs][k],label=nothing, linecolor=color,linewidth=3,linealpha=0.1)
    end
end

@load "FigureData-Beam-PCA-CAS4-tst-0_0Nm.bson" bb
id1 = findfirst(bb[3] .> 0.1)
bb1 = bb
@load "FigureData-Beam-PCA-CAS4-tst-1_0Nm.bson" bb
id2 = findfirst(bb[3] .> 0.1)
bb2 = bb
@load "FigureData-Beam-PCA-CAS4-tst-2_1Nm.bson" bb
id3 = findfirst(bb[3] .> 0.1)
bb3 = bb
@load "FigureData-Beam-PCA-CAS4-tst-3_1Nm.bson" bb
id4 = findfirst(bb[3] .> 0.1)
bb4 = bb

    pgfplotsx()
    push!(Plots.PGFPlotsX.CUSTOM_PREAMBLE,raw"\usepackage{amsmath,bm}")

ddr = range(-0.15,0.15,length=100)
d1 = pdf(kde(vec(bb1[9]),bandwidth=0.003), ddr)
d2 = pdf(kde(vec(bb2[9]),bandwidth=0.003), ddr)
d3 = pdf(kde(vec(bb3[9]),bandwidth=0.003), ddr)
d4 = pdf(kde(vec(bb4[9]),bandwidth=0.003), ddr)
# the scaling factor between forces and accelerations
sfac =  442.75

pl1 = plot([d1+d1[end:-1:1] d2+d2[end:-1:1] d3+d3[end:-1:1] d4+d4[end:-1:1]], [ddr ddr ddr ddr], 
           linestyle=[:solid :dash :dot :dashdot], linecolor=[:red :blue :darkgreen :black], ylims=[0,0.1],xlims=[0,40],leg=false,xlabel="density",ylabel="amplitude",title="(a)")
pl2 = plot(plFreq, [bb1[1][2:id1]/2/pi, bb2[1][2:id2]/2/pi, bb3[1][2:id3]/2/pi, bb4[1][2:id4]/2/pi, vec(tab1[1,:]), vec(tab2[1,:]), vec(tab3[1,:]), vec(tab4[1,:])],
           [bb1[3][2:id1], bb2[3][2:id2], bb3[3][2:id3], bb4[3][2:id4], vec(tab1[2,:])/sfac, vec(tab2[2,:])/sfac, vec(tab3[2,:])/sfac, vec(tab4[2,:])/sfac], 
           xticks = [ 60.1, 60.4, 60.7, 61 ], linestyle=[:solid :dash :dot :dashdot :dot :dot :dot :dot], linecolor=[:red :blue :darkgreen :black :white :white :white :white], markershape =[:none :none :none :none :circle :cross :rect :dtriangle], markercolor=[:red :blue :darkgreen :black :red :blue :darkgreen :black], markerstrokecolor=:match, markersize=2, leg=false,xlims=[60.0,61.0],ylims=[0,0.1],xlabel="frequency [Hz]", yticks=[], title="(b)")
pl3 = plot(plDamp, [bb1[2][2:id1], bb2[2][2:id2], bb3[2][2:id3], bb4[2][2:id4], vec(dam1[1,:]), vec(dam2[1,:]), vec(dam3[1,:]), vec(dam4[1,:])],
           [bb1[3][2:id1], bb2[3][2:id2], bb3[3][2:id3], bb4[3][2:id4], vec(dam1[2,:])/sfac, vec(dam2[2,:])/sfac, vec(dam3[2,:])/sfac, vec(dam4[2,:])/sfac], xticks = [ 0, 0.001, 0.002 ],
            linestyle=[:solid :dash :dot :dashdot :dot :dot :dot :dot],linecolor=[:red :blue :darkgreen :black :white :white :white :white], markershape =[:none :none :none :none :circle :cross :rect :dtriangle], markercolor=[:red :blue :darkgreen :black :red :blue :darkgreen :black], markerstrokecolor=:match, markersize=2,
            label=["0 Nm" "1.0 Nm" "2.1 Nm" "3.1 Nm"], legend_position=:topleft,xlims=[-0.0005,0.0025],
            ylims=[0,0.1],xlabel="damping ratio [-]", xscale=:identity, yticks=[], title="(c)")
#          linestyle=[:solid :dash :dot :dashdot :dot :dot :dot :dot],linecolor=[:red :blue :darkgreen :black :white :white :white :white], markershape =[:none :none :none :none :circle :cross :rect :dtriangle], markercolor=[:red :blue :darkgreen :black :red :blue :darkgreen :black], markerstrokecolor=:match, markersize=2,
#          leg=false,xlims=[60.0,61.0],ylims=[0,0.1],xlabel="frequency [Hz]", ylabel="amplitude", title="(b)")
pl = plot(pl1, pl2, pl3, layout = @layout([a{0.25w} b{0.35w} c{0.4w}]), size=(900,div(900,3)),margin=5mm, left_margin=5mm, bottom_margin=5mm, fontsize=14, tickfontsize=14, legend_font_pointsize=14, labelfontsize=14, titlefontsize=14)
savefig(pl, "JointedBeam-PCA-CAS4-tst.pdf")

@load "FigureData-Beam-DFT-CAS4-tst-0_0Nm.bson" bb
id1 = findfirst(bb[3] .> 0.1)
bb1 = bb
@load "FigureData-Beam-DFT-CAS4-tst-1_0Nm.bson" bb
id2 = findfirst(bb[3] .> 0.1)
bb2 = bb
@load "FigureData-Beam-DFT-CAS4-tst-2_1Nm.bson" bb
id3 = findfirst(bb[3] .> 0.1)
bb3 = bb
@load "FigureData-Beam-DFT-CAS4-tst-3_1Nm.bson" bb
id4 = findfirst(bb[3] .> 0.1)
bb4 = bb

#push!(PGFPlotsX.CUSTOM_PREAMBLE,raw"\usepackage{amsmath,bm,luatex85}")
#pgfplotsx()
ddr = range(-0.15,0.15,length=100)
d1 = pdf(kde(vec(bb1[9]),bandwidth=0.003), ddr)
d2 = pdf(kde(vec(bb2[9]),bandwidth=0.003), ddr)
d3 = pdf(kde(vec(bb3[9]),bandwidth=0.003), ddr)
d4 = pdf(kde(vec(bb4[9]),bandwidth=0.003), ddr)
# the scaling factor between forces and accelerations
sfac =  442.75

pl1 = plot([d1+d1[end:-1:1] d2+d2[end:-1:1] d3+d3[end:-1:1] d4+d4[end:-1:1]], [ddr ddr ddr ddr], 
           linestyle=[:solid :dash :dot :dashdot], linecolor=[:red :blue :darkgreen :black], ylims=[0,0.1],xlims=[0,40],leg=false,xlabel="density",ylabel="amplitude",title="(a)")
pl2 = plot(plFreq, [bb1[1][2:id1]/2/pi, bb2[1][2:id2]/2/pi, bb3[1][2:id3]/2/pi, bb4[1][2:id4]/2/pi, vec(tab1[1,:]), vec(tab2[1,:]), vec(tab3[1,:]), vec(tab4[1,:])],
           [bb1[3][2:id1], bb2[3][2:id2], bb3[3][2:id3], bb4[3][2:id4], vec(tab1[2,:])/sfac, vec(tab2[2,:])/sfac, vec(tab3[2,:])/sfac, vec(tab4[2,:])/sfac], 
           xticks = [ 60.1, 60.4, 60.7, 61 ], linestyle=[:solid :dash :dot :dashdot :dot :dot :dot :dot], linecolor=[:red :blue :darkgreen :black :white :white :white :white], markershape =[:none :none :none :none :circle :cross :rect :dtriangle], markercolor=[:red :blue :darkgreen :black :red :blue :darkgreen :black], markerstrokecolor=:match, markersize=2, leg=false,xlims=[60.0,61.0],ylims=[0,0.1],xlabel="frequency [Hz]", yticks=[], title="(b)")
pl3 = plot(plDamp, [bb1[2][2:id1], bb2[2][2:id2], bb3[2][2:id3], bb4[2][2:id4], vec(dam1[1,:]), vec(dam2[1,:]), vec(dam3[1,:]), vec(dam4[1,:])],
           [bb1[3][2:id1], bb2[3][2:id2], bb3[3][2:id3], bb4[3][2:id4], vec(dam1[2,:])/sfac, vec(dam2[2,:])/sfac, vec(dam3[2,:])/sfac, vec(dam4[2,:])/sfac], xticks = [ 0, 0.001, 0.002 ],
            linestyle=[:solid :dash :dot :dashdot :dot :dot :dot :dot],linecolor=[:red :blue :darkgreen :black :white :white :white :white], markershape =[:none :none :none :none :circle :cross :rect :dtriangle], markercolor=[:red :blue :darkgreen :black :red :blue :darkgreen :black], markerstrokecolor=:match, markersize=2,
            label=["0 Nm" "1.0 Nm" "2.1 Nm" "3.1 Nm"], legend_position=:topleft,xlims=[-0.0005,0.0025],
            ylims=[0,0.1],xlabel="damping ratio [-]", xscale=:identity, yticks=[], title="(c)")
#          linestyle=[:solid :dash :dot :dashdot :dot :dot :dot :dot],linecolor=[:red :blue :darkgreen :black :white :white :white :white], markershape =[:none :none :none :none :circle :cross :rect :dtriangle], markercolor=[:red :blue :darkgreen :black :red :blue :darkgreen :black], markerstrokecolor=:match, markersize=2,
#          leg=false,xlims=[60.0,61.0],ylims=[0,0.1],xlabel="frequency [Hz]", ylabel="amplitude", title="(b)")
pl = plot(pl1, pl2, pl3, layout = @layout([a{0.25w} b{0.35w} c{0.4w}]), size=(900,div(900,3)),margin=5mm, left_margin=5mm, bottom_margin=5mm, fontsize=14, tickfontsize=14, legend_font_pointsize=14, labelfontsize=14, titlefontsize=14)
savefig(pl, "JointedBeam-DFT-CAS4-tst.pdf")
