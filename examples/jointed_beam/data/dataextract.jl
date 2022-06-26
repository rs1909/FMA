module dr
using FoliationsManifoldsAutoencoders
using LinearAlgebra
using Random # for randperm
using MAT
using BSON:@load,@save

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

for k=1:length(suffix)
    zs, TstepZS = dataRead(["nidata_20151015T165857_FULL1_3tq.mat"], [points[k]])

    xst, yst, Tstep, embedscales = PCAEmbed(zs, TstepZS, 12, freqs)
    xs, ys = dataPrune(xst, yst; nbox=20, curve=1.0, perbox=5000, retbox = 20, scale = 1.0, measure = nothing)
    @save "Beam-PCA-$(suffix[k]).bson" xs ys zs Tstep embedscales 

#     xst, yst, Tstep, embedscales = frequencyEmbed(zs, TstepZS, freqs; period = 2)
#     xs, ys = dataPrune(xst, yst; nbox=20, curve=1, perbox=12000, scale = 1.0, measure = nothing, cut = true)
#     @save "Beam-DFT-$(suffix[k]).bson" xs ys Tstep zs TstepZS embedscales
    
    scale = sqrt.(sum(xs.^2,dims=1))
    A = ((ys.*scale)*transpose(xs)) * inv((xs.*scale)*transpose(xs))

    println("frequencies [Hz] = ", sort(unique(abs.(angle.(eigvals(A))/(2*pi))))/Tstep)
    println("frequencies [rad/s] = ", sort(unique(abs.(angle.(eigvals(A)))))/Tstep)
end
end
