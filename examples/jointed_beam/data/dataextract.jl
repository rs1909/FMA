module dr
using FoliationsManifoldsAutoencoders
using LinearAlgebra
using Random # for randperm
using MAT
using BSON:@load,@save
using DSP

function highpassfilter(signals, fs, cutoff)
    wdo = 2.0 * cutoff / fs
    filth = digitalfilter(Highpass(wdo), FIRWindow(hamming(511)))
    filtfilt(filth, signals)
end

function bandpassfilter(signals, fs, low, high)
    filth = digitalfilter(Bandpass(low,high;fs=fs), FIRWindow(hamming(511)))
    filtfilt(filth, signals)
end

function dataRead(names, points)
    zs = []
    bandzs = []
    freqzs = []
    dampzs = []
    ampzs = []
    freqfitzs = []
    dampfitzs = []
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
                trig_sig = dd["ud"]["dblock"][k]["trig_y"][p][:,2]
                acc_sig = dd["ud"]["dblock"][k]["trig_y"][p][:,1]
                # high-pass at 30Hz
                trig_sig_filt = highpassfilter(trig_sig, 1/Tstep[1], 30.0)
                acc_sig_filt = highpassfilter(acc_sig, 1/Tstep[1], 30.0)
                acc_sig_band = bandpassfilter(acc_sig, 1/Tstep[1], 30.0, 75.0)
                trigend = findlast( log10.(abs.(trig_sig_filt)) .> -3.2) + 25
                satend = findlast( log10.(abs.(acc_sig_filt)) .> -3.2)
                if satend == nothing
                    satend = sz
                elseif trigend > satend
                    trigend = findlast( log10.(abs.(trig_sig_filt[1:satend,2])) .> -3.2) + Int(round(3/60/Tstep[1]))
                end
                # filter at 20Hz
                dat = acc_sig_filt[trigend:satend]
                sig = acc_sig_band[trigend:satend]
                @show size(dat)
                if length(dat) != 0
                    push!(zs, dat)
                    push!(bandzs, sig)
                else
                    @show sz, trigend, satend
                end
                # zero crossings
                cross = findall(sig[2:end].*sig[1:end-1] .< 0)
                lam = sig[cross] ./ (sig[cross] .- sig[cross.+1])
                tm = range(0.0, step = Tstep[1], length=length(sig))
                tcross = tm[cross] .+ lam*Tstep[1]
                settle = 10
                freqs = 2.0./(tcross[settle+4:end] - tcross[settle:end-4])
                amps = [sqrt(sum(sig[cross[k]:cross[k+4]] .^ 2)/(cross[k+4]-cross[k]+1)) for k in settle:length(cross)-4]
                damps = log.(amps[1:end-4]./amps[5:end]) ./ (4*pi)
                damps = [damps; damps[end-3:end]]
                push!(freqzs, freqs)
                push!(ampzs, amps)
                push!(dampzs, damps)
                # frequency fit
                M = FoliationsManifoldsAutoencoders.DensePolyManifold(1, 1, 8)
                X = FoliationsManifoldsAutoencoders.fromData(M, amps, reshape(freqs,1,:))
                freqfit = vec(Eval(M, X, [amps]))
                push!(freqfitzs, freqfit)
                # damping fit
                X = FoliationsManifoldsAutoencoders.fromData(M, amps, reshape(damps,1,:))
                dampfit = vec(Eval(M, X, [amps]))
                push!(dampfitzs, dampfit)            
            end
        end
    end
    Tstep = Tstep[1]
    return zs, bandzs, ampzs, freqzs, dampzs, freqfitzs, dampfitzs, Tstep
end
    
names = ["Point1.mat", "Point2.mat", "Point3.mat", "Point4.mat", "Point5.mat"]
freqs = [59.99035761521204, 172.20587224626271, 322.2764962633326, 553.0038339511226, 807.5810318342275]
# [60.5, 172, 323, 553, 810]

# zs, Tstep = dataRead(names)

suffix = ["0_0Nm" "1_0Nm" "2_1Nm" "3_1Nm"]
points = [[1,2],[3,4],[5,6],[7,8]]

for k=1:length(suffix)
    zs, bandzs, ampzs, freqzs, dampzs, freqfitzs, dampfitzs, TstepZS = dataRead(["nidata_20151015T165857_FULL1_3tq.mat"], [points[k]])

    xst, yst, Tstep, embedscales = PCAEmbed(zs, TstepZS, 18, freqs)
    xs, ys = dataPrune(xst, yst; nbox=20, curve=1.0, perbox=5000, retbox = 20, scale = 1.0, measure = nothing)
    @save "Beam-PCA-$(suffix[k]).bson" xs ys zs bandzs ampzs freqzs dampzs freqfitzs dampfitzs Tstep TstepZS embedscales 

    xst, yst, Tstep, embedscales = frequencyEmbed(zs, TstepZS, freqs)
    xs, ys = dataPrune(xst, yst; nbox=20, curve=1, perbox=2000, scale = 1.0, measure = nothing, cut = false)
    @save "Beam-DFT-$(suffix[k]).bson" xs ys zs bandzs ampzs freqzs dampzs freqfitzs dampfitzs Tstep TstepZS embedscales
    
    scale = sqrt.(sum(xs.^2,dims=1))
    A = ((ys.*scale)*transpose(xs)) * inv((xs.*scale)*transpose(xs))

    println("frequencies [Hz] = ", sort(unique(abs.(angle.(eigvals(A))/(2*pi))))/Tstep)
    println("frequencies [rad/s] = ", sort(unique(abs.(angle.(eigvals(A)))))/Tstep)
end
end
