using FoliationsManifoldsAutoencoders
using BSON: @load, @save

function execute()
    # FULL STATE
    @load "data/AENC-on-manifold-bw.bson" xs ys Tstep
    SysName = "AENC-on-manifold"
#     @load "sysSynth10dim/sys10dimTrainRED-4.bson" xs ys Tstep
    embedscales = ones(1,10)/10
    dataINorig = xs
    dataOUTorig = ys

    freq = 1/2/pi
    bb = AENCIndentify(dataINorig, dataOUTorig, Tstep, embedscales, freq, (S=9,W=7); iteration = (aenc=2500, map=2500))

    @save "FigureData-$(SysName).bson" bb
    return bb
end

bb_data = execute()
