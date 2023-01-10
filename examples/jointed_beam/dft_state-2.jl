using FoliationsManifoldsAutoencoders
using Manopt
using BSON: @load, @save
using LinearAlgebra

function execute()
# BEAM OBSERVED STATE
    @load "data/Beam-DFT-1_0Nm.bson" xs ys Tstep embedscales
    SysName = "Beam-DFT-1_0Nm"
    dataINorig = xs
    dataOUTorig = ys

    freq = 60.0
    bb = FoliationIdentify(dataINorig, dataOUTorig, Tstep, embedscales, SysName, freq)
    @save "FigureData-$(SysName).bson" bb
    return bb
end


bb_date = execute()
