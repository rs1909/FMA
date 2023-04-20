using FoliationsManifoldsAutoencoders
using Manopt
using BSON: @load, @save
using LinearAlgebra

function execute()
# MODEL OBSERVED STATE
    @load "data/sys10dimTrainDFT-4.bson" xs ys Tstep embedscales
    SysName = "10dim-DFT-4"
    dataINorig = xs
    dataOUTorig = ys

    freq = 1/2/pi
    bb = FoliationIdentify(dataINorig, dataOUTorig, Tstep, embedscales, SysName, freq; iterations = (f=8000,l=600))
    @save "FigureData-$(SysName).bson" bb
    return bb
end

bb_date = execute()
