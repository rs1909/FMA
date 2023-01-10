using FoliationsManifoldsAutoencoders
using Manopt
using BSON: @load, @save
using LinearAlgebra

function execute()
    # MODEL FULL STATE
    @load "data/sys10dimTrainRED-1.bson" xs ys Tstep
    embedscales = ones(1,10)/10
    SysName = "10dim-FULL-1"
    dataINorig = xs
    dataOUTorig = ys

    freq = 1/2/pi
    bb = FoliationIdentify(dataINorig, dataOUTorig, Tstep, embedscales, SysName, freq; iterations = (f=8000, l=60), node_rank = 4)
    @save "FigureData-$(SysName).bson" bb
    return bb
end

bb_date = execute()
