using FoliationsManifoldsAutoencoders
using Manopt
using BSON: @load, @save
using LinearAlgebra

function execute()
    # MODEL FULL STATE
    @load "data/sys10dimTrainRED-1.bson" xs ys Tstep
    embedscales = ones(1,10)/10
    SysName = "10dim-KOOPMAN-1"
    dataINorig = xs
    dataOUTorig = ys

    freq = 1/2/pi
    bb = FoliationIdentify(dataINorig, dataOUTorig, Tstep, embedscales, SysName, freq, orders = (P=1,Q=1,U=5,W=3); iterations = (f=8000,l=600))
    @save "FigureData-$(SysName).bson" bb
    return bb
end

bb_date = execute()
