using FoliationsManifoldsAutoencoders
using Manopt
using BSON: @load, @save
using LinearAlgebra

function execute()
# BEAM OBSERVED STATE
    @load "data/Beam-PCA-3_1Nm.bson" xs ys Tstep embedscales
    SysName = "Beam-PCA-3_1Nm"
    dataINorig = xs
    dataOUTorig = ys

    NDIM = size(xs,1)    
    freq = 60.0
    bb = FoliationIdentify(dataINorig, dataOUTorig, Tstep, embedscales, SysName, freq; iterations = (f=8000, l=30), rank_deficiency = NDIM-10, node_rank = 4)
    @save "FigureData-$(SysName).bson" bb
    return bb
end


bb_date = execute()
