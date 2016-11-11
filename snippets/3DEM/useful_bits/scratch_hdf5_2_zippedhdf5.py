### load hdf5 and zip in hdf5 ###
data, elsize = loadh5(datadir, dataset + '_probs0_eed2.h5')
writeh5(data, datadir, dataset + '_zip.h5', dtype='float', element_size_um=elsize)
