import os
import h5py
import numpy as np
from skimage.measure import regionprops, label
import ndio.convert as convert

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_ndio"
dset_name = 'M3S1GNUds7'

in_file = os.path.join(datadir, dset_name + ".h5")
out_file = os.path.join(datadir, dset_name + "_converted.tiff")

convert.convert(in_file, out_file, in_fmt='hdf5', out_fmt='tiff')


load(hdf5_filename)
save(hdf5_filename, array)

load(nifti_filename)
save(nifti_filename, numpy_data)



ln -s /tmp/kleinnijenhuis15 .

chmod o+x /root /root/site /root/site/about
chmod -R +a "_www allow list,search,readattr" /root /root/site /root/site/about

ssh -Y ndcn0180@arcus-b.arc.ox.ac.uk

curl http://MyServer/MyPublic/TokenName/ChannelName/0000.tif
