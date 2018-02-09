import os
import h5py
import numpy as np
from skimage.measure import regionprops, label
import glob

datadir = '/Users/michielk/oxdox/papers/abstracts/ISMRM2017/Eposter/anims/M3S1GNUvols/data'
dset_name = 'M3S1GNUds7'

labelvolume = ['_maskMM_final', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
MM = f[labelvolume[1]][:, :, :].astype('bool')
f.close()

labelvolume = ['_labelALL_shuffled', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
LA = f[labelvolume[1]][:, :, :]
f.close()

LA[MM] = 0

gname = dset_name + '_labelALL_final_maskMM_final' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', MM.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = LA
gfile.close()

pf = '_labelALL_final_maskMM_final'
elsize = [0.05, 0.0511, 0.0511]
axislabels = 'zyx'
field = 'stack'

infiles = glob.glob(os.path.join(datadir, "{}{}*.h5".format(dset_name, pf)))
for fname in infiles:
    try:
        f = h5py.File(fname, 'a')
        f[field].attrs['element_size_um'] = elsize
        for i, l in enumerate(axislabels):
            f[field].dims[i].label = l
        f.close()
        print("%s done" % fname)
    except:
        print(fname)
