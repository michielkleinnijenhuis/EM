import os
import numpy as np
import h5py
datadir = "/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3S1GNU"

filestem = "M3S1GNU"
h5path = os.path.join(datadir, filestem + '.h5')
h5file0 = h5py.File(h5path, 'a')
ds0 = h5file0['stack']

# M3S1GNU.h5 elsize (4dim)
ds0.attrs['element_size_um'] = ds0.attrs['element_size_um'][:3]

# M3S1GNU_masks.h5/maskDS elsize
filestem = "M3S1GNU_masks"
h5path = os.path.join(datadir, filestem + '.h5')
h5file1 = h5py.File(h5path, 'a')
ds = h5file1['maskDS']
ds.attrs['element_size_um'] = ds0.attrs['element_size_um']
h5file1.close()

filestem = "M3S1GNU_probs"
h5path = os.path.join(datadir, filestem + '.h5')
h5file1 = h5py.File(h5path, 'a')
ds = h5file1['volume/predictions']
ds.attrs['element_size_um'] = np.append(ds0.attrs['element_size_um'], 1)
for i, label in enumerate(ds0.attrs['DIMENSION_LABELS']):
    ds.dims[i].label = label

ds.dims[3].label = 'c'
h5file1.close()

h5file0.close()

filestem = "M3S1GNUds7"
h5path = os.path.join(datadir, filestem + '.h5')
h5file0 = h5py.File(h5path, 'a')
ds0 = h5file0['data']

# M3S1GNUds7_probs.h5/volume elsize (1 7 7 1)
filestem = "M3S1GNUds7_probs"
h5path = os.path.join(datadir, filestem + '.h5')
h5file1 = h5py.File(h5path, 'a')
ds = h5file1['volume/predictions']
ds.attrs['element_size_um'] = np.append(ds0.attrs['element_size_um'], 1)
h5file1.close()

h5file0.close()
