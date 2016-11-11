### convert .nii.gz segmentation to .h5 ###
import os
import numpy as np
import nibabel as nib
import h5py

def loadh5(datadir, dname, fieldname='stack'):
    """"""
    f = h5py.File(os.path.join(datadir, dname), 'r')
    if len(f[fieldname].shape) == 2:
        stack = f[fieldname][:,:]
    if len(f[fieldname].shape) == 3:
        stack = f[fieldname][:,:,:]
    if len(f[fieldname].shape) == 4:
        stack = f[fieldname][:,:,:,:]
    if 'element_size_um' in f[fieldname].attrs.keys():
        element_size_um = f[fieldname].attrs['element_size_um']
    else:
        element_size_um = None
    f.close()
    return stack, element_size_um

def writeh5(stack, datadir, fp_out, fieldname='stack', dtype='uint16', element_size_um=None):
    """"""
    g = h5py.File(os.path.join(datadir, fp_out), 'w')
    g.create_dataset(fieldname, stack.shape, dtype=dtype, compression="gzip")
    g[fieldname][:,:,:] = stack
    if element_size_um.any():
        g[fieldname].attrs['element_size_um'] = element_size_um
    g.close()

datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU'
dataset='m000_01000-02000_01000-02000_00030-00460'
pf = '_probs_ws_MA_probs_ws_MA_manseg'

_, elsize = loadh5(datadir, dataset + '.h5')
img = nib.load(os.path.join(datadir, dataset + pf + '.nii.gz'))
data = img.get_data()
writeh5(np.transpose(data), datadir, dataset + pf + '.h5', element_size_um=elsize)
#rsync -avz /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_01000-02000_01000-02000_00000-00430_segman.h5 ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/testblock/
#rsync -avz /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_01000-02000_02000-03000_00030-00460_probs_ws_MA_probs_ws_MA_manseg.h5 ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/
