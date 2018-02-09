import os
import h5py
import SimpleITK as sitk
import numpy as np
from skimage import morphology
import scipy.ndimage as nd
from skimage.morphology import disk

# cylinder
mask = np.tile(disk(50), [430,1,1])
sitk_mask = sitk.GetImageFromArray(mask)
sitk.WriteImage(sitk_mask, "cyl.nii.gz")
res = sitk.BinaryThinning(sitk_mask)
sitk.WriteImage(res, "cyl_skel.nii.gz")

# axons
datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
dset_name = 'M3S1GNUds7'

labelvolume = ['_labelMF_final', 'stack']
lname = dset_name + labelvolume[0] + '.h5'
lpath = os.path.join(datadir, lname)
lfile = h5py.File(lpath, 'r')
lstack = lfile[labelvolume[1]]
labels = lstack[:,:,:]
lfile.close()

labels = labels[:, 500:601, 500:601]
mask = labels != 0

sitk_img = sitk.GetImageFromArray(labels)
sitk.WriteImage(sitk_img, "img.nii.gz")
nb_labels = len(np.unique(labels))
if nb_labels > 1:
    sizes = np.bincount(labels.flatten(),
                        minlength=nb_labels+1)
    sizes[0] = 0
    i_max = np.argmax(sizes)
    mask[labels != i_max] = 0

sitk_mask = sitk.GetImageFromArray(mask.astype('uint8'))
sitk.WriteImage(sitk_mask, "mask.nii.gz")
res = sitk.BinaryThinning(sitk_mask)
sitk.WriteImage(res, "res.nii.gz")
