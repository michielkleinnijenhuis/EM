import os
import h5py
import numpy as np
from skimage.measure import regionprops

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
dset_name = 'M3S1GNUds7'

labelvolume = ['_maskDS', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
DS = f[labelvolume[1]][:, :, :]
f.close()
labelvolume = ['_labelMA_2D_proofread', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
MA = f[labelvolume[1]][:, :, :]
f.close()
labelvolume = ['_labelMM_ws', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
MM = f[labelvolume[1]][:, :, :]
f.close()
labelvolume = ['_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1_amax_nb', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
UA = f[labelvolume[1]][:, :, :]
f.close()

mask = MM != 0
maskUA = UA != 0
maskALL = np.logical_or(mask, maskUA)


gname = dset_name + '_maskUA' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', UA.shape,
                             dtype='uint8', compression='gzip')

outds[:,:,:] = maskUA
gfile.close()

gname = dset_name + '_maskALL' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', UA.shape,
                             dtype='uint8', compression='gzip')

outds[:,:,:] = maskALL
gfile.close()



ulabels_UA = np.unique(UA)
maxlabel_UA = np.amax(ulabels_UA)
ulabels_MA = np.unique(MA)
maxlabel_MA = np.amax(ulabels_MA)

UA[maskUA] = UA[maskUA] + maxlabel_MA

ALL = UA + MM

gname = dset_name + '_labelALL' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', UA.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = ALL
gfile.close()






rpUA = regionprops(UA)

areas_UA = [prop.area for prop in rpUA]


rpDS = regionprops(DS)
rpMA = regionprops(MA)
rpMM = regionprops(MM)
MF = MA + MM
rpMF = regionprops(MF)

elsize = [0.05, 0.0073, 0.0073]
voxvol = np.prod(elsize)

DSareas = np.array([prop.area for prop in rpDS])
MAareas = np.array([prop.area for prop in rpMA])
MMareas = np.array([prop.area for prop in rpMM])
MFareas = np.array([prop.area for prop in rpMF])

DSvol = np.sum(DSareas * voxvol)
MAvol = np.sum(MAareas * voxvol)
MMvol = np.sum(MMareas * voxvol)


MAmean = np.mean(MAareas * voxvol)
MAstd = np.std(MAareas * voxvol)

MMmean = np.mean(MMareas * voxvol)
MMstd = np.std(MMareas * voxvol)


count_maskDS = rpDS[0].area
count_label_MA = np.sum(labels != 0) # labelMA_2D_proofread: 82105067
