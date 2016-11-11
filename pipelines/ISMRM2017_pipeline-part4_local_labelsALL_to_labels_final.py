### seperate into compartments (create labelimages and masks)
# MA and MM from np.unique(labelsMM)
# GL are these:
# ulabels = [4709, 6178, 5118, 3976, 5191, 8883, 5349, 4194, 3961, 9084, 4097, 4709, 5744, 8680, 8962, 4539, 4047, 8262, 4539, 4047, 8257, 4284, 5071, 9263, 4815, 8055, 8056, 5192, 5745, 3977]
# UA is the rest
import os
import h5py
import numpy as np

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
dset_name = 'M3S1GNUds7'

labelvolume = ['_labelALL_final', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
labels = f[labelvolume[1]][:, :, :]
f.close()

maxlabel = np.amax(labels)
ulabels = np.unique(labels)

# MF
labelvolume = ['_labelMF', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
MF = f[labelvolume[1]][:, :, :]
f.close()
ulabelsMF = np.unique(MF)
fw = [l if l in ulabelsMF else 0
      for l in range(0, maxlabel + 1)]

labelsMF = np.array(fw)[labels]

gname = dset_name + '_labelMF_final' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labelsMF.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = labelsMF
gfile.close()
# # MM
# labelvolume = ['_labelMM_ws', 'stack']
# f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
# MM = f[labelvolume[1]][:, :, :]
# f.close()
# ulabelsMM = np.unique(MM)
# fw = [l if l in ulabelsMM else 0
#       for l in range(0, maxlabel + 1)]
#
# labelsMM = np.array(fw)[labels]
#
# gname = dset_name + '_labelMM_final' + '.h5'
# gpath = os.path.join(datadir, gname)
# gfile = h5py.File(gpath, 'w')
# outds = gfile.create_dataset('stack', labelsMM.shape,
#                              dtype='uint32', compression='gzip')
#
# outds[:,:,:] = labelsMM
# gfile.close()

# GL
ulabelsGL = [4709, 6178, 5118, 3976, 5191, 8883, 5349, 4194, 3961, 9084, 4097, 4709, 5744, 8680, 8962, 4539, 4047, 8262, 4539, 4047, 8257, 4284, 5071, 9263, 4815, 8055, 8056, 5192, 5745, 3977]
fw = [l if l in ulabelsGL else 0
      for l in range(0, maxlabel + 1)]

labelsGL = np.array(fw)[labels]

gname = dset_name + '_labelGL_final' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labelsGL.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = labelsGL
gfile.close()

# UA
ulabelsMM = np.unique(labelsMM)
ulabelsUA = set(ulabels) - (set(ulabelsMM) | set(ulabelsGL))

fw = [l if l in ulabelsUA else 0
      for l in range(0, maxlabel + 1)]

labelsUA = np.array(fw)[labels]

gname = dset_name + '_labelUA_final' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labelsUA.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = labelsUA
gfile.close()

# MA unchanged
labelvolume = ['_labelMA_aux', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
labelsMA = f[labelvolume[1]][:, :, :]
f.close()

gname = dset_name + '_labelMA_final' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labelsMA.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = labelsMA
gfile.close()

# MM unchanged
labelvolume = ['_labelMM_ws', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
labelsMM = f[labelvolume[1]][:, :, :]
f.close()

gname = dset_name + '_labelMM_final' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labelsMM.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = labelsMM
gfile.close()


# masks
gname = dset_name + '_maskMA_final' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labelsMA.shape,
                             dtype='uint8', compression='gzip')

outds[:,:,:] = labelsMA != 0
gfile.close()

gname = dset_name + '_maskMM_final' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labelsMM.shape,
                             dtype='uint8', compression='gzip')

outds[:,:,:] = labelsMM != 0
gfile.close()

gname = dset_name + '_maskMF_final' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labelsMF.shape,
                             dtype='uint8', compression='gzip')

outds[:,:,:] = labelsMF != 0
gfile.close()

gname = dset_name + '_maskGL_final' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labelsGL.shape,
                             dtype='uint8', compression='gzip')

outds[:,:,:] = labelsGL != 0
gfile.close()

gname = dset_name + '_maskUA_final' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labelsUA.shape,
                             dtype='uint8', compression='gzip')

outds[:,:,:] = labelsUA != 0
gfile.close()

for comp in 'MA' 'MM' 'MF' 'UA' 'GL'; do
pf=_label${comp}_final
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &
pf=_mask${comp}_final
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &
done
