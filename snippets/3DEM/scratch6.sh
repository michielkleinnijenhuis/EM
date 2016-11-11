python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${lvol}${NoRpf}_iter${iter}.h5" \
$datadir/"${datastem}${lvol}${NoRpf}_iter${iter}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${lvol}${NoRpf}_iter${iter}_automerged.h5" \
$datadir/"${datastem}${lvol}${NoRpf}_iter${iter}_automerged.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &




datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
dset_name = 'M3S1GNUds7'

labelvolume = ['_labelMA_core2D_fw_3Diter3_closed_proofread_closed_proofread_proofread', 'stack']
lname = dset_name + labelvolume[0] + '.h5'
lpath = os.path.join(datadir, lname)
lfile = h5py.File(lpath, 'r')
lstack = lfile[labelvolume[1]]
labels = lstack[:,:,:]
lfile.close()

ulabels = [779, 2106, 2420, 1385, 1386, 2776, 2802, 439, 1935, 2117, 1634, 2015, 2278, 3164,  2273, 3494, 3821, 3822]
ulabels = [2273,3494,3821,3822]
mask = np.zeros_like(labels, dtype='bool')
for l in ulabels:
    print(l)
    mask = mask | (labels == l)

print(np.sum(mask))

labels[~mask] = 0

gname = dset_name + 'oddlabels' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labels.shape,
                             dtype='uint16', compression='gzip')

outds[:,:,:] = labels
gfile.close()



import os
import h5py
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label
from skimage.morphology import watershed

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
dset_name = 'M3S1GNUds7'
dset_name = 'M3S1GNUds7_00000-00438_00419-00838_00030-00460'

maskMM = ['_maskMM', 'stack']
lname = dset_name + maskMM[0] + '.h5'
lpath = os.path.join(datadir, lname)
lfile = h5py.File(lpath, 'r')
lstack = lfile[maskMM[1]]
maskMM = lstack[:,:,:].astype('bool')
lfile.close()

maskMX = ['_maskMM-0.02', 'stack']
lname = dset_name + maskMX[0] + '.h5'
lpath = os.path.join(datadir, lname)
lfile = h5py.File(lpath, 'r')
lstack = lfile[maskMX[1]]
maskMX = lstack[:,:,:].astype('bool')
lfile.close()

maskDS = ['_maskDS', 'stack']
lname = dset_name + maskDS[0] + '.h5'
lpath = os.path.join(datadir, lname)
lfile = h5py.File(lpath, 'r')
lstack = lfile[maskDS[1]]
maskDS = lstack[:,:,:].astype('bool')
lfile.close()

labelvolume = ['_labelMA_core2D_fw_3Diter3', 'stack']
lname = dset_name + labelvolume[0] + '.h5'
lpath = os.path.join(datadir, lname)
lfile = h5py.File(lpath, 'r')
lstack = lfile[labelvolume[1]]
labels = lstack[:,:,:]
lfile.close()


mask1 = ~maskDS | maskMM
mask2 = mask1 | labels.astype('bool')
labels_mask2 = label(~mask2)

counts = np.bincount(labels_mask2.ravel())
bg = np.argmax(counts[1:]) + 1
labels_mask2[labels_mask2 == bg] = 0

mask3 = mask1 | labels_mask2 == bg
ws_mask3 = watershed(mask3, labels, mask=~mask3)


gname = dset_name + 'labels_iter1' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labels.shape,
                             dtype='uint16', compression='gzip')

outds[:,:,:] = new_labels
gfile.close()






M3S1GNUds7_labelMM_MAdilation.h5
pf=_labelMM_MAdilation
pf=_labelMM_ws
pf=_labelMM_MM_wsmask

datastem='M3S1GNUds7_00000-00438_00419-00838_00030-00460'
pf=mask
pf=mask2
pf=labels_iter1
pf=_labelMA_core2D_fw_3Diter3_filled
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &



source datastems_09blocks.sh
for pf in "_maskDS" "_maskMM" "_maskMM-0.02" "${CC2Dstem}_fw_3Diter3"; do
    for datastem in ${datastems[@]}; do
        python $scriptdir/convert/EM_stack2stack.py \
        $datadir/${dataset}${dspf}${ds}${pf}.h5 $datadir/${datastem}${pf}.h5 \
        -e $xe $ye $ze -i 'zyx' -l 'zyx' -p $datastem -b $xo $yo $zo &
    for datastem in ${datastems[@]}; do
        python $scriptdir/convert/EM_stack2stack.py \
        $datadir/${dataset}${dspf}${ds}${pf}.h5 $datadir/${datastem}${pf}.nii \
        -e $xe $ye $ze -i 'zyx' -l 'xyz' -p $datastem -b $xo $yo $zo &
    done
done

datastem=M3S1GNUds7_00000-00438_00419-00838_00030-00460
python $scriptdir/supervoxels/fill_holes.py \
$datadir $datastem -w 4 \
-L "${CC2Dstem}_fw_3Diter3" 'stack' \
--maskDS '_maskDS' 'stack' \
--maskMM '_maskMM' 'stack' \
--maskMX '_maskMM-0.02' 'stack' \
-o "_filled" 'stack'

M3S1GNUds7_00000-00438_00419-00838_00030-00460_labelMA_core2D_fw_3Diter3_filled.h5

datastem=M3S1GNUds7_00000-00438_00419-00838_00030-00460
python $scriptdir/supervoxels/fill_holes.py \
$datadir $datastem -w 4 \
-L "${CC2Dstem}_fw_3Diter3_filled" 'stack' \
--maskDS '_maskDS' 'stack' \
--maskMM '_maskMM-0.02' 'stack' \
-o "_filled" 'stack'
