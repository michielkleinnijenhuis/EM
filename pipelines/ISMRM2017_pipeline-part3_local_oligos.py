
# labeling/watershedding additional space not yet filled
import os
import h5py
import numpy as np
from skimage.measure import regionprops, label

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
dset_name = 'M3S1GNUds7'

labelvolume = ['_maskALL', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
MM = f[labelvolume[1]][:, :, :].astype('bool')
f.close()

labelvolume = ['_maskDS', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
DS = f[labelvolume[1]][:, :, :].astype('bool')
f.close()

mask = ~MM & DS
labels = label(mask)

gname = dset_name + '_maskALLlabeled' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labels.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = labels
gfile.close()

labels_all = np.copy(labels)

from skimage.morphology import remove_small_objects
remove_small_objects(labels, 10000, in_place=True)

gname = dset_name + '_maskALLlabeled_large' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labels.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = labels
gfile.close()

labels_small = labels_all - labels

gname = dset_name + '_maskALLlabeled_small' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labels.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = labels_small
gfile.close()






for pf in "_labelALL"; do
    for datastem in ${datastems[@]}; do
        python $scriptdir/convert/EM_stack2stack.py \
        $datadir/${dataset}${dspf}${ds}${pf}.h5 $datadir/${datastem}${pf}.nii \
        -e $xe $ye $ze -i 'zyx' -l 'xyz' -p $datastem -b $xo $yo $zo -d uint32 &
    done
done

# vessel:
244 in '_labelALL_large'

# oligo's
block01
tiny body: 4709
tiny process: 6178
5118 8500 8581 9871 9695 12545 12089 13985 15396 17845 19408 18777 19747 18101 4684

block02
3976 3990 3921 4088
5191 4272 6398 4928 4069 4451 4472 4351 4707 5146 4807 4029 5253 7017
5118 12089 13537 15921 19006 19532 19894 17845 18777 4684 24292 24924
10144 31719 9263 27933 8883 10516
5349 8318 7867 7779 9699 9064 9605 8485 5410 5542 10271 6116 6001 5393 5410 5542 \
4397 9747 5393 21109 5078 4406 4026

block03
4194 4034 3967 5324 5703 7066 4141
3961
9084 12804
BV: 28949

block04
4097 4534 4362 4379 4235 4197 4188 4194 4549 5110 5801 7198 6408 7576 8899
4709 4246 9077 12766 13358 13263 13936 13736 13008
5744 12704 12863 13230 12863 13225 14826 16828 16708 16375 17434 18920 19128 21462 22346 23031 8499
8680 4255

block05
5191
4709 11190 13736 13936 13008
5744 9952 11145 13230 12788 13230 13696 14765 15306 16028 17633 18238 20518 19512 \
21402 20753 20649 5744 21402 20753 20883 21402
BV: 19683 8054 5191 23967
10144 8962 28174 29230 31352 31252

block06
BV: 4539 12996 6334 7227
21267 20270 4539 12744

block07
4047 4398 4343 6509
8257 24594 29679

block08
4047 4398 4098 4249 4343
8257 24594 19764 20944 21035 25346 24173 25972 25022 27945

block09
4284 4709 4367 4017
5071 17121 18675 5071 15605 17326
9263
4815

### merge labels oligo's and blood vessel
python $scriptdir/supervoxels/delete_labels.py \
$datadir $datastem \
-L "_labelALL" 'stack' \
-o "_proofread" 'stack' \
-m "$datadir/${datastem}_labelALL_manumerge.txt" -O

# outcome oligodendrocytes / BV labels:
# 4709 6178 5118 3976 5191 8883 5349 4194 3961 9084 4097 4709 5744 8680 8962 4539 4047 8262 4539 4047 8257 4284 5071 9263 4815 8055



# create mask with only oligo's
import os
import h5py
import numpy as np
from skimage.measure import regionprops, label

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
dset_name = 'M3S1GNUds7'

labelvolume = ['_labelALL_proofread', 'stack']
lname = dset_name + labelvolume[0] + '.h5'
lpath = os.path.join(datadir, lname)
lfile = h5py.File(lpath, 'r')
lstack = lfile[labelvolume[1]]
labels = lstack[:,:,:]
lfile.close()

ulabels = [4709, 6178, 5118, 3976, 5191, 8883, 5349, 4194, 3961, 9084, 4097, 4709, 5744, 8680, 8962, 4539, 4047, 8262, 4539, 4047, 8257, 4284, 5071, 9263, 4815, 8055]

mask = np.zeros_like(labels, dtype='bool')
for l in ulabels:
    print(l)
    mask = mask | (labels == l)


## add labels from the connected_component image _maskALLlabeled_large to labels and mask
labelvolume = ['_maskALLlabeled_large', 'stack']
lname = dset_name + labelvolume[0] + '.h5'
lpath = os.path.join(datadir, lname)
lfile = h5py.File(lpath, 'r')
lstack = lfile[labelvolume[1]]
labels_large = lstack[:,:,:]
lfile.close()


labelsets = {}
labelsets[8056] = set([244])
labelsets[5192] = set([807, 721, 601])
labelsets[5745] = set([19516])
labelsets[3977] = set([43284, 61751, 38072])

for lsk, lsv in labelsets.items():
    lsmask = np.zeros_like(labels, dtype='bool')
    for l in lsv:
        lsmask = lsmask | (labels_large == l)
    mask = mask | lsmask
    labels[lsmask] = lsk

print(np.sum(mask))

gname = dset_name + '_maskGL' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labels.shape,
                             dtype='uint8', compression='gzip')

outds[:,:,:] = mask
gfile.close()

gname = dset_name + '_labelALL_proofread_addlarge' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labels.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = labels
gfile.close()

labels[~mask] = 0

gname = dset_name + '_labelGL' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labels.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = labels
gfile.close()


# now watershed the empty space in '_labelALL_proofread_addlarge' and re-extract the compartments
import os
import h5py
import numpy as np
from skimage.measure import regionprops, label
from skimage.morphology import watershed

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
dset_name = 'M3S1GNUds7'

# update maskALL to check if we're good
labelvolume = ['_labelALL_proofread_addlarge', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
mask = f[labelvolume[1]][:, :, :].astype('bool')
f.close()
gname = dset_name + '_maskALL' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', mask.shape,
                             dtype='uint8', compression='gzip')

outds[:,:,:] = mask
gfile.close()

# going to get some myelinated axon labels from 'maskALLlabeled_large' first
# 3894: 1669
# 2029: 1714
# 1484: 1484
# etc # no, too much for now

labelvolume = ['_labelALL_proofread_addlarge', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
labels = f[labelvolume[1]][:, :, :]
f.close()

labelvolume = ['', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
data = f[labelvolume[1]][:, :, :]
f.close()

labelvolume = ['_maskDS', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
maskDS = f[labelvolume[1]][:, :, :]
f.close()

mask = ~mask & maskDS  # mask is ok
# seeds do not overlap the empty space, need dilation
from scipy.ndimage.morphology import grey_dilation
labelsdil = grey_dilation(labels, size=(3, 3, 3))

ws = watershed(data, labelsdil, mask=mask)

gname = dset_name + '_labelALL_proofread_addlarge_wsempty' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', mask.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = ws
gfile.close()

labels = labels + ws

gname = dset_name + '_labelALL_final' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', mask.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = labels
gfile.close()

# update and check maskALL (should be filled as maskDS) # check!
mask = labels.astype('bool')
gname = dset_name + '_maskALL' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', mask.shape,
                             dtype='uint8', compression='gzip')

outds[:,:,:] = mask
gfile.close()






# re-evaluate labelMM_ws: I'm going to take the local one

import os
import h5py
import numpy as np

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
dset_name = 'M3S1GNUds7'

labelvolume = ['_labelMM_ws', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
labelsMM = f[labelvolume[1]][:, :, :]
f.close()

labelvolume = ['_labelMA_2D_proofread', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
labelsMA = f[labelvolume[1]][:, :, :]
f.close()

maskMA = labelsMA != 0
labelsMF = np.copy(labelsMM)
labelsMF[maskMA] = labelsMA[maskMA]

gname = dset_name + '_labelMF' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labelsMF.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = labelsMF
gfile.close()

gname = dset_name + '_maskMF' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labelsMF.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = labelsMF != 0
gfile.close()

# take labelALL_final and replace all values in maskMF with labelMF
labelvolume = ['_labelALL_final', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
labels = f[labelvolume[1]][:, :, :]
f.close()

maskMF = labelsMF != 0
labels[maskMF] = labelsMF[maskMF]

gname = dset_name + '_labelALL_final' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labels.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = labels
gfile.close()

# and redo labelsALL_to_labels_final.py


### also choosing a new MA volume

# merge 2760  and 702 in ALL, MM, MF, t0.0_final, _labelMA_2D_proofread
import os
import h5py
import numpy as np

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
dset_name = 'M3S1GNUds7'

vols = ['_labelALL_final', '_labelMM_final', '_labelMA_final', '_labelMF_final', '_labelMA_2D_proofread', '_labelMA_t0.1_final_ws_l0.99_u1.00_s010']
for vol in vols:
    labelvolume = [vol, 'stack']
    f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
    labels = f[labelvolume[1]][:, :, :]
    f.close()
    mask = labels == 2760
    labels[mask] = 702
    gname = dset_name + vol + '.h5'
    gpath = os.path.join(datadir, gname)
    gfile = h5py.File(gpath, 'w')
    outds = gfile.create_dataset('stack', labels.shape, dtype='uint32', compression='gzip')
    outds[:,:,:] = labels
    gfile.close()

# nifti conversion
for comp in 'MA' 'MM' 'MF' 'ALL'; do
pf=_label${comp}_final
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &
done
pf='_labelMA_2D_proofread'
pf='_labelMA_t0.1_final_ws_l0.99_u1.00_s010'
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &


datastem="${dataset}${dspf}${ds}"
l=0.99; u=1.00; s=010;
svoxpf="_ws_l${l}_u${u}_s${s}"
t=0.1
vol2d="_labelMA_2D_proofread"
volws="_labelMA_t${t}_final${svoxpf}"

python $scriptdir/supervoxels/filter_NoR.py \
$datadir $datastem \
-l "${volws}" 'stack' \
-L "${vol2d}" 'stack' \
-o "_labelMA_filterNoR" 'stack'

# do ALLlabels_to_final (for MA only)




## add some missing label in labelMA_NoR
labelvolume = ['_labelMA_filterNoR', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
labelsMA = f[labelvolume[1]][:, :, :]
f.close()

labelvolume = ['_labelMM_final', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
labelsMM = f[labelvolume[1]][:, :, :]
f.close()

labelvolume = ['_labelMF_final', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
labelsMF = f[labelvolume[1]][:, :, :]
f.close()

ulabelsMA = np.unique(labelsMA)
ulabelsMM = np.unique(labelsMM)
ulabelsMF = np.unique(labelsMF)

set(ulabelsMM) - set(ulabelsMA)
# set([677, 2119, 393, 2215, 2997, 157, 3583])

labelvolume = ['_labelMA_2D', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
labelsMA_aux = f[labelvolume[1]][:, :, :]
f.close()

ulabelsMA_aux = np.unique(labelsMA_aux)
set(ulabelsMM) - set(ulabelsMA_aux)
set([])

for l in set(ulabelsMM) - set(ulabelsMA):
    print(l)
    mask = labelsMA_aux == l
    print(np.sum(mask))
    labelsMA[mask] = labelsMA_aux[mask]

gname = dset_name + '_labelMA_aux' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', labelsMA.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = labelsMA
gfile.close()

# and update final ...


# do stats
python $scriptdir/mesh/EM_seg_stats.py \
$datadir $datastem \
--labelMA '_labelMA_final' 'stack' \
--labelMF '_labelMF_final' 'stack' \
--labelUA '_labelUA_final' 'stack' \
--stats 'eccentricity' 'solidity'



### counts and volume
import os
import h5py
import numpy as np

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
dset_name = 'M3S1GNUds7'

labelvolume = ['_labelMA_final', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
labelsMA = f[labelvolume[1]][:, :, :]
f.close()

labelvolume = ['_labelMM_final', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
labelsMM = f[labelvolume[1]][:, :, :]
f.close()

labelvolume = ['_labelMF_final', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
labelsMF = f[labelvolume[1]][:, :, :]
f.close()

labelvolume = ['_labelUA_final', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
labelsUA = f[labelvolume[1]][:, :, :]
f.close()

labelvolume = ['_labelGL_final', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
labelsGL = f[labelvolume[1]][:, :, :]
f.close()

ulabelsMA = np.unique(labelsMA)
ulabelsMM = np.unique(labelsMM)
ulabelsMF = np.unique(labelsMF)

ulabelsUA = np.unique(labelsUA)
ulabelsGL = np.unique(labelsGL)

len(ulabelsMA)
len(ulabelsUA)
len(ulabelsGL)

np.sum(labelsMA != 0)
np.sum(labelsMF != 0)
np.sum(labelsUA != 0)




python $scriptdir/mesh/label2stl.py $datadir $datastem \
-L '_labelMA_final' -f 'stack' -o 30 0 0

python $scriptdir/mesh/label2stl.py $datadir/$datastem $dataset \
-L '_PA' -f '/stack' -n 5 -o $z $y $x \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z

for comp in MM NN GP UA; do
blender -b -P $scriptdir/mesh/stl2blender.py -- \
$datadir/dmcsurf_PAenforceECS ${comp} -L ${comp} -s 0.5 10 -d 0.2  -e 0.01
done
for comp in ECS; do
blender -b -P $scriptdir/mesh/stl2blender.py -- \
$datadir/dmcsurf ${comp} -L ${comp} -d 0.02
done

for ob in bpy.data.objects:
    ob.hide = True
