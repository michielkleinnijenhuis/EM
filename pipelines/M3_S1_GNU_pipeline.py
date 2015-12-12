### convert registered series to h5 stack
# scriptdir="$HOME/workspace/EM"
# datadir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU"
# python $scriptdir/EM_series2stack.py \
# "${datadir}/tifs/reg" "${datadir}/m000.h5" \
# -f 'stack' -r "*_m000.tif" -o -e 0.0073 0.0073 0.05
# python $scriptdir/EM_stack2stack.py \
# "$datadir/${dataset}.h5" \
# "$datadir/${dataset}.nii.gz"

### get a cutout of the data
# dataset='m000'
# python $scriptdir/EM_stack2stack.py \
# "$datadir/${dataset}.h5" \
# "$datadir/${dataset}_cutout00.h5" \
# -x 0 -X 500 -y 0 -Y 500 -m ".nii"
# python $scriptdir/EM_stack2stack.py \
# "$datadir/${dataset}.h5" \
# "$datadir/${dataset}_cutout01.h5" \
# -x 1000 -X 1500 -y 1000 -Y 1500 -m ".nii"


### convert nifti to h5 (for dataset and seg)
# import os
# import h5py
# import numpy as np
# import nibabel as nib
#
# datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/tifs/reg'
# dataset = 'reg_cutout01'
# # dataset = 'reg_cutout01_seg'
#
# img = nib.load(os.path.join(datadir, dataset + '.nii'))
# stack = img.get_data()
# stack = np.transpose(stack)
# g = h5py.File(os.path.join(datadir, dataset + '.h5'), 'w')
# dset = g.create_dataset('stack', stack.shape, dtype='int16')
# dset.attrs['element_size_um'] = [0.05,0.0073,0.0073]
# for i,l in enumerate('zyx'):
#     dset.dims[i].label = l
#
# g['stack'][:,:,:] = stack
# g.close()
#
# python $scriptdir/EM_stack2stack.py \
# "$datadir/${dataset}.h5" \
# "$datadir/${dataset}_view.nii.gz" \
# -e 0.05 0.0073 0.0073 \
# -d 'int32' -f 'stack'
# python $scriptdir/EM_stack2stack.py \
# "$datadir/${dataset}_seg.h5" \
# "$datadir/${dataset}_seg_view.nii.gz" \
# -e 0.05 0.0073 0.0073 \
# -d 'int32' -f 'stack'



scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU"
# dataset='m000'
dataset='m000_cutout01'
pixprob_trainingset="pixprob_training"

### cutout trainingdata
# exported data stack as HDF5 from Fiji
python $scriptdir/EM_stack2stack.py \
"${datadir}/${dataset}.h5" \
"${datadir}/${dataset}_cutout01.h5" \
-x 1000 -X 1500 -y 1000 -Y 1500 -f 'stack' -s 100 100 100 -i 'zyx'
# exported seg stack as HDF5 from Matlab

python $scriptdir/EM_stack2stack.py \
"${datadir}/${dataset}.h5" \
"${datadir}/${dataset}.nii.gz" -i 'zyx' -l 'xyz'
python $scriptdir/EM_stack2stack.py \
"${datadir}/${dataset}_seg.h5" \
"${datadir}/${dataset}_seg.nii.gz" -i 'zyx' -l 'xyz'

### run the Ilastik classifier on the dataset
source activate ilastik-devel
CONDA_ROOT=`conda info --root`
${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project="$DATA/P01/EM/M3/M3_S1_GNU/pipeline_test/${pixprob_trainingset}.ilp" \
--output_axis_order=xyzc \
--output_format=hdf5 \
--output_filename_format="${datadir}/${dataset}_probs.h5" \
--output_internal_path=/volume/predictions \
"${datadir}/${dataset}.h5/stack"

python $scriptdir/EM_stack2stack.py \
"${datadir}/${dataset}_probs.h5" \
"${datadir}/${dataset}_probs.nii.gz" \
-d 'float' -f 'volume/predictions' -i 'zyxc' -l 'xyzc'

### run EED on probabilities (Matlab)
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_eed2.h5" \
"$datadir/${dataset}_eed2.nii.gz"  -i 'zyx' -l 'xyz'
layer=3
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs${layer}_eed2.h5" \
"$datadir/${dataset}_probs${layer}_eed2.nii.gz"  -i 'zyx' -l 'xyz'

### watershed segmentations
import os
import h5py
import numpy as np
import nibabel as nib
from skimage.morphology import watershed, remove_small_objects, dilation, erosion, square
from skimage.segmentation import random_walker, relabel_sequential
from scipy.ndimage.measurements import label
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import medial_axis
from scipy.ndimage import find_objects

datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU'
dataset = 'm000_cutout01'

### load the dataset and segmentation slice
data = loadh5(datadir, dataset + '.h5')
data_smooth = gaussian_filter(data,[0.146,1,1])
writeh5(data_smooth, datadir, dataset + '_smooth.h5')
segm = loadh5(datadir, dataset + '_seg.h5')

### load the probabilities
prob_myel = loadh5(datadir, dataset + '_probs0_eed2.h5')
prob_axon = loadh5(datadir, dataset + '_probs1_eed2.h5')
prob_mito = loadh5(datadir, dataset + '_probs2_eed2.h5')
prob_memb = loadh5(datadir, dataset + '_probs3_eed2.h5')

### simplest watershed on data with manual segmentation slice
ws = watershed(-data, segm)
ws, _, _ = relabel_sequential(ws)
writeh5(ws, datadir, dataset + '_wsseg.h5')
python $scriptdir/EM_stack2stack.py \
"${datadir}/${dataset}_wsseg.h5" \
"${datadir}/${dataset}_wsseg.nii.gz" -i 'zyx' -l 'xyz'

# seeds_axon = loadh5(datadir, dataset + '_probs_wsseeds.h5')
seeds_axon = segm
seeds_axon[seeds_axon<2000] = 0
seedslice = np.zeros(seeds_axon.shape[1:])
for label in np.unique(seeds_axon)[1:]:
    binim = seeds_axon[50,:,:]==label
    mask = erosion(binim, square(5))
    seedslice[mask] = label

seeds_axon[51,:,:] = seedslice

prob=-data_smooth
ws = watershed(prob, seeds_axon, mask=~np.logical_or(ws_myel,MA))
ws, _, _ = relabel_sequential(ws)
writeh5(ws, datadir, dataset + '_probs_wssegmentation_axon.h5')

medaxim = np.zeros_like(ws,dtype='int32')
for label in np.unique(ws):
    binim = ws==label
    medax = medial_axis(binim)
    medaxim[medax] = label



### combine the MM, MA and UA segmentations
MM = loadh5(datadir, dataset + '_probs_wssegmentation_myelin.h5')
MA = loadh5(datadir, dataset + '_probs_wsseeds_myelin.h5')
UA = loadh5(datadir, dataset + '_probs_wssegmentation_axon.h5')
segm = np.zeros_like(MM)
nlabels = 0
for seg in [MM,MA,UA]:  # TODO: use forward map? # TODO: label in specific ranges? # TODO: specific hierarchy?
    segmentation, _, _ = relabel_sequential(seg, nlabels)
    nlabels = len(np.unique(segmentation))
writeh5(segmentation, datadir, dataset + '_segmentation.h5')






python $scriptdir/EM_stack2stack.py \
"${datadir}/${dataset}_smooth.h5" \
"${datadir}/${dataset}_smooth.nii.gz" -i 'zyx' -l 'xyz'

python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_wsseeds.h5" \
"$datadir/${dataset}_probs_wsseeds.nii.gz" -i 'zyx' -l 'xyz'
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_wssegmentation.h5" \
"$datadir/${dataset}_probs_wssegmentation.nii.gz" -i 'zyx' -l 'xyz'
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_wsseeds_myelin.h5" \
"$datadir/${dataset}_probs_wsseeds_myelin.nii.gz" -i 'zyx' -l 'xyz'
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_wsdistance_myelin.h5" \
"$datadir/${dataset}_probs_wsdistance_myelin.nii.gz" -i 'zyx' -l 'xyz'
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_wssegmentation_myelin.h5" \
"$datadir/${dataset}_probs_wssegmentation_myelin.nii.gz" -i 'zyx' -l 'xyz'
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_wssegmentation_axon.h5" \
"$datadir/${dataset}_probs_wssegmentation_axon.nii.gz" -i 'zyx' -l 'xyz'

python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_segmentation.h5" \
"$datadir/${dataset}_segmentation.nii.gz" -i 'zyx' -l 'xyz'
